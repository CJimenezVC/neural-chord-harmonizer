#include "PluginProcessor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "PluginEditor.h"

namespace ParamID
{
    constexpr auto tune      = "tune";
    constexpr auto gate      = "gate";
    constexpr auto polyphony = "polyphony";
}

NeuralChordHarmonizerProcessor::NeuralChordHarmonizerProcessor()
    : AudioProcessor (BusesProperties()
          .withInput  ("Input",     juce::AudioChannelSet::stereo(), true)
          .withOutput ("Output",    juce::AudioChannelSet::stereo(), true)
          .withInput  ("Sidechain", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "PARAMETERS", createParameterLayout())
{
    if (const char* dir = std::getenv ("AVT_MODELS_DIR"))
    {
        juce::File d (juce::String::fromUTF8 (dir));
        if (d.isDirectory()) loadModels (d);
    }
}

NeuralChordHarmonizerProcessor::~NeuralChordHarmonizerProcessor() = default;

juce::AudioProcessorValueTreeState::ParameterLayout
NeuralChordHarmonizerProcessor::createParameterLayout()
{
    using namespace juce;
    AudioProcessorValueTreeState::ParameterLayout layout;

    // The slider text box routes through the parameter's getText(), so the
    // displayed precision must be set here (a Slider's setNumDecimalPlaces is
    // overridden by the APVTS attachment).
    auto threeDp = [] (float v, int) { return String (v, 3); };

    layout.add (std::make_unique<AudioParameterFloat> (
        ParameterID { ParamID::tune, 1 }, "Tune",
        NormalisableRange<float> (0.0f, 1.0f), 0.7f,
        AudioParameterFloatAttributes().withStringFromValueFunction (threeDp)));   // natural .. tight
    layout.add (std::make_unique<AudioParameterFloat> (
        ParameterID { ParamID::gate, 1 }, "Gate",
        NormalisableRange<float> (-80.0f, -10.0f), -45.0f,
        AudioParameterFloatAttributes()
            .withLabel ("dB")
            .withStringFromValueFunction ([] (float v, int) { return String (v, 3) + " dB"; })));   // instrument noise gate
    layout.add (std::make_unique<AudioParameterInt> (
        ParameterID { ParamID::polyphony, 1 }, "Polyphony", 1, maxVoices, 4));
    return layout;
}

void NeuralChordHarmonizerProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    hostRate = sampleRate;

    for (auto& v : voices) v.prepare (1024, 256);
    voiceYin.prepare (hostRate, yinWindow);
    scDownsampler.prepare (hostRate, detectorRate);   // ratio = host/24k

    detectorFft = chordDetector.isLoaded() ? chordDetector.getFftSize() : 2048;
    scFifo.prepare (detectorFft * 4);
    detectFrame.assign ((size_t) detectorFft, 0.0f);
    pcAct.assign (12, 0.0f);

    const int maxScOut = (int) std::ceil (samplesPerBlock * detectorRate / hostRate) + 4;
    scDownScratch.assign ((size_t) std::max (maxScOut, 64), 0.0f);

    voiceMono.assign ((size_t) samplesPerBlock, 0.0f);
    voiceOut.assign  ((size_t) samplesPerBlock, 0.0f);
    mixBuf.assign    ((size_t) samplesPerBlock, 0.0f);
    yinBuf.assign ((size_t) yinWindow, 0.0f);
    yinFill = 0;

    std::fill (std::begin (chordHeld), std::end (chordHeld), 0.0f);
    for (auto& r : voiceRatio) r = 1.0f;
    setLatencySamples (voices[0].getLatencySamples());
}

void NeuralChordHarmonizerProcessor::releaseResources()
{
    for (auto& v : voices) v.reset();
    scFifo.reset();
}

bool NeuralChordHarmonizerProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto& mainIn  = layouts.getMainInputChannelSet();
    const auto& mainOut = layouts.getMainOutputChannelSet();
    if (mainIn != mainOut || mainIn.isDisabled())
        return false;
    // Sidechain may be mono/stereo or disabled.
    return true;
}

bool NeuralChordHarmonizerProcessor::loadModels (const juce::File& dir)
{
    suspendProcessing (true);
    const bool ok = chordDetector.load (dir.getChildFile ("chordnet.rtneural"),
                                        dir.getChildFile ("chord_info.json"));
    suspendProcessing (false);
    return ok;
}

void NeuralChordHarmonizerProcessor::runDetector()
{
    // Peak-hold-with-decay so the chord sustains (notes held) even as the
    // instrument note decays, instead of flickering.
    constexpr float decay = 0.97f;          // per detector hop (~21 ms @ 24k)
    while (scFifo.size() >= detectorFft)
    {
        std::copy (scFifo.data(), scFifo.data() + detectorFft, detectFrame.begin());

        // Noise gate: only detect when the instrument is actually above the gate
        // (the level-invariant feature would otherwise read notes out of noise).
        float ms = 0.0f;
        for (int i = 0; i < detectorFft; ++i) ms += detectFrame[(size_t) i] * detectFrame[(size_t) i];
        const bool open = std::sqrt (ms / (float) detectorFft) >= gateLinear;

        if (open)
            chordDetector.detect (detectFrame.data(), detectorFft, pcAct.data());
        else
            std::fill (pcAct.begin(), pcAct.end(), 0.0f);   // gated -> chord decays away

        int mask = 0;
        for (int i = 0; i < 12; ++i)
        {
            chordHeld[i] = std::max (decay * chordHeld[i], pcAct[(size_t) i]);
            if (chordHeld[i] > 0.5f) mask |= (1 << i);
        }
        chordMask.store (mask);
        scFifo.consume (detectorHop);
    }
}

int NeuralChordHarmonizerProcessor::collectTargets (float voiceHz, int* midiOut) const
{
    if (voiceHz < 60.0f)
        return 0;
    const float vMidi = 69.0f + 12.0f * std::log2 (voiceHz / 440.0f);

    // Gather active pitch classes with their strength, keep the strongest up to
    // the Polyphony limit (e.g. a 6-note guitar chord).
    int   pcs[12];
    float str[12];
    int nc = 0;
    for (int pc = 0; pc < 12; ++pc)
        if (chordHeld[pc] > 0.5f) { pcs[nc] = pc; str[nc] = chordHeld[pc]; ++nc; }
    for (int i = 0; i < nc; ++i)
        for (int j = i + 1; j < nc; ++j)
            if (str[j] > str[i]) { std::swap (str[i], str[j]); std::swap (pcs[i], pcs[j]); }

    const int n = std::min ({ nc, polyphony, maxVoices });
    for (int i = 0; i < n; ++i)
    {
        int m = pcs[i] + 12 * (int) std::lround ((vMidi - pcs[i]) / 12.0);
        midiOut[i] = std::clamp (m, chordDetector.getMidiLo(), chordDetector.getMidiHi());
    }
    // Order by distance to the voice (lead first).
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (std::abs (midiOut[j] - vMidi) < std::abs (midiOut[i] - vMidi))
                std::swap (midiOut[i], midiOut[j]);
    return n;
}

void NeuralChordHarmonizerProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                                    juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    const int numSamples = buffer.getNumSamples();
    const float tune = apvts.getRawParameterValue (ParamID::tune)->load();
    gateLinear = std::pow (10.0f, apvts.getRawParameterValue (ParamID::gate)->load() / 20.0f);
    polyphony  = (int) apvts.getRawParameterValue (ParamID::polyphony)->load();

    auto mainIn  = getBusBuffer (buffer, true, 0);
    auto mainOut = getBusBuffer (buffer, false, 0);

    // Voice = mono sum of the main input.
    for (int n = 0; n < numSamples; ++n)
    {
        float s = 0.0f;
        for (int ch = 0; ch < mainIn.getNumChannels(); ++ch) s += mainIn.getReadPointer (ch)[n];
        voiceMono[(size_t) n] = s / (float) std::max (1, mainIn.getNumChannels());
    }

    // Sidechain (instrument) -> 24 kHz -> chord detector.
    if (getBus (true, 1) != nullptr && getBus (true, 1)->isEnabled())
    {
        auto side = getBusBuffer (buffer, true, 1);
        // voiceOut is free here (the voice loop overwrites it later) -> mono scratch.
        for (int n = 0; n < numSamples; ++n)
        {
            float s = 0.0f;
            for (int ch = 0; ch < side.getNumChannels(); ++ch) s += side.getReadPointer (ch)[n];
            voiceOut[(size_t) n] = s / (float) std::max (1, side.getNumChannels());
        }
        int numOut = (int) std::floor (numSamples * detectorRate / hostRate);
        numOut = std::min (numOut, (int) scDownScratch.size());
        if (numOut > 0)
        {
            scDownsampler.process (voiceOut.data(), scDownScratch.data(), numOut);
            scFifo.push (scDownScratch.data(), numOut);
        }
        runDetector();
    }

    // Voice F0 over a rolling window.
    if (numSamples >= yinWindow)
        std::copy (voiceMono.begin() + (numSamples - yinWindow),
                   voiceMono.begin() + numSamples, yinBuf.begin());
    else
    {
        std::memmove (yinBuf.data(), yinBuf.data() + numSamples,
                      sizeof (float) * (size_t) (yinWindow - numSamples));
        std::copy (voiceMono.begin(), voiceMono.begin() + numSamples,
                   yinBuf.begin() + (yinWindow - numSamples));
    }
    const auto [voiceHz, conf] = voiceYin.detect (yinBuf.data(), yinWindow);

    // Choir: one pitch-shifted voice per detected chord tone, summed.
    int targets[maxVoices];
    const int nTargets = (conf > 0.2f) ? collectTargets (voiceHz, targets) : 0;

    std::fill (mixBuf.begin(), mixBuf.begin() + numSamples, 0.0f);
    int nMixed = 0;
    for (int v = 0; v < maxVoices; ++v)
    {
        const bool active = (v < nTargets) && (voiceHz >= 60.0f);
        float rTarget = 1.0f;
        if (active)
        {
            const float targetHz = 440.0f * std::pow (2.0f, (float) (targets[v] - 69) / 12.0f);
            rTarget = std::pow (targetHz / voiceHz, tune);   // tune: 0 = dry .. 1 = full chord
        }
        voiceRatio[v] += 0.25f * (rTarget - voiceRatio[v]);  // smooth (glide)
        voices[v].setRatio (voiceRatio[v]);
        voices[v].process (voiceMono.data(), voiceOut.data(), numSamples);

        if (active)
        {
            for (int n = 0; n < numSamples; ++n) mixBuf[(size_t) n] += voiceOut[(size_t) n];
            ++nMixed;
        }
    }
    // No detected chord (or unvoiced) -> mixBuf stays zero -> output is silent.

    const float norm = nMixed > 1 ? 1.0f / std::sqrt ((float) nMixed) : 1.0f;
    for (int ch = 0; ch < mainOut.getNumChannels(); ++ch)
    {
        auto* o = mainOut.getWritePointer (ch);
        for (int n = 0; n < numSamples; ++n) o[n] = mixBuf[(size_t) n] * norm;
    }
}

juce::AudioProcessorEditor* NeuralChordHarmonizerProcessor::createEditor()
{
    return new NeuralChordHarmonizerEditor (*this);
}

void NeuralChordHarmonizerProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    if (auto state = apvts.copyState(); state.isValid())
        if (auto xml = state.createXml())
            copyXmlToBinary (*xml, destData);
}

void NeuralChordHarmonizerProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary (data, sizeInBytes))
        apvts.replaceState (juce::ValueTree::fromXml (*xml));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new NeuralChordHarmonizerProcessor();
}
