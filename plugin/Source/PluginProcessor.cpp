#include "PluginProcessor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "PluginEditor.h"

namespace ParamID { constexpr auto tune = "tune"; }

AdaptiveVoiceTransformProcessor::AdaptiveVoiceTransformProcessor()
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

AdaptiveVoiceTransformProcessor::~AdaptiveVoiceTransformProcessor() = default;

juce::AudioProcessorValueTreeState::ParameterLayout
AdaptiveVoiceTransformProcessor::createParameterLayout()
{
    using namespace juce;
    AudioProcessorValueTreeState::ParameterLayout layout;
    layout.add (std::make_unique<AudioParameterFloat> (
        ParameterID { ParamID::tune, 1 }, "Tune",
        NormalisableRange<float> (0.0f, 1.0f), 0.7f));   // natural .. tight
    return layout;
}

void AdaptiveVoiceTransformProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
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

void AdaptiveVoiceTransformProcessor::releaseResources()
{
    for (auto& v : voices) v.reset();
    scFifo.reset();
}

bool AdaptiveVoiceTransformProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto& mainIn  = layouts.getMainInputChannelSet();
    const auto& mainOut = layouts.getMainOutputChannelSet();
    if (mainIn != mainOut || mainIn.isDisabled())
        return false;
    // Sidechain may be mono/stereo or disabled.
    return true;
}

bool AdaptiveVoiceTransformProcessor::loadModels (const juce::File& dir)
{
    suspendProcessing (true);
    const bool ok = chordDetector.load (dir.getChildFile ("chordnet.rtneural"),
                                        dir.getChildFile ("chord_info.json"));
    suspendProcessing (false);
    return ok;
}

void AdaptiveVoiceTransformProcessor::runDetector()
{
    // Peak-hold-with-decay so the chord sustains (notes held) even as the
    // instrument note decays, instead of flickering.
    constexpr float decay = 0.97f;          // per detector hop (~21 ms @ 24k)
    while (scFifo.size() >= detectorFft)
    {
        std::copy (scFifo.data(), scFifo.data() + detectorFft, detectFrame.begin());
        chordDetector.detect (detectFrame.data(), detectorFft, pcAct.data());
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

int AdaptiveVoiceTransformProcessor::collectTargets (float voiceHz, int* midiOut) const
{
    if (voiceHz < 60.0f)
        return 0;
    const float vMidi = 69.0f + 12.0f * std::log2 (voiceHz / 440.0f);

    // One target per active pitch class, placed at the octave nearest the voice,
    // then ordered by distance to the voice (lead first).
    int n = 0;
    for (int pc = 0; pc < 12 && n < maxVoices; ++pc)
    {
        if (chordHeld[pc] <= 0.5f) continue;
        int m = pc + 12 * (int) std::lround ((vMidi - pc) / 12.0);
        m = std::clamp (m, chordDetector.getMidiLo(), chordDetector.getMidiHi());
        midiOut[n++] = m;
    }
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (std::abs (midiOut[j] - vMidi) < std::abs (midiOut[i] - vMidi))
                std::swap (midiOut[i], midiOut[j]);
    return n;
}

void AdaptiveVoiceTransformProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                                    juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    const int numSamples = buffer.getNumSamples();
    const float tune = apvts.getRawParameterValue (ParamID::tune)->load();

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

juce::AudioProcessorEditor* AdaptiveVoiceTransformProcessor::createEditor()
{
    return new AdaptiveVoiceTransformEditor (*this);
}

void AdaptiveVoiceTransformProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    if (auto state = apvts.copyState(); state.isValid())
        if (auto xml = state.createXml())
            copyXmlToBinary (*xml, destData);
}

void AdaptiveVoiceTransformProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary (data, sizeInBytes))
        apvts.replaceState (juce::ValueTree::fromXml (*xml));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AdaptiveVoiceTransformProcessor();
}
