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

    pitchShifter.prepare (1024, 256);
    voiceYin.prepare (hostRate, yinWindow);
    scDownsampler.prepare (hostRate, detectorRate);   // ratio = host/24k

    detectorFft = chordDetector.isLoaded() ? chordDetector.getFftSize() : 2048;
    scFifo.prepare (detectorFft * 4);
    detectFrame.assign ((size_t) detectorFft, 0.0f);
    pcAct.assign (12, 0.0f);

    const int maxScOut = (int) std::ceil (samplesPerBlock * detectorRate / hostRate) + 4;
    scDownScratch.assign ((size_t) std::max (maxScOut, 64), 0.0f);

    voiceMono.assign ((size_t) samplesPerBlock, 0.0f);
    outMono.assign ((size_t) samplesPerBlock, 0.0f);
    yinBuf.assign ((size_t) yinWindow, 0.0f);
    yinFill = 0;

    std::fill (std::begin (chordSmoothed), std::end (chordSmoothed), 0.0f);
    smoothedRatio = 1.0f;
    setLatencySamples (pitchShifter.getLatencySamples());
}

void AdaptiveVoiceTransformProcessor::releaseResources()
{
    pitchShifter.reset();
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
    while (scFifo.size() >= detectorFft)
    {
        std::copy (scFifo.data(), scFifo.data() + detectorFft, detectFrame.begin());
        chordDetector.detect (detectFrame.data(), detectorFft, pcAct.data());
        int mask = 0;
        for (int i = 0; i < 12; ++i)
        {
            chordSmoothed[i] = 0.7f * chordSmoothed[i] + 0.3f * pcAct[(size_t) i];
            if (chordSmoothed[i] > 0.5f) mask |= (1 << i);
        }
        chordMask.store (mask);
        scFifo.consume (detectorHop);
    }
}

float AdaptiveVoiceTransformProcessor::chooseRatio (float voiceHz, float tune) const
{
    if (voiceHz < 60.0f)
        return 1.0f;
    const float vMidi = 69.0f + 12.0f * std::log2 (voiceHz / 440.0f);

    int best = -1;
    float bestDist = 1.0e9f;
    for (int m = chordDetector.getMidiLo(); m <= chordDetector.getMidiHi(); ++m)
    {
        if (chordSmoothed[m % 12] <= 0.5f) continue;
        const float d = std::abs ((float) m - vMidi);
        if (d < bestDist) { bestDist = d; best = m; }
    }
    if (best < 0)
        return 1.0f;

    const float targetHz = 440.0f * std::pow (2.0f, (best - 69) / 12.0f);
    const float full = targetHz / voiceHz;
    return std::pow (full, tune);          // tune 0 = no shift, 1 = full snap
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
        // reuse outMono as a scratch for the mono instrument
        for (int n = 0; n < numSamples; ++n)
        {
            float s = 0.0f;
            for (int ch = 0; ch < side.getNumChannels(); ++ch) s += side.getReadPointer (ch)[n];
            outMono[(size_t) n] = s / (float) std::max (1, side.getNumChannels());
        }
        int numOut = (int) std::floor (numSamples * detectorRate / hostRate);
        numOut = std::min (numOut, (int) scDownScratch.size());
        if (numOut > 0)
        {
            scDownsampler.process (outMono.data(), scDownScratch.data(), numOut);
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

    // Choose + smooth the pitch ratio, shift the voice.
    const float target = (conf > 0.2f) ? chooseRatio (voiceHz, tune) : 1.0f;
    smoothedRatio += 0.25f * (target - smoothedRatio);
    pitchShifter.setRatio (smoothedRatio);
    pitchShifter.process (voiceMono.data(), outMono.data(), numSamples);

    for (int ch = 0; ch < mainOut.getNumChannels(); ++ch)
        std::copy (outMono.begin(), outMono.begin() + numSamples, mainOut.getWritePointer (ch));
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
