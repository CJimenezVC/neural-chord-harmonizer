#include "PluginProcessor.h"

#include "PluginEditor.h"

namespace ParamID
{
    constexpr auto styleShift   = "styleShift";
    constexpr auto brightness   = "brightness";
    constexpr auto formantShift = "formantShift";
    constexpr auto pitchShift   = "pitchShift";
}

AdaptiveVoiceTransformProcessor::AdaptiveVoiceTransformProcessor()
    : AudioProcessor (BusesProperties()
          .withInput  ("Input",  juce::AudioChannelSet::mono(), true)
          .withOutput ("Output", juce::AudioChannelSet::mono(), true)),
      apvts (*this, nullptr, "PARAMETERS", createParameterLayout())
{
}

AdaptiveVoiceTransformProcessor::~AdaptiveVoiceTransformProcessor() = default;

juce::AudioProcessorValueTreeState::ParameterLayout
AdaptiveVoiceTransformProcessor::createParameterLayout()
{
    using namespace juce;
    AudioProcessorValueTreeState::ParameterLayout layout;

    layout.add (std::make_unique<AudioParameterFloat> (
        ParameterID { ParamID::styleShift, 1 }, "Style Shift",
        NormalisableRange<float> (0.0f, 1.0f), 0.0f));
    layout.add (std::make_unique<AudioParameterFloat> (
        ParameterID { ParamID::brightness, 1 }, "Timbral Brightness",
        NormalisableRange<float> (-1.0f, 1.0f), 0.0f));
    layout.add (std::make_unique<AudioParameterFloat> (
        ParameterID { ParamID::formantShift, 1 }, "Formant Shift",
        NormalisableRange<float> (-12.0f, 12.0f), 0.0f));
    layout.add (std::make_unique<AudioParameterFloat> (
        ParameterID { ParamID::pitchShift, 1 }, "Pitch Shift",
        NormalisableRange<float> (-24.0f, 24.0f), 0.0f));

    return layout;
}

void AdaptiveVoiceTransformProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused (samplesPerBlock);

    featureExtractor.prepare (sampleRate, frameSize, hopSize);
    neuralProcessor.prepare (sampleRate, frameSize);
    neuralProcessor.setModels (&modelManager);

    inputBuffer.prepare (frameSize, hopSize, 1);
    outputBuffer.prepare (frameSize, hopSize, 1);

    scratchFrame.assign ((size_t) frameSize, 0.0f);
    scratchAudio.assign ((size_t) frameSize, 0.0f);

    // Report processing latency to the host so it can compensate.
    setLatencySamples (frameSize + 4 * hopSize);
}

void AdaptiveVoiceTransformProcessor::releaseResources()
{
    neuralProcessor.reset();
    inputBuffer.reset();
    outputBuffer.reset();
}

bool AdaptiveVoiceTransformProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto& in  = layouts.getMainInputChannelSet();
    const auto& out = layouts.getMainOutputChannelSet();
    return in == out && (in == juce::AudioChannelSet::mono()
                         || in == juce::AudioChannelSet::stereo());
}

StyleParams AdaptiveVoiceTransformProcessor::readStyleParams() const
{
    StyleParams p;
    p.styleShift   = apvts.getRawParameterValue (ParamID::styleShift)->load();
    p.brightness   = apvts.getRawParameterValue (ParamID::brightness)->load();
    p.formantShift = apvts.getRawParameterValue (ParamID::formantShift)->load();
    p.pitchShift   = apvts.getRawParameterValue (ParamID::pitchShift)->load();
    return p;
}

void AdaptiveVoiceTransformProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                                    juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    const int numSamples = buffer.getNumSamples();
    const auto params = readStyleParams();
    neuralProcessor.setStyleParams (params);

    if (! modelManager.isLoaded())
        return;   // pass-through until models are available

    const float* in = buffer.getReadPointer (0);
    inputBuffer.push (in, numSamples);

    // Drain whole frames from the input ring buffer.
    std::vector<float>& frame = scratchFrame;     // pre-allocated
    while (inputBuffer.pop (frame.data(), frameSize, hopSize))
    {
        const auto features = featureExtractor.process (frame.data(), frameSize);
        const int produced  = neuralProcessor.processFrame (features,
                                                            scratchAudio.data(),
                                                            (int) scratchAudio.size());
        outputBuffer.add (scratchAudio.data(), produced);
    }

    // Emit reconstructed audio.
    float* out = buffer.getWritePointer (0);
    outputBuffer.read (out, numSamples);

    for (int ch = 1; ch < buffer.getNumChannels(); ++ch)
        buffer.copyFrom (ch, 0, out, numSamples);
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
