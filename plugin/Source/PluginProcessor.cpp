#include "PluginProcessor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

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
    // Dev convenience: auto-load models from $AVT_MODELS_DIR if set, e.g.
    //   export AVT_MODELS_DIR=/path/to/adaptive-voice-transform/models/pretrained
    if (const char* dir = std::getenv ("AVT_MODELS_DIR"))
    {
        juce::File d (juce::String::fromUTF8 (dir));
        if (d.isDirectory())
            loadModels (d);
    }
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
    hostSampleRate = sampleRate;
    hostBlockSize  = samplesPerBlock;

    // Adopt the loaded model's rate if available; otherwise keep the default.
    if (modelManager.isLoaded())
        modelSampleRate = modelManager.info().sampleRate;

    configureForRates();
}

void AdaptiveVoiceTransformProcessor::configureForRates()
{
    if (hostSampleRate <= 0.0 || modelSampleRate <= 0.0 || hostBlockSize <= 0)
        return;

    // DSP + neural stages all operate at the model rate (read from model_info).
    featureExtractor.prepare (modelSampleRate, frameSize, hopSize);
    neuralProcessor.prepare (modelSampleRate, frameSize);
    neuralProcessor.setModels (&modelManager);

    downsampler.prepare (hostSampleRate, modelSampleRate);   // ratio = host/model
    upsampler.prepare   (modelSampleRate, hostSampleRate);   // ratio = model/host

    inputBuffer.prepare (frameSize, hopSize, 1);
    outputBuffer.prepare (frameSize, hopSize, 1);

    // Worst-case model samples per host block (largest when host rate < model).
    const double toModel = modelSampleRate / hostSampleRate;
    const int maxModelPerBlock = (int) std::ceil (hostBlockSize * toModel) + 4;

    scratchFrame.assign ((size_t) frameSize, 0.0f);
    scratchAudio.assign ((size_t) frameSize, 0.0f);
    downScratch.assign  ((size_t) std::max (maxModelPerBlock, frameSize), 0.0f);
    olaScratch.assign   ((size_t) hopSize, 0.0f);

    // Generous FIFO headroom so the audio thread never reallocates.
    hostInFifo.prepare   (hostBlockSize * 4 + 64);
    modelOutFifo.prepare (frameSize * 8 + maxModelPerBlock + 64);

    downsampler.reset();
    upsampler.reset();
    hostInFifo.reset();
    modelOutFifo.reset();

    // Latency = model-rate frame + lookahead, expressed in host samples.
    const double toHost = hostSampleRate / modelSampleRate;
    setLatencySamples ((int) std::ceil ((frameSize + 4 * hopSize) * toHost));
}

bool AdaptiveVoiceTransformProcessor::loadModels (const juce::File& dir)
{
    // Loading happens off the audio thread; suspend processing while we swap
    // models and re-prepare the rate-dependent buffers.
    suspendProcessing (true);

    const bool ok = modelManager.loadFromDirectory (dir);
    if (ok)
    {
        const double rate = modelManager.info().sampleRate;
        if (rate > 0.0)
            modelSampleRate = rate;
        configureForRates();   // no-op until prepareToPlay has set host rate/block
    }

    suspendProcessing (false);
    return ok;
}

void AdaptiveVoiceTransformProcessor::releaseResources()
{
    neuralProcessor.reset();
    inputBuffer.reset();
    outputBuffer.reset();
    downsampler.reset();
    upsampler.reset();
    hostInFifo.reset();
    modelOutFifo.reset();
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
    neuralProcessor.setStyleParams (readStyleParams());

    if (! modelManager.isLoaded())
        return;   // pass-through until models are available

    // 1) Downsample the host block to 24 kHz and queue it for analysis.
    hostInFifo.push (buffer.getReadPointer (0), numSamples);
    {
        // Produce as many 24 kHz samples as the buffered host input allows,
        // capped to the scratch size; the FIFO retains any unused remainder.
        int wantModel = (int) std::floor (hostInFifo.size() / downsampler.getSpeedRatio()) - 1;
        wantModel = std::min (wantModel, (int) downScratch.size());
        if (wantModel > 0)
        {
            const int used = downsampler.process (hostInFifo.data(), downScratch.data(), wantModel);
            hostInFifo.consume (used);
            inputBuffer.push (downScratch.data(), wantModel);
        }
    }

    // 2) Run the neural chain on every complete 24 kHz analysis frame, draining
    //    the hop-sized reconstruction into the model-rate output FIFO.
    while (inputBuffer.pop (scratchFrame.data(), frameSize, hopSize))
    {
        const auto features = featureExtractor.process (scratchFrame.data(), frameSize);
        const int produced  = neuralProcessor.processFrame (features, scratchAudio.data(),
                                                            (int) scratchAudio.size());
        // The vocoder emits exactly one hop of samples per frame (frame-rate
        // synthesis), so the chunks tile contiguously — no overlap-add
        // windowing. (OverlapAddBuffer is retained for a future frame-length
        // synthesis vocoder.)
        modelOutFifo.push (scratchAudio.data(), produced);
    }

    // 3) Upsample 24 kHz output back to the host rate, filling the block exactly.
    float* out = buffer.getWritePointer (0);
    if (modelOutFifo.size() >= upsampler.inputSamplesNeeded (numSamples))
    {
        const int used = upsampler.process (modelOutFifo.data(), out, numSamples);
        modelOutFifo.consume (used);
    }
    else
    {
        std::fill (out, out + numSamples, 0.0f);   // still filling initial latency
    }

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
