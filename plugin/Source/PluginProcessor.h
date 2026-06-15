#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include <vector>

#include "DSP/CircularAudioBuffer.h"
#include "DSP/FeatureExtractor.h"
#include "DSP/OverlapAddBuffer.h"
#include "DSP/Resampler.h"
#include "DSP/SampleFifo.h"
#include "ML/ModelManager.h"
#include "ML/NeuralAudioProcessor.h"
#include "Parameters/StyleInterpolation.h"

/**
    Main audio processor for Adaptive Voice Transform.

    Dual sample-rate design: the host runs at any rate (typically 48 kHz) but
    the neural models run at a fixed 24 kHz. The block is downsampled to 24 kHz,
    pushed through the hybrid DSP-neural chain (feature extraction → encoder →
    style modulation → decoder → vocoder), then upsampled back to the host rate.

    All heavy buffers are pre-allocated in prepareToPlay(); processBlock() is
    real-time safe (no allocations, no locks).
*/
class AdaptiveVoiceTransformProcessor : public juce::AudioProcessor
{
public:
    AdaptiveVoiceTransformProcessor();
    ~AdaptiveVoiceTransformProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported (const BusesLayout&) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Adaptive Voice Transform"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock&) override;
    void setStateInformation (const void*, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState& getValueTreeState() noexcept { return apvts; }
    ModelManager& getModelManager() noexcept { return modelManager; }

    /** Load RTNeural models from @p dir and re-prepare at the model's sample
        rate (read from model_info.json). Safe to call while playing. */
    bool loadModels (const juce::File& dir);

    /** UI scope taps: drain recent input ("before") / output ("after") samples.
        Single-consumer (message thread); returns the number of samples copied. */
    int readScopeBefore (float* dst, int maxSamples);
    int readScopeAfter  (float* dst, int maxSamples);

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    StyleParams readStyleParams() const;

    static constexpr int scopeFifoSize = 1 << 14;
    juce::AbstractFifo scopeBeforeFifo { scopeFifoSize };
    juce::AbstractFifo scopeAfterFifo  { scopeFifoSize };
    std::vector<float> scopeBeforeBuf, scopeAfterBuf;
    static void pushScope (juce::AbstractFifo&, std::vector<float>&, const float* src, int n);
    static int  readScope (juce::AbstractFifo&, std::vector<float>&, float* dst, int maxSamples);

    /** (Re)prepare all rate-dependent buffers from the current host/model rates. */
    void configureForRates();

    juce::AudioProcessorValueTreeState apvts;

    ModelManager        modelManager;
    FeatureExtractor    featureExtractor;
    NeuralAudioProcessor neuralProcessor;

    Resampler   downsampler;   // host rate → model rate
    Resampler   upsampler;     // model rate → host rate
    SampleFifo  hostInFifo;    // host-rate input awaiting downsampling
    SampleFifo  modelOutFifo;  // model-rate vocoder output awaiting upsampling

    CircularAudioBuffer inputBuffer;    // model-rate analysis-frame ring
    OverlapAddBuffer    outputBuffer;   // model-rate overlap-add reconstruction

    // Model inference rate; default until model_info.json overrides it on load.
    double modelSampleRate = 24000.0;
    double hostSampleRate  = 48000.0;
    int    hostBlockSize   = 0;
    int frameSize = 512;   // model-rate analysis frame
    int hopSize   = 128;   // model-rate hop

    // Pre-allocated scratch buffers (audio thread must not allocate).
    std::vector<float> scratchFrame;   // one analysis frame (24 kHz)
    std::vector<float> scratchAudio;   // vocoder output for one frame (24 kHz)
    std::vector<float> downScratch;    // downsampled block (24 kHz)
    std::vector<float> olaScratch;     // hop-sized OLA drain (24 kHz)

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdaptiveVoiceTransformProcessor)
};
