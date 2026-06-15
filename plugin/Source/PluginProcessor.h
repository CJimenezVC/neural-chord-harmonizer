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

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    StyleParams readStyleParams() const;

    juce::AudioProcessorValueTreeState apvts;

    ModelManager        modelManager;
    FeatureExtractor    featureExtractor;
    NeuralAudioProcessor neuralProcessor;

    // Fixed model inference rate; the host stream is resampled to/from this.
    static constexpr double modelSampleRate = 24000.0;

    Resampler   downsampler;   // host rate → 24 kHz
    Resampler   upsampler;     // 24 kHz   → host rate
    SampleFifo  hostInFifo;    // host-rate input awaiting downsampling
    SampleFifo  modelOutFifo;  // 24 kHz vocoder output awaiting upsampling

    CircularAudioBuffer inputBuffer;    // 24 kHz analysis-frame ring
    OverlapAddBuffer    outputBuffer;   // 24 kHz overlap-add reconstruction

    double hostSampleRate = 48000.0;
    int frameSize = 512;   // model-rate (24 kHz) analysis frame
    int hopSize   = 128;   // model-rate hop

    // Pre-allocated scratch buffers (audio thread must not allocate).
    std::vector<float> scratchFrame;   // one analysis frame (24 kHz)
    std::vector<float> scratchAudio;   // vocoder output for one frame (24 kHz)
    std::vector<float> downScratch;    // downsampled block (24 kHz)
    std::vector<float> olaScratch;     // hop-sized OLA drain (24 kHz)

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdaptiveVoiceTransformProcessor)
};
