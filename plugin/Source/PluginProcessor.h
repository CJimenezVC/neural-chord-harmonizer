#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include <vector>

#include "DSP/CircularAudioBuffer.h"
#include "DSP/FeatureExtractor.h"
#include "DSP/OverlapAddBuffer.h"
#include "ML/ModelManager.h"
#include "ML/NeuralAudioProcessor.h"
#include "Parameters/StyleInterpolation.h"

/**
    Main audio processor for Adaptive Voice Transform.

    Streams input audio through the hybrid DSP-neural chain:
    feature extraction → encoder → style modulation → decoder → vocoder.
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

    CircularAudioBuffer inputBuffer;
    OverlapAddBuffer    outputBuffer;

    int frameSize = 512;
    int hopSize   = 128;

    // Pre-allocated scratch buffers (audio thread must not allocate).
    std::vector<float> scratchFrame;
    std::vector<float> scratchAudio;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdaptiveVoiceTransformProcessor)
};
