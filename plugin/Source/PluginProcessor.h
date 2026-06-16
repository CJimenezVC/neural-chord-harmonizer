#pragma once

#include <atomic>
#include <vector>

#include <juce_audio_processors/juce_audio_processors.h>

#include "DSP/PitchShifter.h"
#include "DSP/Resampler.h"
#include "DSP/SampleFifo.h"
#include "DSP/YINPitchDetector.h"
#include "ML/ChordDetector.h"

/**
    Chord-following vocal harmonizer / auto-tune.

    A sidechain instrument (guitar/bass/piano) drives a neural polyphonic
    pitch-class detector; the main voice input is pitch-shifted (formant-
    preserving) to the nearest tone of the detected chord. The "Tune" control
    sweeps from natural (gentle) to tight (hard-snap / T-Pain).

    Detector runs at 24 kHz (its trained feature rate); the voice path runs at
    the host rate.
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
    bool loadModels (const juce::File& dir);
    bool modelsLoaded() const noexcept { return chordDetector.isLoaded(); }

    /** 12-bit mask of currently-detected pitch classes (for the editor). */
    int getChordMask() const noexcept { return chordMask.load(); }

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    void runDetector();                       // drains scFifo, updates chordMask
    float chooseRatio (float voiceHz, float tune) const;

    juce::AudioProcessorValueTreeState apvts;

    ChordDetector     chordDetector;
    PitchShifter      pitchShifter;
    YINPitchDetector  voiceYin;
    Resampler         scDownsampler;          // sidechain host rate -> 24 kHz

    static constexpr double detectorRate = 24000.0;
    double hostRate = 48000.0;
    int    detectorFft = 2048, detectorHop = 512;

    SampleFifo scFifo;                        // 24 kHz instrument awaiting detection
    std::vector<float> scDownScratch;         // resampled sidechain block
    std::vector<float> detectFrame;           // one detector frame (24 kHz)
    std::vector<float> pcAct;                 // 12 activations

    std::vector<float> voiceMono, outMono;    // per-block scratch (host rate)
    std::vector<float> yinBuf;                // rolling voice window for F0
    int yinWindow = 2048, yinFill = 0;

    float chordSmoothed[12] = { 0 };
    float smoothedRatio = 1.0f;
    std::atomic<int> chordMask { 0 };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdaptiveVoiceTransformProcessor)
};
