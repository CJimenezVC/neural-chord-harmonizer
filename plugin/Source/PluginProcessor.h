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
    preserving) onto the detected chord tones, one harmony voice per tone, summed
    into a choir (up to maxVoices, capped by the Polyphony control). The "Tune"
    control sweeps from natural (gentle) to tight (hard-snap auto-tune). With no
    chord detected (or the instrument below the Gate), the output is silent.

    Detector runs at 24 kHz (its trained feature rate); the voice path runs at
    the host rate.
*/
class NeuralChordHarmonizerProcessor : public juce::AudioProcessor
{
public:
    NeuralChordHarmonizerProcessor();
    ~NeuralChordHarmonizerProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported (const BusesLayout&) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Neural Chord Harmonizer"; }
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

    // --- Spectrogram feed for the editor (lock-free SPSC ring of feature columns) ---
    static constexpr int   kSpecBins  = 61;   // one log-freq bin per semitone (MIDI 36..96)
    static constexpr int   kSpecCols  = 512;  // history depth
    static constexpr float kSpecFloor = -16.0f;   // log-magnitude value used for silence
    int      getSpectrumBins() const noexcept     { return kSpecBins; }
    int      getSpectrumCapacity() const noexcept { return kSpecCols; }
    uint32_t getSpectrumWriteIndex() const noexcept { return specWrite.load (std::memory_order_acquire); }
    /** Copy column @p idx (kSpecBins floats) into @p dst. */
    void readSpectrumColumn (uint32_t idx, float* dst) const noexcept
    {
        const float* src = &specRing[(size_t) (idx % kSpecCols) * kSpecBins];
        std::copy (src, src + kSpecBins, dst);
    }

private:
    static constexpr int maxVoices = 6;       // up to 6 chord tones (e.g. a guitar chord)

    void pushSpectrumColumn (const float* bins, int n) noexcept;

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    void runDetector();                       // drains scFifo, updates chord (held)
    int  collectTargets (float voiceHz, int* midiOut) const;   // chord tones near the voice

    juce::AudioProcessorValueTreeState apvts;

    ChordDetector     chordDetector;
    PitchShifter      voices[maxVoices];      // one per harmony voice
    YINPitchDetector  voiceYin;
    Resampler         scDownsampler;          // sidechain host rate -> 24 kHz

    static constexpr double detectorRate = 24000.0;
    double hostRate = 48000.0;
    int    detectorFft = 2048, detectorHop = 512;

    SampleFifo scFifo;                        // 24 kHz instrument awaiting detection
    std::vector<float> scDownScratch;         // resampled sidechain block
    std::vector<float> detectFrame;           // one detector frame (24 kHz)
    std::vector<float> pcAct;                 // 12 activations

    std::vector<float> voiceMono, voiceOut, mixBuf;   // per-block scratch (host rate)
    std::vector<float> yinBuf;                // rolling voice window for F0
    int yinWindow = 2048, yinFill = 0;

    float chordHeld[12] = { 0 };              // peak-hold-with-decay chord state
    float voiceRatio[maxVoices] = { 0 };      // smoothed per-voice pitch ratio
    float gateLinear = 0.0f;                  // instrument noise-gate threshold (RMS)
    int   polyphony = maxVoices;              // max simultaneous detected notes
    std::atomic<int> chordMask { 0 };

    std::vector<float>    specRing;           // kSpecCols * kSpecBins feature history
    std::atomic<uint32_t> specWrite { 0 };    // monotonically increasing column index

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NeuralChordHarmonizerProcessor)
};
