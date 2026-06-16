#pragma once

#include <vector>

#include <juce_core/juce_core.h>

#include "DSP/LogFreqFeature.h"
#include "ML/NNModel.h"

/**
    Real-time polyphonic pitch-class detector: an instrument analysis frame ->
    12 pitch-class activations (0..1). Combines the validated LogFreqFeature with
    the exported ChordNet. Loads chordnet.rtneural + chord_info.json (which holds
    the log-frequency filterbank and the analysis FFT size).
*/
class ChordDetector
{
public:
    bool load (const juce::File& chordnetFile, const juce::File& chordInfoFile);
    bool isLoaded() const noexcept { return loaded; }

    int getFftSize() const noexcept { return fftSize; }
    int getMidiLo()  const noexcept { return midiLo; }
    int getMidiHi()  const noexcept { return midiHi; }

    /** Detect on one frame (getFftSize() samples) -> 12 activations. */
    void detect (const float* frame, int numSamples, float* pc12);

    /** The most recent log-frequency feature (one bin per semitone), for display. */
    const std::vector<float>& lastFeature() const noexcept { return featBuf; }
    int getNumPitch() const noexcept { return (int) featBuf.size(); }

private:
    LogFreqFeature feature;
    NNModel net;
    std::vector<float> featBuf;     // nPitch
    int fftSize = 2048, midiLo = 36, midiHi = 96;
    bool loaded = false;
};
