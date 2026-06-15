#pragma once

#include <vector>

#include "NNModel.h"

namespace juce { class File; }

/**
    WaveRNN vocoder wrapper: a mel frame -> waveform samples. The model emits a
    256-way mu-law categorical distribution *per frame*; this wrapper decodes
    the argmax bin to an amplitude and fills the hop with a click-free ramp.
    The GRU hidden state is carried across calls (streaming); reset() clears it.

    NOTE: this is a frame-rate categorical head (one value per mel frame), not a
    sample-rate autoregressive vocoder — see VocoderNetwork.cpp for the caveat.
*/
class VocoderNetwork
{
public:
    bool loadModel (const juce::File& jsonFile);
    bool isLoaded() const noexcept { return model.isLoaded(); }

    /** Synthesize audio for one mel frame; returns sample count written. */
    int synthesize (const float* melFrame, float* audioOut, int maxSamples);

    void reset();

    void setSamplesPerFrame (int n) noexcept { samplesPerFrame = n; }

private:
    NNModel model;
    std::vector<float> probs;        // 256-way output scratch
    int   samplesPerFrame = 128;     // hop length
    float prevSample = 0.0f;         // for inter-frame ramp
    int   quantBins = 256;
};
