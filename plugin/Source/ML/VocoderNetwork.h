#pragma once

#include <memory>
#include <vector>

namespace juce { class File; }

/**
    RTNeural wrapper for the WaveRNN vocoder: a mel frame -> waveform samples.

    The GRU hidden state is carried across calls (streaming). reset() clears it,
    e.g. on transport stop or model reload.
*/
class VocoderNetwork
{
public:
    VocoderNetwork();
    ~VocoderNetwork();

    bool loadModel (const juce::File& jsonFile);
    bool isLoaded() const noexcept { return loaded; }

    /** Synthesize audio for one mel frame; returns sample count written. */
    int synthesize (const float* melFrame, float* audioOut, int maxSamples);

    void reset();   // clear recurrent state

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    bool loaded = false;
    int  melBins = 128;
    int  samplesPerFrame = 128;   // hop length
};
