#pragma once

#include <memory>
#include <vector>

namespace juce { class File; }

/**
    RTNeural wrapper for the encoder: mel-spectrogram window -> style vector.

    The actual RTNeural model object is held behind a pimpl-style pointer so the
    heavy RTNeural template instantiation stays in the .cpp.
*/
class EncoderNetwork
{
public:
    EncoderNetwork();
    ~EncoderNetwork();

    bool loadModel (const juce::File& jsonFile);
    bool isLoaded() const noexcept { return loaded; }

    /** Encode @p numFrames mel frames (T x melBins) into @p styleOut (styleDim). */
    void encode (const float* mel, int numFrames, float* styleOut);

    int getStyleDim() const noexcept { return styleDim; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    bool loaded = false;
    int  styleDim = 64;
    int  melBins = 128;
};
