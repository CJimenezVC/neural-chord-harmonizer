#pragma once

#include <memory>
#include <vector>

namespace juce { class File; }

/**
    RTNeural wrapper for the decoder: a mel frame concatenated with the style
    vector -> a transformed mel frame.
*/
class DecoderNetwork
{
public:
    DecoderNetwork();
    ~DecoderNetwork();

    bool loadModel (const juce::File& jsonFile);
    bool isLoaded() const noexcept { return loaded; }

    /** Decode one mel frame (melBins) conditioned on @p style (styleDim). */
    void decode (const float* melFrame, const float* style, float* melOut);

    int getMelBins() const noexcept { return melBins; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    bool loaded = false;
    int  melBins = 128;
    int  styleDim = 64;
};
