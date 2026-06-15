#pragma once

#include "NNModel.h"

namespace juce { class File; }

/**
    Encoder wrapper: a mel-spectrogram frame -> style vector. Backed by the
    self-contained NNModel engine that loads the exported .rtneural JSON.
*/
class EncoderNetwork
{
public:
    bool loadModel (const juce::File& jsonFile);
    bool isLoaded() const noexcept { return model.isLoaded(); }

    /** Encode @p numFrames mel frames (only the first is used in streaming
        mode) into @p styleOut (getStyleDim() values). */
    void encode (const float* mel, int numFrames, float* styleOut);

    int getStyleDim() const noexcept { return model.outputSize(); }
    int getMelBins()  const noexcept { return model.inputSize(); }

private:
    NNModel model;
};
