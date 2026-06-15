#pragma once

#include <vector>

#include "NNModel.h"

namespace juce { class File; }

/**
    Decoder wrapper: a mel frame conditioned on the style vector -> a
    transformed mel frame. Input to the model is (mel ⊕ style); this wrapper
    builds that concatenation. Backed by the NNModel engine.
*/
class DecoderNetwork
{
public:
    bool loadModel (const juce::File& jsonFile);
    bool isLoaded() const noexcept { return model.isLoaded(); }

    /** Decode one mel frame (getMelBins()) conditioned on @p style. */
    void decode (const float* melFrame, const float* style, float* melOut);

    int getMelBins()  const noexcept { return model.outputSize(); }
    int getStyleDim() const noexcept { return model.inputSize() - model.outputSize(); }

private:
    NNModel model;
    std::vector<float> input;   // mel ⊕ style scratch (inputSize())
};
