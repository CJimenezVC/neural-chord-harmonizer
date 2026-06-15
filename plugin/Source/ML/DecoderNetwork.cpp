#include "DecoderNetwork.h"

#include <algorithm>

#include <juce_core/juce_core.h>

bool DecoderNetwork::loadModel (const juce::File& jsonFile)
{
    const bool ok = model.loadFromFile (jsonFile);
    if (ok)
        input.assign ((size_t) model.inputSize(), 0.0f);
    return ok;
}

void DecoderNetwork::decode (const float* melFrame, const float* style, float* melOut)
{
    const int melBins  = getMelBins();
    const int styleDim = getStyleDim();

    // Build (mel ⊕ style) frame-wise, matching the training-time concatenation.
    std::copy (melFrame, melFrame + melBins, input.begin());
    std::copy (style, style + styleDim, input.begin() + melBins);

    model.forward (input.data(), (int) input.size(), melOut);
}
