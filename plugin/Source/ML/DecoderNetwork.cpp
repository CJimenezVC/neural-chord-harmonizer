#include "DecoderNetwork.h"

#include <algorithm>
#include <juce_core/juce_core.h>

struct DecoderNetwork::Impl
{
    // std::unique_ptr<RTNeural::Model<float>> model;
    std::vector<float> input;   // mel ⊕ style scratch
};

DecoderNetwork::DecoderNetwork() : impl (std::make_unique<Impl>()) {}
DecoderNetwork::~DecoderNetwork() = default;

bool DecoderNetwork::loadModel (const juce::File& jsonFile)
{
    if (! jsonFile.existsAsFile())
        return false;

    // TODO: parse RTNeural JSON; read melBins/styleDim from spec.
    loaded = true;
    return loaded;
}

void DecoderNetwork::decode (const float* melFrame, const float* style, float* melOut)
{
    if (! loaded)
    {
        std::copy (melFrame, melFrame + melBins, melOut);   // pass-through
        return;
    }

    // Placeholder: identity + small style-conditioned bias. Replace with the
    // RTNeural conv/dense forward pass over (mel ⊕ style).
    for (int b = 0; b < melBins; ++b)
        melOut[b] = melFrame[b] + 0.0f * style[b % styleDim];
}
