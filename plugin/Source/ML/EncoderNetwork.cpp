#include "EncoderNetwork.h"

#include <juce_core/juce_core.h>

// NOTE: include <RTNeural/RTNeural.h> and define the concrete model type once
// the exported architecture is finalized. Kept abstract here so the project
// builds before models exist.
struct EncoderNetwork::Impl
{
    // std::unique_ptr<RTNeural::Model<float>> model;
    std::vector<float> pooled;   // scratch for global mean pooling
};

EncoderNetwork::EncoderNetwork() : impl (std::make_unique<Impl>()) {}
EncoderNetwork::~EncoderNetwork() = default;

bool EncoderNetwork::loadModel (const juce::File& jsonFile)
{
    if (! jsonFile.existsAsFile())
        return false;

    // TODO: parse RTNeural JSON, build model, read styleDim/melBins from spec.
    // impl->model = RTNeural::json_parser::parseJson<float> (jsonFile.loadFileAsString());
    loaded = true;
    return loaded;
}

void EncoderNetwork::encode (const float* mel, int numFrames, float* styleOut)
{
    if (! loaded)
    {
        std::fill (styleOut, styleOut + styleDim, 0.0f);
        return;
    }

    // Placeholder: global-mean-pool the mel then project. Replace with the
    // RTNeural forward pass over the conv stack.
    impl->pooled.assign ((size_t) melBins, 0.0f);
    for (int t = 0; t < numFrames; ++t)
        for (int b = 0; b < melBins; ++b)
            impl->pooled[(size_t) b] += mel[t * melBins + b] / (float) numFrames;

    for (int i = 0; i < styleDim; ++i)
        styleOut[i] = impl->pooled[(size_t) (i % melBins)];
}
