#include "VocoderNetwork.h"

#include <algorithm>
#include <cmath>
#include <juce_core/juce_core.h>

struct VocoderNetwork::Impl
{
    // std::unique_ptr<RTNeural::ModelT<...>> model;   // conv stack + GRU + dense
    std::vector<float> gruState;   // persistent recurrent state
};

VocoderNetwork::VocoderNetwork() : impl (std::make_unique<Impl>()) {}
VocoderNetwork::~VocoderNetwork() = default;

bool VocoderNetwork::loadModel (const juce::File& jsonFile)
{
    if (! jsonFile.existsAsFile())
        return false;

    // TODO: parse RTNeural JSON; size gruState from the GRU hidden dim.
    impl->gruState.assign (64, 0.0f);
    loaded = true;
    return loaded;
}

void VocoderNetwork::reset()
{
    std::fill (impl->gruState.begin(), impl->gruState.end(), 0.0f);
}

int VocoderNetwork::synthesize (const float* melFrame, float* audioOut, int maxSamples)
{
    const int n = std::min (samplesPerFrame, maxSamples);
    if (! loaded)
    {
        std::fill (audioOut, audioOut + n, 0.0f);
        return n;
    }

    // Placeholder: deterministic mel-energy-driven signal so the chain is
    // audible before the real autoregressive GRU is wired up.
    float energy = 0.0f;
    for (int b = 0; b < melBins; ++b)
        energy += melFrame[b];
    energy = std::tanh (energy / (float) melBins);

    for (int i = 0; i < n; ++i)
        audioOut[i] = energy * 0.0f;   // silence placeholder (no synthesis yet)

    return n;
}
