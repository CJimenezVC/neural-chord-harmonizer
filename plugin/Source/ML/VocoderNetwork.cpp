#include "VocoderNetwork.h"

#include <algorithm>
#include <cmath>

#include <juce_core/juce_core.h>

namespace
{
    // Inverse mu-law: bin index [0, bins) -> amplitude in [-1, 1].
    float muLawDecode (int bin, int bins)
    {
        const float mu = (float) (bins - 1);
        const float y = 2.0f * (float) bin / mu - 1.0f;           // [-1, 1]
        const float mag = (std::pow (1.0f + mu, std::abs (y)) - 1.0f) / mu;
        return (y < 0.0f ? -mag : mag);
    }
}

bool VocoderNetwork::loadModel (const juce::File& jsonFile)
{
    const bool ok = model.loadFromFile (jsonFile);
    if (ok)
    {
        quantBins = model.outputSize();
        probs.assign ((size_t) quantBins, 0.0f);
        prevSample = 0.0f;
    }
    return ok;
}

void VocoderNetwork::reset()
{
    model.reset();
    prevSample = 0.0f;
}

int VocoderNetwork::synthesize (const float* melFrame, float* audioOut, int maxSamples)
{
    const int n = std::min (samplesPerFrame, maxSamples);
    if (! model.isLoaded())
    {
        std::fill (audioOut, audioOut + n, 0.0f);
        return n;
    }

    // One forward pass -> 256-way mu-law distribution for this frame.
    model.forward (melFrame, model.inputSize(), probs.data());

    const int bin = (int) std::distance (probs.begin(),
                                         std::max_element (probs.begin(), probs.end()));
    const float target = muLawDecode (bin, quantBins);

    // Click-free: ramp from the previous frame's amplitude across the hop.
    for (int i = 0; i < n; ++i)
    {
        const float a = (float) (i + 1) / (float) n;
        audioOut[i] = prevSample + (target - prevSample) * a;
    }
    prevSample = target;
    return n;
}
