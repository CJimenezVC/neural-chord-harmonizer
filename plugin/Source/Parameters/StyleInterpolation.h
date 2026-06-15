#pragma once

#include <cmath>
#include <vector>

#include "VoiceTransformParameters.h"

/**
    Blends the encoder's style vector toward a target by StyleShift before
    decoding. (Brightness and formant are applied later as spectral DSP in
    NeuralAudioProcessor.)
*/
class StyleInterpolation
{
public:
    void prepare (int styleDim)
    {
        dim = styleDim;
        target.assign ((size_t) styleDim, 0.0f);
    }

    /** Set the target speaker's style vector (e.g. from a preset). */
    void setTargetStyle (const float* t)
    {
        for (int i = 0; i < dim; ++i)
            target[(size_t) i] = t[i];
    }

    /** Blend @p sourceStyle toward the target by StyleShift. (Brightness and
        formant are applied as spectral DSP in NeuralAudioProcessor.) */
    void apply (const float* sourceStyle, const StyleParams& p, float* out) const
    {
        const float a = p.styleShift;
        for (int i = 0; i < dim; ++i)
            out[i] = (1.0f - a) * sourceStyle[i] + a * target[(size_t) i];
    }

    int getDim() const noexcept { return dim; }

private:
    int dim = 64;
    std::vector<float> target;
};
