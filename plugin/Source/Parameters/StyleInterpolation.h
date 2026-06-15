#pragma once

#include <cmath>
#include <vector>

#include "VoiceTransformParameters.h"

/**
    Applies user parameters to the encoder's style vector before decoding:
    blends source/target style by StyleShift and bakes a brightness tilt into
    the style space. Formant/pitch shifts are applied in the DSP stages but are
    surfaced here so the decoder can condition on them.
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

    /** Blend @p sourceStyle toward the target and apply brightness tilt. */
    void apply (const float* sourceStyle, const StyleParams& p, float* out) const
    {
        const float a = p.styleShift;
        for (int i = 0; i < dim; ++i)
        {
            float v = (1.0f - a) * sourceStyle[i] + a * target[(size_t) i];
            // Brightness: tilt later style dims (higher-frequency content).
            const float tilt = 1.0f + p.brightness * (float) i / (float) dim;
            out[i] = v * tilt;
        }
    }

    int getDim() const noexcept { return dim; }

private:
    int dim = 64;
    std::vector<float> target;
};
