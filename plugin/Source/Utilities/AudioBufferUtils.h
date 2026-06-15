#pragma once

#include <algorithm>
#include <cmath>

/** Small, header-only helpers for raw audio buffers (no JUCE dependency). */
namespace AudioBufferUtils
{
    inline void clear (float* data, int n) noexcept
    {
        std::fill (data, data + n, 0.0f);
    }

    inline float rms (const float* data, int n) noexcept
    {
        if (n <= 0) return 0.0f;
        float acc = 0.0f;
        for (int i = 0; i < n; ++i) acc += data[i] * data[i];
        return std::sqrt (acc / (float) n);
    }

    inline float peak (const float* data, int n) noexcept
    {
        float m = 0.0f;
        for (int i = 0; i < n; ++i) m = std::max (m, std::abs (data[i]));
        return m;
    }

    inline void applyGain (float* data, int n, float gain) noexcept
    {
        for (int i = 0; i < n; ++i) data[i] *= gain;
    }
}
