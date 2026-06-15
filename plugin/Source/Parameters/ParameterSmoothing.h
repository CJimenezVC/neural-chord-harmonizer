#pragma once

#include <cmath>

/**
    One-pole exponential smoother to remove zipper noise from parameter jumps.
    Real-time safe; no allocations.
*/
class SmoothedValue
{
public:
    void prepare (double sampleRate, float timeMs = 20.0f)
    {
        const float tau = timeMs * 0.001f;
        coeff = std::exp (-1.0f / (tau * (float) sampleRate));
    }

    void setTarget (float t) noexcept { target = t; }
    void snapTo (float v) noexcept { current = target = v; }

    float next() noexcept
    {
        current = target + coeff * (current - target);
        return current;
    }

    float getCurrent() const noexcept { return current; }

private:
    float coeff = 0.0f;
    float current = 0.0f;
    float target = 0.0f;
};
