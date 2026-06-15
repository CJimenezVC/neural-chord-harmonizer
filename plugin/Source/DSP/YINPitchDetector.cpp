#include "YINPitchDetector.h"

#include <algorithm>
#include <cmath>

void YINPitchDetector::prepare (double sr, int bufferSize)
{
    sampleRate = sr;
    maxPeriod = std::min (maxPeriod, bufferSize / 2);
    diff.assign ((size_t) (maxPeriod + 1), 0.0f);
    cumulativeMean.assign ((size_t) (maxPeriod + 1), 0.0f);
}

void YINPitchDetector::reset()
{
    std::fill (diff.begin(), diff.end(), 0.0f);
    std::fill (cumulativeMean.begin(), cumulativeMean.end(), 0.0f);
}

std::pair<float, float> YINPitchDetector::detect (const float* buffer, int numSamples)
{
    const int maxTau = std::min (maxPeriod, numSamples / 2);

    // 1) Difference function.
    for (int tau = 0; tau <= maxTau; ++tau)
    {
        float sum = 0.0f;
        for (int i = 0; i < numSamples - tau; ++i)
        {
            const float d = buffer[i] - buffer[i + tau];
            sum += d * d;
        }
        diff[(size_t) tau] = sum;
    }

    // 2) Cumulative mean normalized difference.
    cumulativeMean[0] = 1.0f;
    float running = 0.0f;
    for (int tau = 1; tau <= maxTau; ++tau)
    {
        running += diff[(size_t) tau];
        cumulativeMean[(size_t) tau] = diff[(size_t) tau] * tau / (running + 1e-9f);
    }

    // 3) Absolute threshold: first dip below threshold.
    int tauEstimate = -1;
    for (int tau = minPeriod; tau <= maxTau; ++tau)
    {
        if (cumulativeMean[(size_t) tau] < threshold)
        {
            while (tau + 1 <= maxTau
                   && cumulativeMean[(size_t) (tau + 1)] < cumulativeMean[(size_t) tau])
                ++tau;
            tauEstimate = tau;
            break;
        }
    }

    if (tauEstimate < 0)
        return { 0.0f, 0.0f };   // unvoiced

    // 4) Parabolic interpolation around the minimum.
    float betterTau = (float) tauEstimate;
    if (tauEstimate > 0 && tauEstimate < maxTau)
    {
        const float s0 = cumulativeMean[(size_t) (tauEstimate - 1)];
        const float s1 = cumulativeMean[(size_t) tauEstimate];
        const float s2 = cumulativeMean[(size_t) (tauEstimate + 1)];
        const float denom = 2.0f * (2.0f * s1 - s2 - s0);
        if (std::abs (denom) > 1e-9f)
            betterTau += (s2 - s0) / denom;
    }

    const float f0 = (float) sampleRate / betterTau;
    const float confidence = 1.0f - cumulativeMean[(size_t) tauEstimate];
    return { f0, std::clamp (confidence, 0.0f, 1.0f) };
}
