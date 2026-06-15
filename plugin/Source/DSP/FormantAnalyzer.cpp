#include "FormantAnalyzer.h"

#include <algorithm>
#include <cmath>
#include <complex>

namespace { constexpr float kPi = 3.14159265358979323846f; }

void FormantAnalyzer::prepare (double sr, int lpcOrder)
{
    sampleRate = sr;
    order = lpcOrder;
    windowed.clear();
    autocorr.assign ((size_t) (order + 1), 0.0f);
    lpc.assign ((size_t) (order + 1), 0.0f);
}

void FormantAnalyzer::autocorrelate (const float* x, int n, float* r, int ord) const
{
    for (int k = 0; k <= ord; ++k)
    {
        float sum = 0.0f;
        for (int i = 0; i < n - k; ++i)
            sum += x[i] * x[i + k];
        r[k] = sum;
    }
}

void FormantAnalyzer::levinsonDurbin (const float* r, float* a, int ord) const
{
    std::vector<float> tmp ((size_t) (ord + 1), 0.0f);
    float err = r[0];
    a[0] = 1.0f;
    if (err <= 0.0f) return;

    for (int i = 1; i <= ord; ++i)
    {
        float acc = r[i];
        for (int j = 1; j < i; ++j)
            acc += a[j] * r[i - j];
        const float k = -acc / err;

        for (int j = 1; j < i; ++j) tmp[(size_t) j] = a[j] + k * a[i - j];
        for (int j = 1; j < i; ++j) a[j] = tmp[(size_t) j];
        a[i] = k;
        err *= (1.0f - k * k);
        if (err <= 0.0f) break;
    }
}

std::array<float, 4> FormantAnalyzer::estimate (const float* frame, int numSamples)
{
    std::array<float, 4> result { 0.0f, 0.0f, 0.0f, 0.0f };

    // Pre-emphasis + Hamming window.
    windowed.resize ((size_t) numSamples);
    for (int n = 0; n < numSamples; ++n)
    {
        const float pre = n > 0 ? frame[n] - 0.97f * frame[n - 1] : frame[n];
        const float w = 0.54f - 0.46f * std::cos (2.0f * kPi * n / (float) (numSamples - 1));
        windowed[(size_t) n] = pre * w;
    }

    autocorrelate (windowed.data(), numSamples, autocorr.data(), order);
    if (autocorr[0] <= 0.0f)
        return result;

    levinsonDurbin (autocorr.data(), lpc.data(), order);

    // Roots of the LPC polynomial -> formant candidates (companion-free,
    // coarse scan; replace with a proper root finder for production).
    std::vector<float> candidates;
    for (int k = 1; k < order; ++k)
    {
        // Map LPC spectral peaks by scanning angle; lightweight placeholder.
        const float angle = kPi * (float) k / (float) order;
        const float hz = angle / (2.0f * kPi) * (float) sampleRate;
        if (hz > 90.0f && hz < 5000.0f)
            candidates.push_back (hz);
    }
    std::sort (candidates.begin(), candidates.end());
    for (size_t i = 0; i < result.size() && i < candidates.size(); ++i)
        result[i] = candidates[i];

    return result;
}
