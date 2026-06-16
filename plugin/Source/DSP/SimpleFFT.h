#pragma once

#include <cmath>
#include <utility>
#include <vector>

/**
    Minimal in-place radix-2 Cooley-Tukey FFT for power-of-two sizes. JUCE-free
    on purpose: it backs both the log-frequency detector feature and the voice
    pitch shifters, and the same feature code also compiles into the pybind11
    training binding, guaranteeing identical features.

    Not the fastest FFT, but it is not the real-time bottleneck (the per-voice
    pitch shifters are), and correctness/portability matter more here.
*/
class SimpleFFT
{
public:
    void prepare (int size)
    {
        n = size;
        logN = 0;
        while ((1 << logN) < n) ++logN;

        rev.resize ((size_t) n);
        for (int i = 0; i < n; ++i)
        {
            int x = i, r = 0;
            for (int b = 0; b < logN; ++b) { r = (r << 1) | (x & 1); x >>= 1; }
            rev[(size_t) i] = r;
        }

        cosT.resize ((size_t) (n / 2));
        sinT.resize ((size_t) (n / 2));
        for (int k = 0; k < n / 2; ++k)
        {
            const double a = -2.0 * M_PI * k / n;   // forward (negative exponent)
            cosT[(size_t) k] = (float) std::cos (a);
            sinT[(size_t) k] = (float) std::sin (a);
        }
    }

    /** In-place complex FFT. inverse=true conjugates twiddles and scales 1/n. */
    void transform (float* re, float* im, bool inverse) const
    {
        for (int i = 0; i < n; ++i)
        {
            const int j = rev[(size_t) i];
            if (j > i) { std::swap (re[i], re[j]); std::swap (im[i], im[j]); }
        }

        for (int len = 2; len <= n; len <<= 1)
        {
            const int half = len >> 1;
            const int step = n / len;
            for (int i = 0; i < n; i += len)
                for (int k = 0; k < half; ++k)
                {
                    const float c = cosT[(size_t) (k * step)];
                    const float s = inverse ? -sinT[(size_t) (k * step)]
                                            :  sinT[(size_t) (k * step)];
                    const int a = i + k, b = i + k + half;
                    const float tre = re[b] * c - im[b] * s;
                    const float tim = re[b] * s + im[b] * c;
                    re[b] = re[a] - tre; im[b] = im[a] - tim;
                    re[a] += tre;        im[a] += tim;
                }
        }

        if (inverse)
        {
            const float invN = 1.0f / (float) n;
            for (int i = 0; i < n; ++i) { re[i] *= invN; im[i] *= invN; }
        }
    }

    int size() const noexcept { return n; }

private:
    int n = 0, logN = 0;
    std::vector<int> rev;
    std::vector<float> cosT, sinT;
};
