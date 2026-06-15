#include "SpectrogramProcessor.h"

#include <cmath>

namespace
{
    constexpr float kPi = 3.14159265358979323846f;

    float hzToMel (float hz)  { return 2595.0f * std::log10 (1.0f + hz / 700.0f); }
    float melToHz (float mel) { return 700.0f * (std::pow (10.0f, mel / 2595.0f) - 1.0f); }
}

void SpectrogramProcessor::prepare (double sampleRate, int size, int nMels,
                                    float fMin, float fMax)
{
    fftSize = size;
    numBins = size / 2 + 1;
    numMels = nMels;

    fft = std::make_unique<juce::dsp::FFT> ((int) std::log2 ((double) size));

    window.resize ((size_t) size);
    for (int n = 0; n < size; ++n)
        window[(size_t) n] = 0.5f - 0.5f * std::cos (2.0f * kPi * n / (float) (size - 1));

    fftBuffer.assign ((size_t) (2 * size), 0.0f);
    buildMelFilterbank (sampleRate, fMin, fMax);
}

void SpectrogramProcessor::reset()
{
    std::fill (fftBuffer.begin(), fftBuffer.end(), 0.0f);
}

void SpectrogramProcessor::buildMelFilterbank (double sampleRate, float fMin, float fMax)
{
    melFilters.assign ((size_t) numMels, std::vector<float> ((size_t) numBins, 0.0f));

    const float melMin = hzToMel (fMin), melMax = hzToMel (fMax);
    std::vector<float> centres ((size_t) numMels + 2);
    for (int i = 0; i < numMels + 2; ++i)
    {
        const float mel = melMin + (melMax - melMin) * (float) i / (float) (numMels + 1);
        centres[(size_t) i] = melToHz (mel);
    }

    const float binHz = (float) sampleRate / (float) fftSize;
    for (int m = 0; m < numMels; ++m)
    {
        const float lo = centres[(size_t) m], ce = centres[(size_t) m + 1], hi = centres[(size_t) m + 2];
        for (int k = 0; k < numBins; ++k)
        {
            const float f = k * binHz;
            float w = 0.0f;
            if (f >= lo && f <= ce)      w = (f - lo) / (ce - lo);
            else if (f > ce && f <= hi)  w = (hi - f) / (hi - ce);
            melFilters[(size_t) m][(size_t) k] = std::max (0.0f, w);
        }
    }
}

void SpectrogramProcessor::computeMagnitude (const float* frame, int numSamples, float* magOut)
{
    const int n = std::min (numSamples, fftSize);
    std::fill (fftBuffer.begin(), fftBuffer.end(), 0.0f);
    for (int i = 0; i < n; ++i)
        fftBuffer[(size_t) i] = frame[i] * window[(size_t) i];

    fft->performRealOnlyForwardTransform (fftBuffer.data());

    for (int k = 0; k < numBins; ++k)
    {
        const float re = fftBuffer[(size_t) (2 * k)];
        const float im = fftBuffer[(size_t) (2 * k + 1)];
        magOut[k] = std::sqrt (re * re + im * im);
    }
}

void SpectrogramProcessor::magnitudeToLogMel (const float* magnitude, float* melOut) const
{
    for (int m = 0; m < numMels; ++m)
    {
        float acc = 0.0f;
        const auto& filt = melFilters[(size_t) m];
        for (int k = 0; k < numBins; ++k)
            acc += filt[(size_t) k] * magnitude[k];
        melOut[m] = std::log (acc + 1e-6f);
    }
}
