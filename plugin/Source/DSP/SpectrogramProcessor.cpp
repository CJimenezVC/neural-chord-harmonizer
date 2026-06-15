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
    ifftBuffer.assign ((size_t) (2 * size), 0.0f);
    melLinScratch.assign ((size_t) numMels, 0.0f);
    buildMelFilterbank (sampleRate, fMin, fMax);
    buildInverseMel();
}

void SpectrogramProcessor::reset()
{
    std::fill (fftBuffer.begin(), fftBuffer.end(), 0.0f);
    std::fill (ifftBuffer.begin(), ifftBuffer.end(), 0.0f);
}

void SpectrogramProcessor::buildInverseMel()
{
    // Diagonal pseudo-inverse of the mel filterbank: each linear bin is the
    // filterbank-transpose projection of the mel energies, normalized by the
    // column energy. Good enough for resynthesis (paired with input phase).
    colNorm.assign ((size_t) numBins, 0.0f);
    for (int k = 0; k < numBins; ++k)
    {
        float s = 0.0f;
        for (int m = 0; m < numMels; ++m)
            s += melFilters[(size_t) m][(size_t) k] * melFilters[(size_t) m][(size_t) k];
        colNorm[(size_t) k] = s + 1e-8f;
    }
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

void SpectrogramProcessor::computeMagnitudeAndPhase (const float* frame, int numSamples,
                                                     float* magOut, float* phaseOut)
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
        magOut[k]   = std::sqrt (re * re + im * im);
        phaseOut[k] = std::atan2 (im, re);
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

void SpectrogramProcessor::melToFrame (const float* melLog, const float* phase, float* outFrame)
{
    // log-mel -> linear mel energies.
    for (int m = 0; m < numMels; ++m)
        melLinScratch[(size_t) m] = std::exp (melLog[m]);

    // Inverse-mel to a linear magnitude spectrum, then build the complex
    // spectrum using the supplied (input-frame) phase.
    for (int k = 0; k < numBins; ++k)
    {
        float acc = 0.0f;
        for (int m = 0; m < numMels; ++m)
            acc += melFilters[(size_t) m][(size_t) k] * melLinScratch[(size_t) m];
        const float mag = std::max (0.0f, acc) / colNorm[(size_t) k];
        ifftBuffer[(size_t) (2 * k)]     = mag * std::cos (phase[k]);
        ifftBuffer[(size_t) (2 * k + 1)] = mag * std::sin (phase[k]);
    }

    fft->performRealOnlyInverseTransform (ifftBuffer.data());
    std::copy (ifftBuffer.begin(), ifftBuffer.begin() + fftSize, outFrame);
}
