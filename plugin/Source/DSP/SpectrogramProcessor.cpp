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

    fft.prepare (size);

    window.resize ((size_t) size);
    for (int n = 0; n < size; ++n)
        window[(size_t) n] = 0.5f - 0.5f * std::cos (2.0f * kPi * n / (float) (size - 1));

    reBuf.assign ((size_t) size, 0.0f);
    imBuf.assign ((size_t) size, 0.0f);
    melLinScratch.assign ((size_t) numMels, 0.0f);
    magScratch.assign ((size_t) numBins, 0.0f);
    buildMelFilterbank (sampleRate, fMin, fMax);
    buildInverseMel();
}

void SpectrogramProcessor::reset()
{
    std::fill (reBuf.begin(), reBuf.end(), 0.0f);
    std::fill (imBuf.begin(), imBuf.end(), 0.0f);
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

void SpectrogramProcessor::setMelFilterbank (const float* fb, int nMels, int nBins)
{
    if (nMels <= 0 || nBins <= 0)
        return;

    numMels = nMels;
    melFilters.assign ((size_t) nMels, std::vector<float> ((size_t) nBins, 0.0f));
    for (int m = 0; m < nMels; ++m)
        for (int k = 0; k < nBins && k < numBins; ++k)
            melFilters[(size_t) m][(size_t) k] = fb[m * nBins + k];

    melLinScratch.assign ((size_t) nMels, 0.0f);
    buildInverseMel();
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
    forwardFFT (frame, numSamples);
    for (int k = 0; k < numBins; ++k)
        magOut[k] = reBuf[(size_t) k] * reBuf[(size_t) k]
                  + imBuf[(size_t) k] * imBuf[(size_t) k];   // power (librosa default)
}

void SpectrogramProcessor::computeMagnitudeAndPhase (const float* frame, int numSamples,
                                                     float* magOut, float* phaseOut)
{
    forwardFFT (frame, numSamples);
    for (int k = 0; k < numBins; ++k)
    {
        const float re = reBuf[(size_t) k], im = imBuf[(size_t) k];
        magOut[k]   = re * re + im * im;          // power (matches librosa default)
        phaseOut[k] = std::atan2 (im, re);
    }
}

void SpectrogramProcessor::forwardFFT (const float* frame, int numSamples)
{
    const int n = std::min (numSamples, fftSize);
    std::fill (reBuf.begin(), reBuf.end(), 0.0f);
    std::fill (imBuf.begin(), imBuf.end(), 0.0f);
    for (int i = 0; i < n; ++i)
        reBuf[(size_t) i] = frame[i] * window[(size_t) i];
    fft.transform (reBuf.data(), imBuf.data(), false);
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

void SpectrogramProcessor::setInverseMel (const float* inv, int nBins, int nMels)
{
    if (nBins <= 0 || nMels <= 0)
        return;
    invMel.assign ((size_t) (numBins * numMels), 0.0f);
    for (int k = 0; k < nBins && k < numBins; ++k)
        for (int m = 0; m < nMels && m < numMels; ++m)
            invMel[(size_t) (k * numMels + m)] = inv[k * nMels + m];
    hasInvMel = true;
}

void SpectrogramProcessor::melToMagnitude (const float* melLog, float* magOut)
{
    for (int m = 0; m < numMels; ++m)
        melLinScratch[(size_t) m] = std::exp (melLog[m]);   // power mel energies

    if (hasInvMel)
    {
        for (int k = 0; k < numBins; ++k)
        {
            const float* w = &invMel[(size_t) (k * numMels)];
            float p = 0.0f;
            for (int m = 0; m < numMels; ++m)
                p += w[m] * melLinScratch[(size_t) m];       // linear power (pinv)
            magOut[k] = std::sqrt (std::max (0.0f, p));
        }
    }
    else
    {
        for (int k = 0; k < numBins; ++k)                    // diagonal fallback
        {
            float acc = 0.0f;
            for (int m = 0; m < numMels; ++m)
                acc += melFilters[(size_t) m][(size_t) k] * melLinScratch[(size_t) m];
            magOut[k] = std::sqrt (std::max (0.0f, acc) / colNorm[(size_t) k]);
        }
    }
}

void SpectrogramProcessor::magnitudeToFrame (const float* mag, const float* phase, float* outFrame)
{
    // Build the positive-frequency half, then mirror with Hermitian symmetry.
    for (int k = 0; k < numBins; ++k)
    {
        reBuf[(size_t) k] = mag[k] * std::cos (phase[k]);
        imBuf[(size_t) k] = mag[k] * std::sin (phase[k]);
    }
    for (int k = 1; k < numBins - 1; ++k)   // conj-mirror bins 1..N/2-1
    {
        reBuf[(size_t) (fftSize - k)] =  reBuf[(size_t) k];
        imBuf[(size_t) (fftSize - k)] = -imBuf[(size_t) k];
    }
    fft.transform (reBuf.data(), imBuf.data(), true);   // inverse
    for (int i = 0; i < fftSize; ++i)
        outFrame[i] = reBuf[(size_t) i];                // imaginary part ~0
}

void SpectrogramProcessor::melToFrame (const float* melLog, const float* phase, float* outFrame)
{
    melToMagnitude (melLog, magScratch.data());
    magnitudeToFrame (magScratch.data(), phase, outFrame);
}
