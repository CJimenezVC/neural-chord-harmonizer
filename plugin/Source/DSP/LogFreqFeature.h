#pragma once

#include <cmath>
#include <vector>

#include "SimpleFFT.h"

/**
    Log-frequency (one-bin-per-semitone) spectrogram feature for the chord
    detector. JUCE-free (SimpleFFT + a baked filterbank) so it matches the
    Python training feature exactly and is standalone-testable.

    Mirrors training/chord_synth.py:frame_feature():
      log( filterbank @ |rfft(frame * hann)| + 1e-6 )
*/
class LogFreqFeature
{
public:
    void prepare (int fftSizeIn)
    {
        fftSize = fftSizeIn;
        numBins = fftSize / 2 + 1;
        fft.prepare (fftSize);
        window.resize ((size_t) fftSize);
        for (int n = 0; n < fftSize; ++n)
            window[(size_t) n] = 0.5f - 0.5f * std::cos (2.0f * 3.14159265358979f * n
                                                         / (float) (fftSize - 1));
        re.assign ((size_t) fftSize, 0.0f);
        im.assign ((size_t) fftSize, 0.0f);
        mag.assign ((size_t) numBins, 0.0f);
    }

    /** @p fb is row-major [nPitch][nBins] (from chord_info.json). */
    void setFilterbank (const float* fb, int nPitchIn, int nBinsIn)
    {
        nPitch = nPitchIn;
        filt.assign ((size_t) (nPitch * numBins), 0.0f);
        for (int p = 0; p < nPitch; ++p)
            for (int k = 0; k < nBinsIn && k < numBins; ++k)
                filt[(size_t) (p * numBins + k)] = fb[p * nBinsIn + k];
    }

    int getNumPitch() const noexcept { return nPitch; }

    /** Window + FFT a frame, project to log-frequency bins. Writes nPitch values. */
    void compute (const float* frame, int numSamples, float* out)
    {
        const int n = std::min (numSamples, fftSize);
        std::fill (re.begin(), re.end(), 0.0f);
        std::fill (im.begin(), im.end(), 0.0f);
        for (int i = 0; i < n; ++i) re[(size_t) i] = frame[i] * window[(size_t) i];
        fft.transform (re.data(), im.data(), false);
        for (int k = 0; k < numBins; ++k)
            mag[(size_t) k] = std::sqrt (re[(size_t) k] * re[(size_t) k]
                                       + im[(size_t) k] * im[(size_t) k]);
        for (int p = 0; p < nPitch; ++p)
        {
            const float* w = &filt[(size_t) (p * numBins)];
            float acc = 0.0f;
            for (int k = 0; k < numBins; ++k) acc += w[k] * mag[(size_t) k];
            out[p] = std::log (acc + 1e-6f);
        }
    }

private:
    SimpleFFT fft;
    int fftSize = 2048, numBins = 1025, nPitch = 0;
    std::vector<float> window, re, im, mag, filt;
};
