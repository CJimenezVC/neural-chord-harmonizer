#pragma once

#include <vector>

#include <juce_dsp/juce_dsp.h>

/**
    STFT magnitude and log-mel projection. Wraps juce::dsp::FFT and a
    precomputed triangular mel filterbank. All scratch buffers are allocated in
    prepare().
*/
class SpectrogramProcessor
{
public:
    void prepare (double sampleRate, int fftSize, int nMels,
                  float fMin = 20.0f, float fMax = 8000.0f);
    void reset();

    /** Window @p frame, FFT, write magnitude spectrum (fftSize/2 + 1 bins). */
    void computeMagnitude (const float* frame, int numSamples, float* magOut);

    /** Project a magnitude spectrum onto the log-mel filterbank. */
    void magnitudeToLogMel (const float* magnitude, float* melOut) const;

private:
    void buildMelFilterbank (double sampleRate, float fMin, float fMax);

    std::unique_ptr<juce::dsp::FFT> fft;
    int fftSize = 512;
    int numBins = 257;
    int numMels = 128;

    std::vector<float> window;
    std::vector<float> fftBuffer;                  // 2 * fftSize for JUCE FFT
    std::vector<std::vector<float>> melFilters;    // [numMels][numBins]
};
