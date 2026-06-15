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

    /** Replace the internally-built filterbank with an exact one (e.g. librosa's
        Slaney filterbank from model_info.json). @p fb is row-major [nMels][nBins]. */
    void setMelFilterbank (const float* fb, int nMels, int nBins);

    /** Window @p frame, FFT, write power spectrum (fftSize/2 + 1 bins). */
    void computeMagnitude (const float* frame, int numSamples, float* magOut);

    /** Like computeMagnitude but also writes the per-bin phase (radians). */
    void computeMagnitudeAndPhase (const float* frame, int numSamples,
                                   float* magOut, float* phaseOut);

    /** Project a magnitude spectrum onto the log-mel filterbank. */
    void magnitudeToLogMel (const float* magnitude, float* melOut) const;

    /** Resynthesize one time frame (fftSize samples) from a log-mel spectrum
        and a per-bin phase: inverse-mel -> linear magnitude, combine with phase,
        inverse FFT. Output is NOT windowed (the overlap-add stage windows it). */
    void melToFrame (const float* melLog, const float* phase, float* outFrame);

    int getNumBins() const noexcept { return numBins; }

private:
    void buildMelFilterbank (double sampleRate, float fMin, float fMax);
    void buildInverseMel();

    std::unique_ptr<juce::dsp::FFT> fft;
    int fftSize = 512;
    int numBins = 257;
    int numMels = 128;

    std::vector<float> window;
    std::vector<float> fftBuffer;                  // 2 * fftSize for JUCE FFT
    std::vector<float> ifftBuffer;                 // 2 * fftSize for inverse
    std::vector<std::vector<float>> melFilters;    // [numMels][numBins]
    std::vector<float> colNorm;                    // [numBins] inverse-mel norm
    std::vector<float> melLinScratch;              // [numMels]
};
