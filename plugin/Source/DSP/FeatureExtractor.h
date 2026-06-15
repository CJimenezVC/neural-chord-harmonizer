#pragma once

#include <array>
#include <vector>

#include "SpectrogramProcessor.h"
#include "YINPitchDetector.h"
#include "FormantAnalyzer.h"

/** Per-frame voice features consumed by the neural front end. */
struct Features
{
    std::vector<float> mel;            // log-mel, n_mels bins
    std::vector<float> phase;          // STFT phase, fftSize/2 + 1 bins (radians)
    float f0 = 0.0f;                   // Hz
    std::array<float, 4> formants{};   // F1..F4 (Hz)
    float voicedConfidence = 0.0f;     // [0,1]
};

/**
    Extracts mel-spectrogram, F0 (YIN) and formants (LPC) from a single audio
    frame. All buffers are pre-allocated in prepare(); process() is allocation
    free.
*/
class FeatureExtractor
{
public:
    void prepare (double sampleRate, int frameSize, int hopSize, int nMels = 128);
    void reset();

    /** Use an exact (e.g. librosa) mel filterbank instead of the built-in one. */
    void setMelFilterbank (const float* fb, int nMels, int nBins)
    {
        spectrogram.setMelFilterbank (fb, nMels, nBins);
    }

    Features process (const float* frame, int numSamples);

    int getNumMels() const noexcept { return numMels; }

private:
    SpectrogramProcessor spectrogram;
    YINPitchDetector     pitch;
    FormantAnalyzer      formants;

    double sr = 48000.0;
    int    numMels = 128;
    int    frameSize = 512;
    int    hopSize = 128;

    Features scratch;                 // reused output
    std::vector<float> magnitude;     // STFT magnitude scratch
};
