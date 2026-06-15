#pragma once

#include <atomic>
#include <vector>

#include "DSP/FeatureExtractor.h"
#include "DSP/SpectrogramProcessor.h"
#include "ML/ModelManager.h"
#include "Parameters/StyleInterpolation.h"
#include "Parameters/VoiceTransformParameters.h"

/**
    Drives the per-frame neural pipeline: normalize mel -> encode -> style
    modulation -> decode -> denormalize -> mel-inversion resynthesis (transformed
    mel + the input frame's phase -> waveform frame). Real-time safe:
    setStyleParams() publishes atomically and all buffers are pre-allocated.
*/
class NeuralAudioProcessor
{
public:
    void prepare (double sampleRate, int frameSize);
    void reset();

    void setModels (ModelManager* m) noexcept { models = m; }
    void setStyleParams (const StyleParams& p) noexcept;

    /** Use an exact (e.g. librosa) mel filterbank for resynthesis. */
    void setMelFilterbank (const float* fb, int nMels, int nBins)
    {
        synth.setMelFilterbank (fb, nMels, nBins);
    }

    /** Use the pseudo-inverse filterbank for resynthesis. */
    void setInverseMel (const float* inv, int nBins, int nMels)
    {
        synth.setInverseMel (inv, nBins, nMels);
    }

    /** Process one feature frame; returns audio sample count written. */
    int processFrame (const Features& feats, float* audioOut, int maxSamples);

private:
    ModelManager* models = nullptr;
    StyleInterpolation styleInterp;

    std::atomic<StyleParams> styleParams { StyleParams{} };

    SpectrogramProcessor synth;      // mel -> waveform resynthesis (ISTFT)

    void applyBrightness (float* mag, float brightness) const;   // spectral tilt
    void applyFormantShift (const float* mag, float semitones, float* out) const;

    int frameSize = 512;
    int numBins = 257;
    int styleDim = 64;
    int melBins = 128;
    float fMin = 20.0f, fMax = 8000.0f;
    // Hann-OLA COLA normalization (75% overlap). The retrained model reconstructs
    // at full energy, so no makeup gain is needed (output ≈ input level).
    float outputGain = 1.0f / 1.5f;

    // The encoder pools features over a window in training; in single-frame
    // streaming we approximate that window-mean with an EMA of the style vector.
    float styleAlpha = 0.95f;
    bool  styleInit = false;

    std::vector<float> melNorm;      // normalized encoder/decoder input
    std::vector<float> styleVec;     // encoder output (per frame)
    std::vector<float> styleAvg;     // EMA-smoothed style
    std::vector<float> styleMod;     // after interpolation
    std::vector<float> melOut;       // decoder output (normalized)
    std::vector<float> melDenorm;    // denormalized log-mel for resynthesis
    std::vector<float> magBuf;       // linear magnitude spectrum (numBins)
    std::vector<float> magWarp;      // formant-warped magnitude (numBins)
};
