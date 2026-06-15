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

    /** Process one feature frame; returns audio sample count written. */
    int processFrame (const Features& feats, float* audioOut, int maxSamples);

private:
    ModelManager* models = nullptr;
    StyleInterpolation styleInterp;

    std::atomic<StyleParams> styleParams { StyleParams{} };

    SpectrogramProcessor synth;      // mel -> waveform resynthesis (ISTFT)

    int frameSize = 512;
    int styleDim = 64;
    int melBins = 128;
    float fMin = 20.0f, fMax = 8000.0f;
    // Output gain = (1/1.5 Hann-OLA COLA) × (~4.5 makeup for the small model's
    // energy loss), so output ≈ input level. Model-specific; tune to taste.
    float outputGain = 3.0f;

    std::vector<float> melNorm;      // normalized encoder/decoder input
    std::vector<float> styleVec;     // encoder output
    std::vector<float> styleMod;     // after interpolation
    std::vector<float> melOut;       // decoder output (normalized)
    std::vector<float> melDenorm;    // denormalized log-mel for resynthesis
};
