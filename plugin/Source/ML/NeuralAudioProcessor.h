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
    float outputGain = 1.0f / 1.5f;  // Hann synthesis window, 75% overlap (COLA ≈ 1.5)

    std::vector<float> melNorm;      // normalized encoder/decoder input
    std::vector<float> styleVec;     // encoder output
    std::vector<float> styleMod;     // after interpolation
    std::vector<float> melOut;       // decoder output (normalized)
    std::vector<float> melDenorm;    // denormalized log-mel for resynthesis
};
