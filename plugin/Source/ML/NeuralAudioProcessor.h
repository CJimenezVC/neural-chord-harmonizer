#pragma once

#include <atomic>
#include <vector>

#include "DSP/FeatureExtractor.h"
#include "ML/ModelManager.h"
#include "Parameters/StyleInterpolation.h"
#include "Parameters/VoiceTransformParameters.h"

/**
    Drives the per-frame neural pipeline: encode -> style modulation -> decode
    -> vocode. Real-time safe: setStyleParams() publishes atomically and all
    buffers are pre-allocated in prepare().
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

    int frameSize = 512;
    int styleDim = 64;
    int melBins = 128;

    std::vector<float> styleVec;     // encoder output
    std::vector<float> styleMod;     // after interpolation
    std::vector<float> melOut;       // decoder output
};
