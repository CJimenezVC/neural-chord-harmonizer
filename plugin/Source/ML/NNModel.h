#pragma once

#include <vector>

#include <juce_core/juce_core.h>

/**
    Minimal self-contained feed-forward engine for the exported `.rtneural`
    models. Loads the JSON written by training/export_rtneural.py and runs a
    single-frame (streaming) forward pass.

    The math is validated to match the PyTorch models to ~1e-8 (see
    training/verify_rtneural_export.py). It is deliberately dependency-light
    (no RTNeural parser coupling) so it compiles and runs predictably; RTNeural
    remains linked and can back an optimized path later.

    Supported layers: dense, conv1d (single-frame centre tap), global_mean_pool
    (identity for one frame), gru (state carried across forward() calls).
*/
class NNModel
{
public:
    bool loadFromFile (const juce::File& jsonFile);
    bool isLoaded() const noexcept { return loaded; }

    int inputSize() const noexcept { return inSize; }
    int outputSize() const noexcept { return outSize; }

    /** Run all layers on one input frame; writes outputSize() values to out.
        Carries recurrent (GRU) state across calls. */
    void forward (const float* input, int inputLen, float* out);

    /** Clear recurrent state (e.g. on transport stop / model reload). */
    void reset();

private:
    enum class LayerType { Dense, Conv1dCentre, GlobalMeanPool, Gru };
    enum class Activation { None, ReLU, Softmax, Tanh, Sigmoid };

    struct Layer
    {
        LayerType  type = LayerType::Dense;
        Activation activation = Activation::None;
        int inDim = 0, outDim = 0;

        std::vector<float> weight;   // dense: [in][out]; conv centre: [out][in]
        std::vector<float> bias;     // [outDim]

        // GRU only:
        int hidden = 0;
        std::vector<float> Wih, Whh, bih, bhh;  // [3H*in] [3H*H] [3H] [3H]
        std::vector<float> state;               // [H]
    };

    bool parseLayer (const juce::var& layerVar, Layer& out);
    void applyLayer (Layer& layer, const float* in, int inLen, float* out, int& outLen);

    std::vector<Layer> layers;
    std::vector<float> bufA, bufB;     // ping-pong scratch
    std::vector<float> gateIh, gateHh; // GRU gate scratch (3H each)
    int inSize = 0, outSize = 0, maxDim = 0, maxHidden = 0;
    bool loaded = false;
};
