#include "NNModel.h"

#include <algorithm>
#include <cmath>

#include "NNMath.h"

namespace
{
    void flatten2D (const juce::var& v, std::vector<float>& dst)
    {
        dst.clear();
        if (auto* rows = v.getArray())
            for (auto& row : *rows)
                if (auto* cols = row.getArray())
                    for (auto& c : *cols)
                        dst.push_back ((float) (double) c);
    }

    void flatten1D (const juce::var& v, std::vector<float>& dst)
    {
        dst.clear();
        if (auto* a = v.getArray())
            for (auto& c : *a)
                dst.push_back ((float) (double) c);
    }

    // Maps the exporter's activation string onto NNModel::Activation values
    // (None=0, ReLU=1, Softmax=2, Tanh=3, Sigmoid=4).
    int activationFromString (const juce::String& s)
    {
        if (s == "relu")    return 1;
        if (s == "softmax") return 2;
        if (s == "tanh")    return 3;
        if (s == "sigmoid") return 4;
        return 0;
    }
}

bool NNModel::loadFromFile (const juce::File& jsonFile)
{
    loaded = false;
    layers.clear();

    if (! jsonFile.existsAsFile())
        return false;

    auto json = juce::JSON::parse (jsonFile.loadFileAsString());
    if (! json.isObject())
        return false;

    inSize = (int) json["in_shape"];
    maxDim = inSize;
    maxHidden = 0;

    auto layersVar = json["layers"];
    if (! layersVar.isArray())
        return false;

    int lastOut = inSize;
    for (auto& lv : *layersVar.getArray())
    {
        Layer layer;
        if (! parseLayer (lv, layer))
            return false;

        if (layer.type != LayerType::GlobalMeanPool)
            lastOut = layer.outDim;

        maxDim = std::max ({ maxDim, layer.inDim, layer.outDim, layer.hidden });
        maxHidden = std::max (maxHidden, layer.hidden);
        layers.push_back (std::move (layer));
    }

    outSize = lastOut;
    bufA.assign ((size_t) std::max (maxDim, 1), 0.0f);
    bufB.assign ((size_t) std::max (maxDim, 1), 0.0f);
    gateIh.assign ((size_t) std::max (3 * maxHidden, 1), 0.0f);
    gateHh.assign ((size_t) std::max (3 * maxHidden, 1), 0.0f);

    loaded = true;
    return true;
}

bool NNModel::parseLayer (const juce::var& lv, Layer& layer)
{
    const juce::String type = lv["type"].toString();
    layer.activation = (Activation) activationFromString (lv["activation"].toString());

    if (type == "dense")
    {
        auto shape = lv["shape"];
        layer.type   = LayerType::Dense;
        layer.inDim  = (int) shape[0];
        layer.outDim = (int) shape[1];
        flatten2D (lv["weights"][0], layer.weight);    // [in][out]
        flatten1D (lv["weights"][1], layer.bias);
        return (int) layer.weight.size() == layer.inDim * layer.outDim;
    }

    if (type == "conv1d")
    {
        layer.type   = LayerType::Conv1dCentre;
        layer.inDim  = (int) lv["in_channels"];
        layer.outDim = (int) lv["out_channels"];
        if (layer.activation == Activation::None)
            layer.activation = Activation::ReLU;          // exporter default

        const int k  = (int) lv["kernel_size"];
        const int kc = k / 2;                             // centre tap
        auto w = lv["weights"][0];                        // [out][in][k]
        layer.weight.assign ((size_t) (layer.outDim * layer.inDim), 0.0f);
        if (auto* outs = w.getArray())
            for (int o = 0; o < layer.outDim && o < outs->size(); ++o)
                if (auto* ins = (*outs)[o].getArray())
                    for (int i = 0; i < layer.inDim && i < ins->size(); ++i)
                        if (auto* ks = (*ins)[i].getArray())
                            layer.weight[(size_t) (o * layer.inDim + i)] = (float) (double) (*ks)[kc];
        flatten1D (lv["weights"][1], layer.bias);
        return true;
    }

    if (type == "global_mean_pool")
    {
        layer.type = LayerType::GlobalMeanPool;           // identity for one frame
        return true;
    }

    if (type == "gru")
    {
        layer.type   = LayerType::Gru;
        layer.hidden = (int) lv["hidden_size"];
        auto w = lv["weights"];
        flatten2D (w["weight_ih"], layer.Wih);
        flatten2D (w["weight_hh"], layer.Whh);
        flatten1D (w["bias_ih"], layer.bih);
        flatten1D (w["bias_hh"], layer.bhh);
        layer.outDim = layer.hidden;
        layer.inDim  = layer.hidden > 0 ? (int) layer.Wih.size() / (3 * layer.hidden) : 0;
        layer.state.assign ((size_t) layer.hidden, 0.0f);
        return layer.hidden > 0 && ! layer.Wih.empty();
    }

    return false;   // unknown layer type
}

void NNModel::applyLayer (Layer& layer, const float* in, int inLen, float* out, int& outLen)
{
    switch (layer.type)
    {
        case LayerType::Dense:
            nnmath::denseForward (in, layer.inDim, layer.weight.data(), layer.bias.data(),
                                  layer.outDim, out);
            outLen = layer.outDim;
            nnmath::activate (out, outLen, (int) layer.activation);
            break;

        case LayerType::Conv1dCentre:
            nnmath::convCentreForward (in, layer.inDim, layer.weight.data(), layer.bias.data(),
                                       layer.outDim, out);
            outLen = layer.outDim;
            nnmath::activate (out, outLen, (int) layer.activation);
            break;

        case LayerType::GlobalMeanPool:                   // single frame -> identity
            std::copy (in, in + inLen, out);
            outLen = inLen;
            break;

        case LayerType::Gru:
            nnmath::gruStep (in, layer.inDim, layer.Wih.data(), layer.Whh.data(),
                             layer.bih.data(), layer.bhh.data(), layer.hidden,
                             layer.state.data(), gateIh.data(), gateHh.data());
            std::copy (layer.state.begin(), layer.state.end(), out);
            outLen = layer.hidden;
            break;
    }
}

void NNModel::forward (const float* input, int inputLen, float* outPtr)
{
    if (! loaded) { std::fill (outPtr, outPtr + outSize, 0.0f); return; }

    int len = std::min (inputLen, (int) bufA.size());
    std::copy (input, input + len, bufA.begin());

    float* cur  = bufA.data();
    float* next = bufB.data();
    for (auto& layer : layers)
    {
        int outLen = 0;
        applyLayer (layer, cur, len, next, outLen);
        std::swap (cur, next);
        len = outLen;
    }

    std::copy (cur, cur + std::min (len, outSize), outPtr);
}

void NNModel::reset()
{
    for (auto& layer : layers)
        if (layer.type == LayerType::Gru)
            std::fill (layer.state.begin(), layer.state.end(), 0.0f);
}
