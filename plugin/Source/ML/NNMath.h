#pragma once

#include <algorithm>
#include <cmath>

/**
    Pure, dependency-free neural-network math used by NNModel. Kept JUCE-free so
    it can be unit-tested directly against independent reference formulas (see
    plugin/Tests/test_nn.cpp).

    Weight layouts match the exporter:
      dense  W: row-major [in][out]   -> out[o] = b[o] + Σ_i in[i]·W[i·outDim+o]
      conv   W: row-major [out][in]   -> out[o] = b[o] + Σ_i in[i]·W[o·inDim+i]
      gru    Wih [3H][in], Whh [3H][H], gate order r,z,n (PyTorch)
*/
namespace nnmath
{
    enum Activation { None = 0, ReLU = 1, Softmax = 2, Tanh = 3, Sigmoid = 4 };

    inline void activate (float* v, int n, int act)
    {
        switch (act)
        {
            case ReLU:    for (int i = 0; i < n; ++i) v[i] = std::max (0.0f, v[i]); break;
            case Tanh:    for (int i = 0; i < n; ++i) v[i] = std::tanh (v[i]); break;
            case Sigmoid: for (int i = 0; i < n; ++i) v[i] = 1.0f / (1.0f + std::exp (-v[i])); break;
            case Softmax:
            {
                float m = v[0];
                for (int i = 1; i < n; ++i) m = std::max (m, v[i]);
                float sum = 0.0f;
                for (int i = 0; i < n; ++i) { v[i] = std::exp (v[i] - m); sum += v[i]; }
                const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (int i = 0; i < n; ++i) v[i] *= inv;
                break;
            }
            default: break;
        }
    }

    /** Dense / fully-connected: W is row-major [inDim][outDim]. */
    inline void denseForward (const float* in, int inDim,
                              const float* W, const float* b, int outDim, float* out)
    {
        for (int o = 0; o < outDim; ++o)
        {
            float acc = b[o];
            for (int i = 0; i < inDim; ++i)
                acc += in[i] * W[i * outDim + o];
            out[o] = acc;
        }
    }

    /** Single-frame conv1d (centre tap): W is row-major [outDim][inDim]. */
    inline void convCentreForward (const float* in, int inDim,
                                   const float* W, const float* b, int outDim, float* out)
    {
        for (int o = 0; o < outDim; ++o)
        {
            const float* w = &W[o * inDim];
            float acc = b[o];
            for (int i = 0; i < inDim; ++i)
                acc += in[i] * w[i];
            out[o] = acc;
        }
    }

    /** One GRU step (PyTorch semantics). Updates @p state in place; @p gateIh and
        @p gateHh are caller-provided scratch of length 3·H. */
    inline void gruStep (const float* in, int inN,
                         const float* Wih, const float* Whh,
                         const float* bih, const float* bhh,
                         int H, float* state, float* gateIh, float* gateHh)
    {
        for (int g = 0; g < 3 * H; ++g)
        {
            const float* wi = &Wih[g * inN];
            float gi = bih[g];
            for (int i = 0; i < inN; ++i) gi += wi[i] * in[i];
            gateIh[g] = gi;

            const float* wh = &Whh[g * H];
            float gh = bhh[g];
            for (int j = 0; j < H; ++j) gh += wh[j] * state[j];
            gateHh[g] = gh;
        }
        for (int j = 0; j < H; ++j)
        {
            const float r = 1.0f / (1.0f + std::exp (-(gateIh[j] + gateHh[j])));
            const float z = 1.0f / (1.0f + std::exp (-(gateIh[H + j] + gateHh[H + j])));
            const float n = std::tanh (gateIh[2 * H + j] + r * gateHh[2 * H + j]);
            state[j] = (1.0f - z) * n + z * state[j];
        }
    }
}
