#include "test_framework.h"

#include <cmath>
#include <vector>

#include "ML/NNMath.h"

namespace
{
    float sigmoidf (float x) { return 1.0f / (1.0f + std::exp (-x)); }
}

TEST_CASE (nn_dense_matches_reference)
{
    // in=[1,2]; W row-major [in][out]=[[1,0],[0,1]]; b=[0.5,-0.5]
    const float in[]  = { 1.0f, 2.0f };
    const float W[]   = { 1.0f, 0.0f, 0.0f, 1.0f };
    const float b[]   = { 0.5f, -0.5f };
    float out[2] = { 0, 0 };
    nnmath::denseForward (in, 2, W, b, 2, out);
    CHECK_NEAR (out[0], 1.5f, 1e-6f);   // 0.5 + 1*1 + 2*0
    CHECK_NEAR (out[1], 1.5f, 1e-6f);   // -0.5 + 1*0 + 2*1
}

TEST_CASE (nn_conv_centre_matches_reference)
{
    // in=[1,2,3]; W row-major [out][in]=[[1,1,1],[0,1,0]]; b=[0,1]
    const float in[]  = { 1.0f, 2.0f, 3.0f };
    const float W[]   = { 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f };
    const float b[]   = { 0.0f, 1.0f };
    float out[2] = { 0, 0 };
    nnmath::convCentreForward (in, 3, W, b, 2, out);
    CHECK_NEAR (out[0], 6.0f, 1e-6f);   // 1+2+3
    CHECK_NEAR (out[1], 3.0f, 1e-6f);   // 1 + 2
}

TEST_CASE (nn_activate_relu_and_softmax)
{
    float r[] = { -2.0f, 0.0f, 3.0f };
    nnmath::activate (r, 3, nnmath::ReLU);
    CHECK_NEAR (r[0], 0.0f, 1e-6f);
    CHECK_NEAR (r[2], 3.0f, 1e-6f);

    float s[] = { 1.0f, 1.0f };
    nnmath::activate (s, 2, nnmath::Softmax);
    CHECK_NEAR (s[0], 0.5f, 1e-6f);
    CHECK_NEAR (s[0] + s[1], 1.0f, 1e-6f);
}

TEST_CASE (nn_gru_step_matches_independent_formula)
{
    // H=1, inN=1, distinct per-gate weights + nonzero state to exercise the
    // r/z/n gate order and the r * (hh n-gate) coupling.
    const int H = 1, inN = 1;
    const float Wih[] = { 0.1f, 0.2f, 0.3f };   // [r, z, n] x in
    const float Whh[] = { 0.4f, 0.5f, 0.6f };   // [r, z, n] x h
    const float bih[] = { 0.0f, 0.0f, 0.0f };
    const float bhh[] = { 0.0f, 0.0f, 0.0f };
    const float x = 1.0f;

    float state = 0.7f;
    float gIh[3], gHh[3];
    nnmath::gruStep (&x, inN, Wih, Whh, bih, bhh, H, &state, gIh, gHh);

    // Independent reference (same GRU equations, written out by hand).
    const float h0 = 0.7f;
    const float r = sigmoidf (0.1f * x + 0.4f * h0);
    const float z = sigmoidf (0.2f * x + 0.5f * h0);
    const float n = std::tanh (0.3f * x + r * (0.6f * h0));
    const float expected = (1.0f - z) * n + z * h0;

    CHECK_NEAR (state, expected, 1e-6f);
}
