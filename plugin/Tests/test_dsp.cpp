#include "test_framework.h"

#include "DSP/CircularAudioBuffer.h"
#include "DSP/OverlapAddBuffer.h"
#include "Parameters/ParameterSmoothing.h"

TEST_CASE (circular_buffer_streams_frames)
{
    CircularAudioBuffer buf;
    buf.prepare (/*frameSize*/ 8, /*hopSize*/ 4, 1);

    std::vector<float> in (16);
    for (int i = 0; i < 16; ++i) in[(size_t) i] = (float) i;
    buf.push (in.data(), 16);

    std::vector<float> frame (8);
    CHECK (buf.pop (frame.data(), 8, 4));
    CHECK_NEAR (frame[0], 0.0f, 1e-6f);
    CHECK_NEAR (frame[7], 7.0f, 1e-6f);

    // After one hop of 4, the next frame should start at sample 4.
    CHECK (buf.pop (frame.data(), 8, 4));
    CHECK_NEAR (frame[0], 4.0f, 1e-6f);
}

TEST_CASE (circular_buffer_underflow_returns_false)
{
    CircularAudioBuffer buf;
    buf.prepare (8, 4, 1);
    std::vector<float> frame (8);
    CHECK (! buf.pop (frame.data(), 8, 4));   // nothing pushed yet
}

TEST_CASE (overlap_add_reconstructs_length)
{
    OverlapAddBuffer ola;
    ola.prepare (8, 4, 1);

    std::vector<float> frame (8, 1.0f);
    ola.add (frame.data(), 8);

    std::vector<float> out (4);
    CHECK (ola.read (out.data(), 4) == 4);
}

TEST_CASE (smoothed_value_approaches_target)
{
    SmoothedValue s;
    s.prepare (48000.0, 5.0f);
    s.snapTo (0.0f);
    s.setTarget (1.0f);
    for (int i = 0; i < 10000; ++i) s.next();
    CHECK_NEAR (s.getCurrent(), 1.0f, 1e-2f);
}
