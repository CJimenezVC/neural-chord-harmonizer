#include "test_framework.h"

#include "DSP/CircularAudioBuffer.h"
#include "DSP/OverlapAddBuffer.h"
#include "DSP/SampleFifo.h"
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

TEST_CASE (sample_fifo_push_consume_compacts)
{
    SampleFifo fifo;
    fifo.prepare (16);

    std::vector<float> a { 1, 2, 3, 4, 5 };
    CHECK (fifo.push (a.data(), 5) == 5);
    CHECK (fifo.size() == 5);

    fifo.consume (2);                 // drop 1,2 -> {3,4,5}
    CHECK (fifo.size() == 3);
    CHECK_NEAR (fifo.data()[0], 3.0f, 1e-6f);
    CHECK_NEAR (fifo.data()[2], 5.0f, 1e-6f);

    std::vector<float> b { 6, 7 };
    fifo.push (b.data(), 2);          // -> {3,4,5,6,7}
    CHECK (fifo.size() == 5);
    CHECK_NEAR (fifo.data()[4], 7.0f, 1e-6f);
}

TEST_CASE (sample_fifo_respects_capacity)
{
    SampleFifo fifo;
    fifo.prepare (4);
    std::vector<float> a { 1, 2, 3, 4, 5, 6 };
    CHECK (fifo.push (a.data(), 6) == 4);   // only 4 fit
    CHECK (fifo.size() == 4);
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
