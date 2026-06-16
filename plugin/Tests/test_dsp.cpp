#include "test_framework.h"

#include "DSP/OverlapAddBuffer.h"
#include "DSP/SampleFifo.h"

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
