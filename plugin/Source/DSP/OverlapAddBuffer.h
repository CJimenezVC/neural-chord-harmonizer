#pragma once

#include <vector>

/**
    Overlap-add reconstruction buffer. Synthesis frames are windowed and summed
    with a hop offset; read() drains finished samples. COLA-compliant windowing
    yields artifact-free reconstruction. Pre-allocated in prepare().
*/
class OverlapAddBuffer
{
public:
    void prepare (int frameSize, int hopSize, int channels);
    void reset();

    /** Window and accumulate one synthesis frame at the current write head. */
    void add (const float* frame, int numSamples);

    /** Copy out @p numSamples of finished audio, advancing the read head. */
    int read (float* dst, int numSamples);

private:
    std::vector<float> accumulator;   // running OLA sum
    std::vector<float> window;        // analysis/synthesis window
    int frame = 512;
    int hop = 128;
    int writePos = 0;
    int readPos = 0;
};
