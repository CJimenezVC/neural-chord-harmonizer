#include "OverlapAddBuffer.h"

#include <algorithm>
#include <cmath>

void OverlapAddBuffer::prepare (int frameSize, int hopSize, int /*channels*/)
{
    frame = frameSize;
    hop = hopSize;
    accumulator.assign ((size_t) (frameSize * 8), 0.0f);
    writePos = readPos = 0;

    // Hann window (COLA-compliant at 75% overlap).
    window.resize ((size_t) frameSize);
    for (int n = 0; n < frameSize; ++n)
        window[(size_t) n] = 0.5f - 0.5f * std::cos (2.0f * 3.14159265358979f * n
                                                     / (float) (frameSize - 1));
}

void OverlapAddBuffer::reset()
{
    std::fill (accumulator.begin(), accumulator.end(), 0.0f);
    writePos = readPos = 0;
}

void OverlapAddBuffer::add (const float* frameData, int numSamples)
{
    const int cap = (int) accumulator.size();
    const int n = std::min (numSamples, frame);
    for (int i = 0; i < n; ++i)
        accumulator[(size_t) ((writePos + i) % cap)] += frameData[i] * window[(size_t) i];

    writePos = (writePos + hop) % cap;
}

int OverlapAddBuffer::read (float* dst, int numSamples)
{
    const int cap = (int) accumulator.size();
    for (int i = 0; i < numSamples; ++i)
    {
        const int idx = (readPos + i) % cap;
        dst[i] = accumulator[(size_t) idx];
        accumulator[(size_t) idx] = 0.0f;   // consume
    }
    readPos = (readPos + numSamples) % cap;
    return numSamples;
}
