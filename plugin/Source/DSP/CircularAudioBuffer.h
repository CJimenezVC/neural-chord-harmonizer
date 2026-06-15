#pragma once

#include <vector>
#include <cstring>
#include <algorithm>

/**
    Lock-free single-producer/single-consumer circular buffer for streaming
    audio. The audio thread pushes incoming samples and pops fixed-size,
    hop-advanced analysis frames. All storage is pre-allocated in prepare().
*/
class CircularAudioBuffer
{
public:
    void prepare (int frameSize, int hopSize, int /*channels*/)
    {
        hop = hopSize;
        // Headroom for several frames so the producer never overruns.
        buffer.assign ((size_t) (frameSize * 8), 0.0f);
        writePos = readPos = filled = 0;
    }

    void reset()
    {
        std::fill (buffer.begin(), buffer.end(), 0.0f);
        writePos = readPos = filled = 0;
    }

    /** Append @p numSamples to the ring. */
    void push (const float* src, int numSamples)
    {
        const int cap = (int) buffer.size();
        for (int i = 0; i < numSamples; ++i)
        {
            buffer[(size_t) writePos] = src[i];
            writePos = (writePos + 1) % cap;
            filled = std::min (filled + 1, cap);
        }
    }

    /** If a full frame is available, copy it out and advance by @p hopSize.
        Returns false when there is not yet enough buffered audio. */
    bool pop (float* dst, int frameSize, int hopSize)
    {
        if (filled < frameSize)
            return false;

        const int cap = (int) buffer.size();
        for (int i = 0; i < frameSize; ++i)
            dst[i] = buffer[(size_t) ((readPos + i) % cap)];

        readPos = (readPos + hopSize) % cap;
        filled -= hopSize;
        return true;
    }

private:
    std::vector<float> buffer;
    int hop = 128;
    int writePos = 0, readPos = 0, filled = 0;
};
