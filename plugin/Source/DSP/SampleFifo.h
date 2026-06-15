#pragma once

#include <algorithm>
#include <cstring>
#include <vector>

/**
    Pre-allocated linear mono FIFO used to buffer samples between the
    host-rate and model-rate stages of the resampling chain.

    Real-time safe: storage is reserved in prepare(); push/consume never
    allocate. consume() compacts the remaining samples to the front with a
    single memmove (cheap for the few-hundred-sample sizes used here).
*/
class SampleFifo
{
public:
    void prepare (int capacity)
    {
        buffer.assign ((size_t) capacity, 0.0f);
        fill = 0;
    }

    void reset() noexcept { fill = 0; }

    int  size() const noexcept     { return fill; }
    int  capacity() const noexcept { return (int) buffer.size(); }
    bool empty() const noexcept    { return fill == 0; }
    const float* data() const noexcept { return buffer.data(); }

    /** Append up to @p n samples; returns the number actually stored. */
    int push (const float* src, int n)
    {
        n = std::min (n, capacity() - fill);
        std::copy (src, src + n, buffer.begin() + fill);
        fill += n;
        return n;
    }

    /** Drop the first @p n samples, shifting the remainder to the front. */
    void consume (int n) noexcept
    {
        n = std::min (n, fill);
        if (n > 0 && fill - n > 0)
            std::memmove (buffer.data(), buffer.data() + n, sizeof (float) * (size_t) (fill - n));
        fill -= n;
    }

private:
    std::vector<float> buffer;
    int fill = 0;
};
