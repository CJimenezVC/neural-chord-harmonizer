#pragma once

#include <cmath>

#include <juce_audio_basics/juce_audio_basics.h>

/**
    Arbitrary-ratio mono resampler wrapping juce::LagrangeInterpolator.

    Used once in the plugin: downsampling the sidechain instrument from the host
    rate to the detector's 24 kHz (the voice path stays at host rate). The
    interpolator keeps an internal fractional read position, so feeding it from a
    FIFO and advancing by the reported input-used count avoids drift.

    speedRatio = inputRate / outputRate
      > 1 downsamples (e.g. 48k→24k = 2.0)   ← the case used here
      < 1 upsamples   (e.g. 24k→48k = 0.5)   ← supported, currently unused
*/
class Resampler
{
public:
    void prepare (double inputRate, double outputRate)
    {
        speedRatio = inputRate / outputRate;
        interpolator.reset();
    }

    void reset() { interpolator.reset(); }

    /** Produce @p numOut output samples from @p in; returns input samples used.
        @p in must hold at least inputSamplesNeeded(numOut) valid samples. */
    int process (const float* in, float* out, int numOut)
    {
        return interpolator.process (speedRatio, in, out, numOut);
    }

    /** Worst-case input samples consumed to produce @p numOut output samples. */
    int inputSamplesNeeded (int numOut) const noexcept
    {
        return (int) std::ceil (numOut * speedRatio) + 2;
    }

    double getSpeedRatio() const noexcept { return speedRatio; }

private:
    juce::LagrangeInterpolator interpolator;
    double speedRatio = 1.0;
};
