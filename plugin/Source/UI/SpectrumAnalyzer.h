#pragma once

#include <array>
#include <atomic>

#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>

/**
    Lightweight FFT magnitude display. The audio thread pushes samples via
    pushSample(); the component repaints on a timer.
*/
class SpectrumAnalyzer : public juce::Component,
                         private juce::Timer
{
public:
    SpectrumAnalyzer();
    ~SpectrumAnalyzer() override;

    /** Real-time safe: copy a sample into the analysis FIFO. */
    void pushSample (float sample) noexcept;

    void paint (juce::Graphics&) override;

private:
    void timerCallback() override;

    static constexpr int fftOrder = 10;            // 1024-point
    static constexpr int fftSize  = 1 << fftOrder;

    std::array<float, fftSize>     fifo {};
    std::array<float, fftSize * 2> fftData {};
    int  fifoIndex = 0;
    std::atomic<bool> nextBlockReady { false };

    juce::dsp::FFT     fft { fftOrder };
    juce::dsp::WindowingFunction<float> window
        { (size_t) fftSize, juce::dsp::WindowingFunction<float>::hann };

    std::array<float, fftSize / 2> scopeData {};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SpectrumAnalyzer)
};
