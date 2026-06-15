#include "SpectrumAnalyzer.h"

SpectrumAnalyzer::SpectrumAnalyzer()
{
    setOpaque (true);
    startTimerHz (30);
}

SpectrumAnalyzer::~SpectrumAnalyzer() = default;

void SpectrumAnalyzer::pushSample (float sample) noexcept
{
    if (fifoIndex == fftSize)
    {
        if (! nextBlockReady.load())
        {
            std::fill (fftData.begin(), fftData.end(), 0.0f);
            std::copy (fifo.begin(), fifo.end(), fftData.begin());
            nextBlockReady.store (true);
        }
        fifoIndex = 0;
    }
    fifo[(size_t) fifoIndex++] = sample;
}

void SpectrumAnalyzer::timerCallback()
{
    if (! nextBlockReady.load())
        return;

    window.multiplyWithWindowingTable (fftData.data(), fftSize);
    fft.performFrequencyOnlyForwardTransform (fftData.data());

    for (int i = 0; i < fftSize / 2; ++i)
    {
        const float mag = fftData[(size_t) i];
        const float db = juce::Decibels::gainToDecibels (mag) - juce::Decibels::gainToDecibels ((float) fftSize);
        scopeData[(size_t) i] = juce::jlimit (0.0f, 1.0f, (db + 100.0f) / 100.0f);
    }

    nextBlockReady.store (false);
    repaint();
}

void SpectrumAnalyzer::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colour (0xff14141a));
    g.setColour (juce::Colour (0xff4fc3f7));

    const auto w = (float) getWidth();
    const auto h = (float) getHeight();
    juce::Path p;
    p.startNewSubPath (0.0f, h);
    for (int i = 0; i < fftSize / 2; ++i)
    {
        const float x = juce::jmap ((float) i, 0.0f, (float) (fftSize / 2), 0.0f, w);
        const float y = h - scopeData[(size_t) i] * h;
        p.lineTo (x, y);
    }
    g.strokePath (p, juce::PathStrokeType (1.5f));
}
