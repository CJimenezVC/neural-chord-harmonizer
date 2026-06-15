#pragma once

#include <array>
#include <vector>

/**
    Estimates the first four formant frequencies (F1..F4) via linear predictive
    coding: an LPC all-pole fit whose root angles map to formant frequencies.
*/
class FormantAnalyzer
{
public:
    void prepare (double sampleRate, int lpcOrder = 12);

    std::array<float, 4> estimate (const float* frame, int numSamples);

private:
    void autocorrelate (const float* x, int n, float* r, int order) const;
    void levinsonDurbin (const float* r, float* lpc, int order) const;

    double sampleRate = 48000.0;
    int    order = 12;

    std::vector<float> windowed;
    std::vector<float> autocorr;
    std::vector<float> lpc;
};
