#pragma once

#include <utility>
#include <vector>

/**
    YIN fundamental-frequency estimator (de Cheveigné & Kawahara, 2002).

    Returns {f0 in Hz, confidence in [0,1]}. confidence == 0 indicates an
    unvoiced / undecided frame. All scratch storage is pre-allocated.
*/
class YINPitchDetector
{
public:
    void prepare (double sampleRate, int bufferSize);
    void reset();

    std::pair<float, float> detect (const float* buffer, int numSamples);

    void setThreshold (float t) noexcept { threshold = t; }

private:
    double sampleRate = 48000.0;
    float  threshold = 0.1f;
    int    minPeriod = 50;     // ~960 Hz
    int    maxPeriod = 4000;   // ~12 Hz

    std::vector<float> diff;             // difference function
    std::vector<float> cumulativeMean;   // CMND function
};
