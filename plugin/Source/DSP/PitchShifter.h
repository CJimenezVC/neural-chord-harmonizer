#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "OverlapAddBuffer.h"
#include "SampleFifo.h"
#include "SimpleFFT.h"

/**
    Streaming, formant-preserving phase-vocoder pitch shifter (JUCE-free, so it
    compiles into both the plugin and a standalone test).

    Per analysis hop: STFT -> instantaneous frequency from phase difference ->
    whiten by a smoothed spectral envelope (formants) -> shift the excitation
    bins by the current ratio and rescale their frequencies -> re-apply the
    envelope (keeps formants in place -> no chipmunk) -> accumulate synthesis
    phase -> inverse FFT -> overlap-add.

    Direct port of training/harmonizer_demo.py:pitch_shift(). Latency ≈ fftSize.
    setRatio() may change every block.
*/
class PitchShifter
{
public:
    void prepare (int fftSizeIn, int hopIn)
    {
        fftSize = fftSizeIn;
        hop = hopIn;
        numBins = fftSize / 2 + 1;
        fft.prepare (fftSize);

        window.resize ((size_t) fftSize);
        for (int n = 0; n < fftSize; ++n)
            window[(size_t) n] = 0.5f - 0.5f * std::cos (2.0f * kPi * n / (float) (fftSize - 1));

        omega.resize ((size_t) numBins);
        for (int k = 0; k < numBins; ++k)
            omega[(size_t) k] = 2.0f * kPi * hop * k / (float) fftSize;

        inFifo.prepare (fftSize * 4);
        outFifo.prepare (fftSize * 4);
        ola.prepare (fftSize, hop, 1);

        re.assign ((size_t) fftSize, 0.0f);
        im.assign ((size_t) fftSize, 0.0f);
        mag.assign ((size_t) numBins, 0.0f);
        phase.assign ((size_t) numBins, 0.0f);
        prevPhase.assign ((size_t) numBins, 0.0f);
        synPhase.assign ((size_t) numBins, 0.0f);
        env.assign ((size_t) numBins, 0.0f);
        logScratch.assign ((size_t) numBins, 0.0f);
        newMag.assign ((size_t) numBins, 0.0f);
        newInst.assign ((size_t) numBins, 0.0f);
        frame.assign ((size_t) fftSize, 0.0f);
        hopScratch.assign ((size_t) hop, 0.0f);
        reset();
    }

    void reset()
    {
        inFifo.reset();
        outFifo.reset();
        ola.reset();
        std::fill (prevPhase.begin(), prevPhase.end(), 0.0f);
        std::fill (synPhase.begin(), synPhase.end(), 0.0f);
        primed = false;
    }

    void setRatio (float r) noexcept { ratio = r; }
    void setEnvelopeSmoothing (int bins) noexcept { envWidth = bins; }
    int  getLatencySamples() const noexcept { return fftSize; }

    /** Process @p numSamples; writes the same count to @p out (delayed ~fftSize). */
    void process (const float* in, float* out, int numSamples)
    {
        inFifo.push (in, numSamples);

        while (inFifo.size() >= fftSize)
        {
            processFrame (inFifo.data());
            inFifo.consume (hop);
            const int got = ola.read (hopScratch.data(), hop);
            outFifo.push (hopScratch.data(), got);
        }

        if (! primed && outFifo.size() >= fftSize)
            primed = true;

        if (! primed)
        {
            std::fill (out, out + numSamples, 0.0f);
            return;
        }

        const int n = std::min (numSamples, outFifo.size());
        std::copy (outFifo.data(), outFifo.data() + n, out);
        for (int i = n; i < numSamples; ++i) out[i] = 0.0f;
        outFifo.consume (n);
    }

private:
    static constexpr float kPi = 3.14159265358979f;

    static float wrapPi (float x)
    {
        x = std::fmod (x + kPi, 2.0f * kPi);
        if (x < 0.0f) x += 2.0f * kPi;
        return x - kPi;
    }

    void smoothLogEnv()
    {
        const int half = envWidth / 2;
        for (int k = 0; k < numBins; ++k)
            logScratch[(size_t) k] = std::log (mag[(size_t) k] + 1e-8f);
        for (int k = 0; k < numBins; ++k)
        {
            float acc = 0.0f; int cnt = 0;
            for (int j = std::max (0, k - half); j <= std::min (numBins - 1, k + half); ++j)
            { acc += logScratch[(size_t) j]; ++cnt; }
            env[(size_t) k] = std::exp (acc / (float) std::max (1, cnt));
        }
    }

    void processFrame (const float* x)
    {
        for (int n = 0; n < fftSize; ++n) { re[(size_t) n] = x[n] * window[(size_t) n]; im[(size_t) n] = 0.0f; }
        fft.transform (re.data(), im.data(), false);

        for (int k = 0; k < numBins; ++k)
        {
            const float rr = re[(size_t) k], ii = im[(size_t) k];
            mag[(size_t) k] = std::sqrt (rr * rr + ii * ii);
            phase[(size_t) k] = std::atan2 (ii, rr);
        }
        smoothLogEnv();

        std::fill (newMag.begin(), newMag.end(), 0.0f);
        std::fill (newInst.begin(), newInst.end(), 0.0f);
        for (int k = 0; k < numBins; ++k)
        {
            const float d = wrapPi (phase[(size_t) k] - prevPhase[(size_t) k] - omega[(size_t) k]);
            const float inst = omega[(size_t) k] + d;
            const float flat = mag[(size_t) k] / (env[(size_t) k] + 1e-8f);   // whiten
            const int kr = (int) std::lround ((double) k * ratio);
            if (kr >= 0 && kr < numBins)
            {
                newMag[(size_t) kr]  += flat * env[(size_t) kr];              // re-apply formant
                newInst[(size_t) kr]  = inst * ratio;
            }
        }
        prevPhase.assign (phase.begin(), phase.end());

        for (int k = 0; k < numBins; ++k)
        {
            synPhase[(size_t) k] += newInst[(size_t) k];
            re[(size_t) k] = newMag[(size_t) k] * std::cos (synPhase[(size_t) k]);
            im[(size_t) k] = newMag[(size_t) k] * std::sin (synPhase[(size_t) k]);
        }
        for (int k = 1; k < numBins - 1; ++k)
        {
            re[(size_t) (fftSize - k)] =  re[(size_t) k];
            im[(size_t) (fftSize - k)] = -im[(size_t) k];
        }
        fft.transform (re.data(), im.data(), true);

        const float g = 1.0f / 1.5f;   // analysis*synthesis Hann at 75% overlap
        for (int n = 0; n < fftSize; ++n) frame[(size_t) n] = re[(size_t) n] * g;
        ola.add (frame.data(), fftSize);
    }

    SimpleFFT fft;
    int fftSize = 1024, hop = 256, numBins = 513, envWidth = 21;
    float ratio = 1.0f;
    bool primed = false;

    std::vector<float> window, omega;
    std::vector<float> re, im, mag, phase, prevPhase, synPhase, env, logScratch, newMag, newInst, frame, hopScratch;
    SampleFifo inFifo, outFifo;
    OverlapAddBuffer ola;
};
