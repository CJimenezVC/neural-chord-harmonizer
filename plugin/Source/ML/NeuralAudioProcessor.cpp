#include "NeuralAudioProcessor.h"

#include <algorithm>
#include <cmath>

void NeuralAudioProcessor::prepare (double sampleRate, int frame)
{
    frameSize = frame;
    numBins = frame / 2 + 1;
    styleInterp.prepare (styleDim);
    synth.prepare (sampleRate, frameSize, melBins, fMin, fMax);

    melNorm.assign   ((size_t) melBins, 0.0f);
    styleVec.assign  ((size_t) styleDim, 0.0f);
    styleAvg.assign  ((size_t) styleDim, 0.0f);
    styleMod.assign  ((size_t) styleDim, 0.0f);
    styleInit = false;
    melOut.assign    ((size_t) melBins, 0.0f);
    melDenorm.assign ((size_t) melBins, 0.0f);
    inEnv.assign     ((size_t) numBins, 0.0f);
    decEnv.assign    ((size_t) numBins, 0.0f);
    magBuf.assign    ((size_t) numBins, 0.0f);
    magWarp.assign   ((size_t) numBins, 0.0f);
}

void NeuralAudioProcessor::applyBrightness (float* mag, float brightness) const
{
    if (std::abs (brightness) < 1.0e-4f)
        return;
    // ±12 dB spectral tilt centred at mid-band (low cut / high boost as
    // brightness rises). Operates on magnitude.
    for (int k = 0; k < numBins; ++k)
    {
        const float frac = (float) k / (float) (numBins - 1) - 0.5f;   // [-0.5, 0.5]
        mag[k] *= std::pow (10.0f, brightness * frac * (12.0f / 20.0f));
    }
}

void NeuralAudioProcessor::applyFormantShift (const float* mag, float semitones, float* out) const
{
    if (std::abs (semitones) < 1.0e-3f)
    {
        std::copy (mag, mag + numBins, out);
        return;
    }
    // Warp the spectral envelope along frequency: source bin = k / ratio.
    const float ratio = std::pow (2.0f, semitones / 12.0f);
    for (int k = 0; k < numBins; ++k)
    {
        const float src = (float) k / ratio;
        const int i = (int) src;
        if (i >= 0 && i + 1 < numBins)
        {
            const float t = src - (float) i;
            out[k] = mag[i] * (1.0f - t) + mag[i + 1] * t;   // linear interp
        }
        else
        {
            out[k] = 0.0f;
        }
    }
}

void NeuralAudioProcessor::reset()
{
    synth.reset();
    styleInit = false;
    std::fill (styleAvg.begin(), styleAvg.end(), 0.0f);
    if (models != nullptr)
        models->vocoder().reset();
}

void NeuralAudioProcessor::setStyleParams (const StyleParams& p) noexcept
{
    styleParams.store (p);
}

int NeuralAudioProcessor::processFrame (const Features& feats, float* audioOut, int maxSamples)
{
    if (models == nullptr || ! models->isLoaded() || maxSamples < frameSize)
    {
        if (maxSamples > 0)
            std::fill (audioOut, audioOut + maxSamples, 0.0f);
        return maxSamples;
    }

    const StyleParams p = styleParams.load();
    const auto& info = models->info();
    const bool norm = (int) info.melMean.size() == melBins
                   && (int) info.melStd.size()  == melBins;

    // 1) Normalize the mel frame the same way training did.
    for (int i = 0; i < melBins; ++i)
        melNorm[(size_t) i] = norm ? (feats.mel[(size_t) i] - info.melMean[(size_t) i])
                                         / (info.melStd[(size_t) i] + 1e-8f)
                                   : feats.mel[(size_t) i];

    // 2) Encode -> EMA-smooth the style (approximates training's window pooling)
    //    -> style modulation -> decode (all in normalized mel space).
    models->encoder().encode (melNorm.data(), 1, styleVec.data());
    if (! styleInit)
    {
        std::copy (styleVec.begin(), styleVec.end(), styleAvg.begin());
        styleInit = true;
    }
    else
    {
        for (int i = 0; i < styleDim; ++i)
            styleAvg[(size_t) i] = styleAlpha * styleAvg[(size_t) i]
                                 + (1.0f - styleAlpha) * styleVec[(size_t) i];
    }
    styleInterp.apply (styleAvg.data(), p, styleMod.data());
    models->decoder().decode (melNorm.data(), styleMod.data(), melOut.data());

    // 3) Denormalize the transformed mel back to log-mel.
    for (int i = 0; i < melBins; ++i)
        melDenorm[(size_t) i] = norm ? melOut[(size_t) i] * (info.melStd[(size_t) i] + 1e-8f)
                                           + info.melMean[(size_t) i]
                                     : melOut[(size_t) i];

    // 4) Envelope-ratio filter. Rather than rebuild the spectrum from the
    //    decoder mel (decoder-magnitude + input-phase is incoherent -> static),
    //    keep the input's real magnitude AND phase and apply only the smooth
    //    decoder/input spectral-envelope ratio as a gain. This is phase-coherent
    //    (a time-varying filter on the voice), so it reshapes timbre/formants
    //    without the noise, and preserves input content outside the mel band.
    synth.melToMagnitude (feats.mel.data(), inEnv.data());     // input envelope
    synth.melToMagnitude (melDenorm.data(), decEnv.data());    // transformed envelope
    for (int k = 0; k < numBins; ++k)
    {
        float gain = (decEnv[(size_t) k] + 1.0e-6f) / (inEnv[(size_t) k] + 1.0e-6f);
        gain = std::min (gain, 4.0f);                          // cap at +12 dB
        magBuf[(size_t) k] = feats.mag[(size_t) k] * gain;
    }
    applyFormantShift (magBuf.data(), p.formantShift, magWarp.data());
    applyBrightness (magWarp.data(), p.brightness);

    // 5) Resynthesize using the input frame's phase (coherent), with the COLA
    //    gain and a tanh soft-limiter so hot input can't clip.
    synth.magnitudeToFrame (magWarp.data(), feats.phase.data(), audioOut);
    for (int n = 0; n < frameSize; ++n)
        audioOut[n] = std::tanh (audioOut[n] * outputGain);

    return frameSize;
}
