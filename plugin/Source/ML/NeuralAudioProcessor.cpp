#include "NeuralAudioProcessor.h"

#include <algorithm>

void NeuralAudioProcessor::prepare (double sampleRate, int frame)
{
    frameSize = frame;
    styleInterp.prepare (styleDim);
    synth.prepare (sampleRate, frameSize, melBins, fMin, fMax);

    melNorm.assign   ((size_t) melBins, 0.0f);
    styleVec.assign  ((size_t) styleDim, 0.0f);
    styleMod.assign  ((size_t) styleDim, 0.0f);
    melOut.assign    ((size_t) melBins, 0.0f);
    melDenorm.assign ((size_t) melBins, 0.0f);
}

void NeuralAudioProcessor::reset()
{
    synth.reset();
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

    // 2) Encode -> style modulation -> decode (all in normalized mel space).
    models->encoder().encode (melNorm.data(), 1, styleVec.data());
    styleInterp.apply (styleVec.data(), p, styleMod.data());
    models->decoder().decode (melNorm.data(), styleMod.data(), melOut.data());

    // 3) Denormalize the transformed mel back to log-mel.
    for (int i = 0; i < melBins; ++i)
        melDenorm[(size_t) i] = norm ? melOut[(size_t) i] * (info.melStd[(size_t) i] + 1e-8f)
                                           + info.melMean[(size_t) i]
                                     : melOut[(size_t) i];

    // 4) Resynthesize a waveform frame from the transformed mel + input phase.
    synth.melToFrame (melDenorm.data(), feats.phase.data(), audioOut);
    for (int n = 0; n < frameSize; ++n)
        audioOut[n] *= outputGain;

    return frameSize;
}
