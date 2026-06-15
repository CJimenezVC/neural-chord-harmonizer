#include "NeuralAudioProcessor.h"

void NeuralAudioProcessor::prepare (double /*sampleRate*/, int frame)
{
    frameSize = frame;
    styleInterp.prepare (styleDim);

    styleVec.assign ((size_t) styleDim, 0.0f);
    styleMod.assign ((size_t) styleDim, 0.0f);
    melOut.assign ((size_t) melBins, 0.0f);
}

void NeuralAudioProcessor::reset()
{
    if (models != nullptr)
        models->vocoder().reset();
}

void NeuralAudioProcessor::setStyleParams (const StyleParams& p) noexcept
{
    styleParams.store (p);
}

int NeuralAudioProcessor::processFrame (const Features& feats, float* audioOut, int maxSamples)
{
    if (models == nullptr || ! models->isLoaded())
    {
        if (maxSamples > 0)
            std::fill (audioOut, audioOut + maxSamples, 0.0f);
        return maxSamples;
    }

    const StyleParams p = styleParams.load();

    // 1) Encode the mel frame into a style vector.
    models->encoder().encode (feats.mel.data(), 1, styleVec.data());

    // 2) Modulate the style with the user parameters.
    styleInterp.apply (styleVec.data(), p, styleMod.data());

    // 3) Decode into a transformed mel frame.
    models->decoder().decode (feats.mel.data(), styleMod.data(), melOut.data());

    // 4) Vocode the transformed mel frame to audio.
    return models->vocoder().synthesize (melOut.data(), audioOut, maxSamples);
}
