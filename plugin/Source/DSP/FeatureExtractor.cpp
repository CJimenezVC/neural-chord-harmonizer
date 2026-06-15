#include "FeatureExtractor.h"

void FeatureExtractor::prepare (double sampleRate, int frame, int hop, int nMels)
{
    sr = sampleRate;
    frameSize = frame;
    hopSize = hop;
    numMels = nMels;

    spectrogram.prepare (sampleRate, frameSize, nMels);
    pitch.prepare (sampleRate, frameSize);
    formants.prepare (sampleRate, /*lpcOrder*/ 12);

    scratch.mel.assign ((size_t) nMels, 0.0f);
    magnitude.assign ((size_t) (frameSize / 2 + 1), 0.0f);
}

void FeatureExtractor::reset()
{
    spectrogram.reset();
    pitch.reset();
}

Features FeatureExtractor::process (const float* frame, int numSamples)
{
    // 1) STFT magnitude + log-mel projection.
    spectrogram.computeMagnitude (frame, numSamples, magnitude.data());
    spectrogram.magnitudeToLogMel (magnitude.data(), scratch.mel.data());

    // 2) Pitch (YIN) with confidence.
    const auto [f0, conf] = pitch.detect (frame, numSamples);
    scratch.f0 = f0;
    scratch.voicedConfidence = conf;

    // 3) Formants (LPC roots).
    scratch.formants = formants.estimate (frame, numSamples);

    return scratch;
}
