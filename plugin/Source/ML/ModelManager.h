#pragma once

#include <atomic>

#include <juce_core/juce_core.h>

#include "EncoderNetwork.h"
#include "DecoderNetwork.h"
#include "VocoderNetwork.h"

/** Metadata loaded from model_info.json (dims + normalization stats). */
struct ModelInfo
{
    int styleDim = 64;
    int melBins = 128;
    double sampleRate = 24000.0;
    int hopLength = 128;
    std::vector<float> melMean, melStd;
    std::vector<float> melFb;     // [melBins * melFbBins] row-major (Slaney filterbank)
    int melFbBins = 0;            // = n_fft/2 + 1
    std::vector<float> invMelFb;  // [melFbBins * melBins] row-major (pinv of melFb)

    // Voice-conversion mode: learned target-speaker embeddings + names.
    bool conversion = false;
    int  numTargets = 0;
    std::vector<float> speakerEmb;          // [numTargets * styleDim] row-major
    juce::StringArray  targetNames;
};

/**
    Loads and owns the three RTNeural networks plus model metadata. Loading
    happens on the message thread; isLoaded() is queried from the audio thread.
*/
class ModelManager
{
public:
    /** Load encoder/decoder/vocoder + model_info.json from @p dir. */
    bool loadFromDirectory (const juce::File& dir);

    bool isLoaded() const noexcept { return loaded.load(); }

    EncoderNetwork& encoder() noexcept { return enc; }
    DecoderNetwork& decoder() noexcept { return dec; }
    VocoderNetwork& vocoder() noexcept { return voc; }
    const ModelInfo& info() const noexcept { return modelInfo; }

private:
    bool parseInfo (const juce::File& infoFile);

    EncoderNetwork enc;
    DecoderNetwork dec;
    VocoderNetwork voc;
    ModelInfo modelInfo;

    std::atomic<bool> loaded { false };
};
