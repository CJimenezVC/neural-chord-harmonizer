#include "EncoderNetwork.h"

#include <juce_core/juce_core.h>

bool EncoderNetwork::loadModel (const juce::File& jsonFile)
{
    return model.loadFromFile (jsonFile);
}

void EncoderNetwork::encode (const float* mel, int /*numFrames*/, float* styleOut)
{
    // Streaming: encode the current mel frame into a style vector.
    model.forward (mel, model.inputSize(), styleOut);
}
