#pragma once

/**
    Plain-old-data view of the user-facing transform parameters, as read off the
    AudioProcessorValueTreeState each block and handed to the neural processor.
    Kept allocation- and lock-free for the audio thread.
*/
struct StyleParams
{
    float styleShift   = 0.0f;   // [0,1]   source <-> target blend
    float brightness   = 0.0f;   // [-1,1]  HF emphasis
    float formantShift = 0.0f;   // semitones [-12,12]
    float pitchShift   = 0.0f;   // semitones [-24,24]
};
