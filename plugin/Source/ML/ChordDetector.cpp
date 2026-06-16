#include "ChordDetector.h"

bool ChordDetector::load (const juce::File& chordnetFile, const juce::File& chordInfoFile)
{
    loaded = false;
    if (! chordnetFile.existsAsFile() || ! chordInfoFile.existsAsFile())
        return false;

    auto info = juce::JSON::parse (chordInfoFile.loadFileAsString());
    if (! info.isObject())
        return false;

    fftSize = (int) info.getProperty ("n_fft", 2048);
    midiLo  = (int) info.getProperty ("midi_lo", 36);
    midiHi  = (int) info.getProperty ("midi_hi", 96);

    // Log-frequency filterbank: nested [nPitch][nBins] -> flat.
    std::vector<float> fb;
    int nBins = 0, nPitch = 0;
    if (auto* rows = info["logfreq_fb"].getArray())
        for (auto& row : *rows)
            if (auto* cols = row.getArray())
            {
                nBins = cols->size();
                ++nPitch;
                for (auto& c : *cols)
                    fb.push_back ((float) (double) c);
            }
    if (nPitch == 0 || nBins == 0)
        return false;

    feature.prepare (fftSize);
    feature.setFilterbank (fb.data(), nPitch, nBins);
    featBuf.assign ((size_t) nPitch, 0.0f);

    if (! net.loadFromFile (chordnetFile))
        return false;

    loaded = true;
    return true;
}

void ChordDetector::detect (const float* frame, int numSamples, float* pc12)
{
    if (! loaded) { std::fill (pc12, pc12 + 12, 0.0f); return; }
    feature.compute (frame, numSamples, featBuf.data());
    net.forward (featBuf.data(), (int) featBuf.size(), pc12);   // sigmoid output (0..1)
}
