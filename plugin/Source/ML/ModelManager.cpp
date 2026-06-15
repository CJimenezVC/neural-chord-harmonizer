#include "ModelManager.h"

bool ModelManager::loadFromDirectory (const juce::File& dir)
{
    loaded.store (false);

    const auto encFile  = dir.getChildFile ("encoder.rtneural");
    const auto decFile  = dir.getChildFile ("decoder.rtneural");
    const auto vocFile  = dir.getChildFile ("vocoder.rtneural");
    const auto infoFile = dir.getChildFile ("model_info.json");

    if (! (encFile.existsAsFile() && decFile.existsAsFile() && vocFile.existsAsFile()))
        return false;

    parseInfo (infoFile);

    const bool ok = enc.loadModel (encFile)
                 && dec.loadModel (decFile)
                 && voc.loadModel (vocFile);

    if (ok)
        voc.setSamplesPerFrame (modelInfo.hopLength);   // hop samples per frame

    loaded.store (ok);
    return ok;
}

bool ModelManager::parseInfo (const juce::File& infoFile)
{
    if (! infoFile.existsAsFile())
        return false;

    auto json = juce::JSON::parse (infoFile.loadFileAsString());
    if (! json.isObject())
        return false;

    modelInfo.styleDim   = (int) json.getProperty ("style_dim", 64);
    modelInfo.melBins    = (int) json.getProperty ("mel_bins", 128);
    modelInfo.sampleRate = (double) json.getProperty ("sample_rate", 24000.0);
    modelInfo.hopLength  = (int) json.getProperty ("hop_length", 128);

    // Mel normalization stats (the model was trained on (mel - mean) / std).
    auto loadArray = [] (const juce::var& v, std::vector<float>& dst)
    {
        dst.clear();
        if (auto* a = v.getArray())
            for (auto& c : *a)
                dst.push_back ((float) (double) c);
    };
    loadArray (json["mel_mean"], modelInfo.melMean);
    loadArray (json["mel_std"],  modelInfo.melStd);

    // Nested [rows][cols] JSON arrays -> flat row-major; returns the column count.
    auto loadMatrix = [] (const juce::var& v, std::vector<float>& dst) -> int
    {
        dst.clear();
        int cols = 0;
        if (auto* rows = v.getArray())
            for (auto& row : *rows)
                if (auto* c = row.getArray())
                {
                    cols = c->size();
                    for (auto& x : *c)
                        dst.push_back ((float) (double) x);
                }
        return cols;
    };
    modelInfo.melFbBins = loadMatrix (json["mel_fb"], modelInfo.melFb);     // [melBins][nBins]
    loadMatrix (json["inv_mel_fb"], modelInfo.invMelFb);                    // [nBins][melBins]
    return true;
}
