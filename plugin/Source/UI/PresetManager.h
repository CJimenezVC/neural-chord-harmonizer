#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>

/**
    Save/load named presets backed by the AudioProcessorValueTreeState. Presets
    are stored as XML under the user application data directory.
*/
class PresetManager : public juce::Component
{
public:
    explicit PresetManager (juce::AudioProcessorValueTreeState& state);
    ~PresetManager() override = default;

    void resized() override;

    void savePreset (const juce::String& name);
    void loadPreset (const juce::String& name);
    juce::StringArray getPresetNames() const;

private:
    juce::File getPresetDirectory() const;

    juce::AudioProcessorValueTreeState& apvts;
    juce::ComboBox presetBox;
    juce::TextButton saveButton { "Save" };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PresetManager)
};
