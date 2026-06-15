#include "PresetManager.h"

PresetManager::PresetManager (juce::AudioProcessorValueTreeState& state)
    : apvts (state)
{
    addAndMakeVisible (presetBox);
    addAndMakeVisible (saveButton);

    presetBox.setTextWhenNothingSelected ("Default");
    presetBox.addItemList (getPresetNames(), 1);
    presetBox.onChange = [this]
    {
        if (presetBox.getText().isNotEmpty())
            loadPreset (presetBox.getText());
    };

    saveButton.onClick = [this]
    {
        // A real UI would prompt for a name; use a timestamp for now.
        savePreset ("Preset " + juce::Time::getCurrentTime().toString (false, true));
    };
}

void PresetManager::resized()
{
    auto area = getLocalBounds();
    saveButton.setBounds (area.removeFromRight (60));
    presetBox.setBounds (area.reduced (2, 0));
}

juce::File PresetManager::getPresetDirectory() const
{
    auto dir = juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
                   .getChildFile ("AdaptiveVoiceTransform")
                   .getChildFile ("Presets");
    dir.createDirectory();
    return dir;
}

void PresetManager::savePreset (const juce::String& name)
{
    if (auto xml = apvts.copyState().createXml())
        xml->writeTo (getPresetDirectory().getChildFile (name + ".xml"));

    presetBox.clear (juce::dontSendNotification);
    presetBox.addItemList (getPresetNames(), 1);
}

void PresetManager::loadPreset (const juce::String& name)
{
    const auto file = getPresetDirectory().getChildFile (name + ".xml");
    if (auto xml = juce::XmlDocument::parse (file))
        apvts.replaceState (juce::ValueTree::fromXml (*xml));
}

juce::StringArray PresetManager::getPresetNames() const
{
    juce::StringArray names;
    for (const auto& f : getPresetDirectory().findChildFiles (juce::File::findFiles, false, "*.xml"))
        names.add (f.getFileNameWithoutExtension());
    return names;
}
