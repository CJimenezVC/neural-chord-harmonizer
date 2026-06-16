#include "PluginEditor.h"

namespace { const char* kNoteNames[12] =
    { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" }; }

AdaptiveVoiceTransformEditor::AdaptiveVoiceTransformEditor (AdaptiveVoiceTransformProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p)
{
    setLookAndFeel (&lookAndFeel);

    titleLabel.setText ("Chord Harmonizer", juce::dontSendNotification);
    titleLabel.setFont (juce::Font (18.0f, juce::Font::bold));
    addAndMakeVisible (titleLabel);

    tuneKnob.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    tuneKnob.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 80, 20);
    addAndMakeVisible (tuneKnob);
    tuneAttachment = std::make_unique<SliderAttachment> (p.getValueTreeState(), "tune", tuneKnob);
    tuneLabel.setText ("Tune  (natural ... tight)", juce::dontSendNotification);
    tuneLabel.setJustificationType (juce::Justification::centred);
    addAndMakeVisible (tuneLabel);

    chordLabel.setJustificationType (juce::Justification::centred);
    addAndMakeVisible (chordLabel);

    addAndMakeVisible (loadButton);
    loadButton.onClick = [this] { chooseModelsFolder(); };
    statusLabel.setJustificationType (juce::Justification::centredRight);
    addAndMakeVisible (statusLabel);

    setSize (440, 300);
    startTimerHz (30);
}

AdaptiveVoiceTransformEditor::~AdaptiveVoiceTransformEditor()
{
    stopTimer();
    setLookAndFeel (nullptr);
}

void AdaptiveVoiceTransformEditor::timerCallback()
{
    const int mask = processorRef.getChordMask();
    if (mask != chordMask)
    {
        chordMask = mask;
        juce::StringArray notes;
        for (int i = 0; i < 12; ++i)
            if (mask & (1 << i)) notes.add (kNoteNames[i]);
        chordLabel.setText (notes.isEmpty() ? "(listening)" : notes.joinIntoString (" "),
                            juce::dontSendNotification);
    }
    const bool loaded = processorRef.modelsLoaded();
    statusLabel.setText (loaded ? "Models: loaded" : "Models: none", juce::dontSendNotification);
    statusLabel.setColour (juce::Label::textColourId,
                           loaded ? juce::Colours::lightgreen : juce::Colours::orange);
}

void AdaptiveVoiceTransformEditor::paint (juce::Graphics& g)
{
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
}

void AdaptiveVoiceTransformEditor::resized()
{
    auto area = getLocalBounds().reduced (14);
    auto top = area.removeFromTop (30);
    titleLabel.setBounds (top.removeFromLeft (180));
    loadButton.setBounds (top.removeFromRight (110).reduced (0, 2));
    statusLabel.setBounds (top.removeFromRight (130));

    area.removeFromTop (8);
    chordLabel.setBounds (area.removeFromTop (40));
    chordLabel.setFont (juce::Font (22.0f, juce::Font::bold));

    auto knob = area.removeFromTop (170);
    tuneKnob.setBounds (knob.withSizeKeepingCentre (160, 160));
    tuneLabel.setBounds (area.removeFromTop (20));
}

void AdaptiveVoiceTransformEditor::chooseModelsFolder()
{
    chooser = std::make_unique<juce::FileChooser> (
        "Select the models/pretrained folder", juce::File{}, juce::String{});
    const auto flags = juce::FileBrowserComponent::openMode
                     | juce::FileBrowserComponent::canSelectDirectories;
    chooser->launchAsync (flags, [this] (const juce::FileChooser& fc)
    {
        if (fc.getResult().isDirectory())
            processorRef.loadModels (fc.getResult());
    });
}
