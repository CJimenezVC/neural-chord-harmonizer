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

    auto setupKnob = [this, &p] (juce::Slider& k, juce::Label& l, const juce::String& name,
                                 const juce::String& paramId,
                                 std::unique_ptr<SliderAttachment>& att)
    {
        k.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
        k.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 70, 18);
        addAndMakeVisible (k);
        att = std::make_unique<SliderAttachment> (p.getValueTreeState(), paramId, k);
        l.setText (name, juce::dontSendNotification);
        l.setJustificationType (juce::Justification::centred);
        addAndMakeVisible (l);
    };
    setupKnob (tuneKnob, tuneLabel, "Tune",      "tune",      tuneAttachment);
    setupKnob (gateKnob, gateLabel, "Gate",      "gate",      gateAttachment);
    setupKnob (polyKnob, polyLabel, "Polyphony", "polyphony", polyAttachment);
    tuneKnob.setNumDecimalPlacesToDisplay (3);
    gateKnob.setNumDecimalPlacesToDisplay (3);

    chordLabel.setJustificationType (juce::Justification::centred);
    addAndMakeVisible (chordLabel);

    addAndMakeVisible (loadButton);
    loadButton.onClick = [this] { chooseModelsFolder(); };
    statusLabel.setJustificationType (juce::Justification::centredRight);
    addAndMakeVisible (statusLabel);

    setSize (480, 290);
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

    auto knobRow = area.removeFromTop (150);
    auto labelRow = area.removeFromTop (20);
    const int colW = knobRow.getWidth() / 3;
    juce::Slider* knobs[3]  = { &tuneKnob,  &gateKnob,  &polyKnob };
    juce::Label*  labels[3] = { &tuneLabel, &gateLabel, &polyLabel };
    for (int i = 0; i < 3; ++i)
    {
        auto col = knobRow.removeFromLeft (colW);
        knobs[i]->setBounds (col.withSizeKeepingCentre (120, 130));
        labels[i]->setBounds (labelRow.removeFromLeft (colW));
    }
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
