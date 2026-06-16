#include "PluginEditor.h"

namespace { const char* kNoteNames[12] =
    { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" }; }

NeuralChordHarmonizerEditor::NeuralChordHarmonizerEditor (NeuralChordHarmonizerProcessor& p)
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
    setupKnob (tuneKnob,    tuneLabel,    "Tune",      "tune",      tuneAttachment);
    setupKnob (gateKnob,    gateLabel,    "Gate",      "gate",      gateAttachment);
    setupKnob (attackKnob,  attackLabel,  "Attack",    "attack",    attackAttachment);
    setupKnob (releaseKnob, releaseLabel, "Release",   "release",   releaseAttachment);
    setupKnob (polyKnob,    polyLabel,    "Polyphony", "polyphony", polyAttachment);

    addAndMakeVisible (spectrogram);
    spectrogram.configure (processorRef.getSpectrumBins(),
                           processorRef.getSpectrumCapacity(), /*midiLo*/ 36);
    lastSpecRead = processorRef.getSpectrumWriteIndex();

    chordLabel.setJustificationType (juce::Justification::centred);
    chordLabel.setText ("(listening)", juce::dontSendNotification);
    chordLabel.setColour (juce::Label::textColourId, juce::Colours::grey);
    addAndMakeVisible (chordLabel);

    addAndMakeVisible (loadButton);
    loadButton.onClick = [this] { chooseModelsFolder(); };
    statusLabel.setJustificationType (juce::Justification::centredRight);
    addAndMakeVisible (statusLabel);

    setSize (660, 470);
    startTimerHz (30);
}

NeuralChordHarmonizerEditor::~NeuralChordHarmonizerEditor()
{
    stopTimer();
    setLookAndFeel (nullptr);
}

void NeuralChordHarmonizerEditor::timerCallback()
{
    // Drain any new spectrogram columns the audio thread has produced.
    const uint32_t w   = processorRef.getSpectrumWriteIndex();
    const int      cap = processorRef.getSpectrumCapacity();
    const int      bins = processorRef.getSpectrumBins();
    uint32_t from = lastSpecRead;
    if (w - from > (uint32_t) cap) from = w - (uint32_t) cap;   // we fell behind: skip ahead
    float col[64];
    for (uint32_t i = from; i != w; ++i)
    {
        processorRef.readSpectrumColumn (i, col);
        spectrogram.pushColumn (col, bins);
    }
    lastSpecRead = w;

    const int mask = processorRef.getChordMask();
    spectrogram.setChordMask (mask);
    if (mask != chordMask)
    {
        chordMask = mask;
        juce::StringArray notes;
        for (int i = 0; i < 12; ++i)
            if (mask & (1 << i)) notes.add (kNoteNames[i]);
        chordLabel.setText (notes.isEmpty() ? "(listening)" : notes.joinIntoString ("  "),
                            juce::dontSendNotification);
        chordLabel.setColour (juce::Label::textColourId,
                              mask ? juce::Colour (0xff2dffb0) : juce::Colours::grey);
    }
    const bool loaded = processorRef.modelsLoaded();
    statusLabel.setText (loaded ? "Models: loaded" : "Models: none", juce::dontSendNotification);
    statusLabel.setColour (juce::Label::textColourId,
                           loaded ? juce::Colours::lightgreen : juce::Colours::orange);
}

void NeuralChordHarmonizerEditor::paint (juce::Graphics& g)
{
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
}

void NeuralChordHarmonizerEditor::resized()
{
    auto area = getLocalBounds().reduced (14);
    auto top = area.removeFromTop (30);
    titleLabel.setBounds (top.removeFromLeft (180));
    loadButton.setBounds (top.removeFromRight (110).reduced (0, 2));
    statusLabel.setBounds (top.removeFromRight (130));

    // Knobs at the bottom.
    auto labelRow = area.removeFromBottom (20);
    auto knobRow  = area.removeFromBottom (140);
    area.removeFromBottom (6);

    // Chord readout just above the knobs.
    chordLabel.setBounds (area.removeFromBottom (28));
    chordLabel.setFont (juce::Font (20.0f, juce::Font::bold));
    area.removeFromBottom (8);

    // Spectrogram fills the remaining (largest) area.
    area.removeFromTop (8);
    spectrogram.setBounds (area);

    juce::Slider* knobs[5]  = { &tuneKnob,  &gateKnob,  &attackKnob,  &releaseKnob,  &polyKnob };
    juce::Label*  labels[5] = { &tuneLabel, &gateLabel, &attackLabel, &releaseLabel, &polyLabel };
    const int colW = knobRow.getWidth() / 5;
    const int knobW = juce::jmin (110, colW - 6);
    for (int i = 0; i < 5; ++i)
    {
        auto col = knobRow.removeFromLeft (colW);
        knobs[i]->setBounds (col.withSizeKeepingCentre (knobW, knobW + 18));
        labels[i]->setBounds (labelRow.removeFromLeft (colW));
    }
}

void NeuralChordHarmonizerEditor::chooseModelsFolder()
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
