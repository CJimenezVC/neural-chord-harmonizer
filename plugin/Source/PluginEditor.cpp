#include "PluginEditor.h"

namespace
{
    void configureSlider (juce::Slider& s, juce::Label& l, const juce::String& name,
                          juce::Component& parent)
    {
        s.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
        s.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 64, 18);
        l.setText (name, juce::dontSendNotification);
        l.setJustificationType (juce::Justification::centred);
        parent.addAndMakeVisible (s);
        parent.addAndMakeVisible (l);
    }
}

AdaptiveVoiceTransformEditor::AdaptiveVoiceTransformEditor (AdaptiveVoiceTransformProcessor& p)
    : AudioProcessorEditor (&p), processorRef (p),
      presetManager (p.getValueTreeState())
{
    setLookAndFeel (&lookAndFeel);

    addAndMakeVisible (styleKnob);
    configureSlider (brightnessSlider, brightnessLabel, "Brightness", *this);
    configureSlider (formantSlider,    formantLabel,    "Formant",    *this);
    configureSlider (pitchSlider,      pitchLabel,      "Pitch",      *this);

    addAndMakeVisible (presetManager);
    addAndMakeVisible (analyzerBefore);
    addAndMakeVisible (analyzerAfter);

    addAndMakeVisible (loadModelsButton);
    loadModelsButton.onClick = [this] { chooseModelsFolder(); };
    addAndMakeVisible (modelStatusLabel);
    modelStatusLabel.setJustificationType (juce::Justification::centredRight);
    refreshModelStatus();

    scopeScratch.assign (4096, 0.0f);
    startTimerHz (30);   // pump scope taps into the before/after analyzers

    auto& apvts = p.getValueTreeState();
    styleAttachment      = std::make_unique<SliderAttachment> (apvts, "styleShift",   styleKnob);
    brightnessAttachment = std::make_unique<SliderAttachment> (apvts, "brightness",   brightnessSlider);
    formantAttachment    = std::make_unique<SliderAttachment> (apvts, "formantShift", formantSlider);
    pitchAttachment      = std::make_unique<SliderAttachment> (apvts, "pitchShift",   pitchSlider);

    setSize (720, 420);
}

AdaptiveVoiceTransformEditor::~AdaptiveVoiceTransformEditor()
{
    stopTimer();
    setLookAndFeel (nullptr);
}

void AdaptiveVoiceTransformEditor::timerCallback()
{
    const int n = (int) scopeScratch.size();

    int got = processorRef.readScopeBefore (scopeScratch.data(), n);
    for (int i = 0; i < got; ++i)
        analyzerBefore.pushSample (scopeScratch[(size_t) i]);

    got = processorRef.readScopeAfter (scopeScratch.data(), n);
    for (int i = 0; i < got; ++i)
        analyzerAfter.pushSample (scopeScratch[(size_t) i]);

    refreshModelStatus();
}

void AdaptiveVoiceTransformEditor::paint (juce::Graphics& g)
{
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
    g.setColour (juce::Colours::white);
    g.setFont (20.0f);
    g.drawText ("Adaptive Voice Transform", getLocalBounds().removeFromTop (32),
                juce::Justification::centred);
}

void AdaptiveVoiceTransformEditor::resized()
{
    auto area = getLocalBounds().reduced (12);

    auto titleRow = area.removeFromTop (32);              // title + model controls
    loadModelsButton.setBounds (titleRow.removeFromRight (110).reduced (0, 4));
    modelStatusLabel.setBounds (titleRow.removeFromRight (160));

    auto top = area.removeFromTop (200);
    analyzerBefore.setBounds (top.removeFromLeft (top.getWidth() / 2).reduced (4));
    analyzerAfter.setBounds  (top.reduced (4));

    auto controls = area.removeFromTop (150);
    styleKnob.setBounds (controls.removeFromLeft (180).reduced (8));

    const int w = controls.getWidth() / 3;
    for (auto* pair : { &brightnessSlider, &formantSlider, &pitchSlider })
    {
        auto col = controls.removeFromLeft (w);
        pair->setBounds (col.reduced (8).withTrimmedBottom (18));
    }
    brightnessLabel.setBounds (styleKnob.getRight(),          controls.getY(), w, 18);

    presetManager.setBounds (area.removeFromBottom (28));
}

void AdaptiveVoiceTransformEditor::chooseModelsFolder()
{
    chooser = std::make_unique<juce::FileChooser> (
        "Select the models/pretrained folder", juce::File{}, juce::String{});

    const auto flags = juce::FileBrowserComponent::openMode
                     | juce::FileBrowserComponent::canSelectDirectories;

    chooser->launchAsync (flags, [this] (const juce::FileChooser& fc)
    {
        const auto dir = fc.getResult();
        if (dir.isDirectory())
        {
            processorRef.loadModels (dir);
            refreshModelStatus();
        }
    });
}

void AdaptiveVoiceTransformEditor::refreshModelStatus()
{
    const bool loaded = processorRef.getModelManager().isLoaded();
    modelStatusLabel.setText (loaded ? "Models: loaded" : "Models: none",
                              juce::dontSendNotification);
    modelStatusLabel.setColour (juce::Label::textColourId,
                                loaded ? juce::Colours::lightgreen : juce::Colours::orange);
}
