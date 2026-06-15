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

    auto initLabel = [this] (juce::Label& l, const juce::String& text, float fontHeight)
    {
        l.setText (text, juce::dontSendNotification);
        l.setJustificationType (juce::Justification::centred);
        l.setFont (juce::Font (fontHeight));
        addAndMakeVisible (l);
    };

    addAndMakeVisible (styleKnob);
    styleKnob.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 80, 20);
    initLabel (styleLabel, "Style Shift", 15.0f);

    configureSlider (brightnessSlider, brightnessLabel, "Brightness", *this);
    configureSlider (formantSlider,    formantLabel,    "Formant",    *this);
    formantSlider.setTextValueSuffix (" st");

    addAndMakeVisible (presetManager);
    addAndMakeVisible (analyzerBefore);
    addAndMakeVisible (analyzerAfter);
    initLabel (beforeLabel, "Input",  13.0f);
    initLabel (afterLabel,  "Output", 13.0f);

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
    g.setFont (juce::Font (18.0f, juce::Font::bold));
    g.drawText ("Adaptive Voice Transform",
                getLocalBounds().removeFromTop (44).reduced (14, 0),
                juce::Justification::centredLeft);
}

void AdaptiveVoiceTransformEditor::resized()
{
    auto area = getLocalBounds().reduced (12);

    auto titleRow = area.removeFromTop (32);              // title + model controls
    loadModelsButton.setBounds (titleRow.removeFromRight (110).reduced (0, 4));
    modelStatusLabel.setBounds (titleRow.removeFromRight (160));

    // Analyzers, each with a caption above.
    auto top = area.removeFromTop (190);
    auto beforeArea = top.removeFromLeft (top.getWidth() / 2).reduced (4);
    auto afterArea  = top.reduced (4);
    beforeLabel.setBounds (beforeArea.removeFromTop (16));
    afterLabel.setBounds  (afterArea.removeFromTop (16));
    analyzerBefore.setBounds (beforeArea);
    analyzerAfter.setBounds  (afterArea);

    area.removeFromTop (6);
    presetManager.setBounds (area.removeFromBottom (28));
    area.removeFromBottom (6);

    // Control grid: 3 equal columns, each = name label (top) + knob (below).
    auto controls = area;
    juce::Component* knobs[]  = { &styleKnob, &brightnessSlider, &formantSlider };
    juce::Label*     labels[] = { &styleLabel, &brightnessLabel, &formantLabel };
    const int colW = controls.getWidth() / 3;
    for (int i = 0; i < 3; ++i)
    {
        auto col = (i == 2) ? controls : controls.removeFromLeft (colW);
        labels[i]->setBounds (col.removeFromTop (18));
        knobs[i]->setBounds (col.reduced (8, 2));
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
