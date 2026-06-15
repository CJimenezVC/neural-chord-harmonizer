#pragma once

#include <juce_audio_utils/juce_audio_utils.h>

#include "PluginProcessor.h"
#include "UI/LookAndFeel.h"
#include "UI/PresetManager.h"
#include "UI/SpectrumAnalyzer.h"
#include "UI/StyleKnob.h"

/**
    Editor UI: style knob plus brightness / formant sliders, a model loader, a
    preset manager, and before/after spectrum analyzers.
*/
class AdaptiveVoiceTransformEditor : public juce::AudioProcessorEditor,
                                     private juce::Timer
{
public:
    explicit AdaptiveVoiceTransformEditor (AdaptiveVoiceTransformProcessor&);
    ~AdaptiveVoiceTransformEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;   // pumps the scope taps into the analyzers

    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;

    AdaptiveVoiceTransformProcessor& processorRef;
    AvtLookAndFeel lookAndFeel;

    StyleKnob       styleKnob;
    juce::Slider    brightnessSlider, formantSlider;
    juce::Label     styleLabel, brightnessLabel, formantLabel;
    juce::Label     beforeLabel, afterLabel;

    std::unique_ptr<SliderAttachment> styleAttachment, brightnessAttachment,
                                      formantAttachment;

    PresetManager   presetManager;
    SpectrumAnalyzer analyzerBefore, analyzerAfter;

    juce::TextButton loadModelsButton { "Load Models..." };
    juce::Label      modelStatusLabel;
    juce::ComboBox   targetBox;          // target speaker (conversion mode)
    juce::Label      targetLabel;
    std::unique_ptr<juce::FileChooser> chooser;
    std::vector<float> scopeScratch;   // drains the processor's scope FIFOs

    void chooseModelsFolder();
    void refreshModelStatus();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdaptiveVoiceTransformEditor)
};
