#pragma once

#include <juce_audio_utils/juce_audio_utils.h>

#include "PluginProcessor.h"
#include "UI/LookAndFeel.h"
#include "UI/PresetManager.h"
#include "UI/SpectrumAnalyzer.h"
#include "UI/StyleKnob.h"

/**
    Editor UI: style knob plus brightness / formant / pitch sliders, a preset
    manager, and before/after spectrum analyzers.
*/
class AdaptiveVoiceTransformEditor : public juce::AudioProcessorEditor
{
public:
    explicit AdaptiveVoiceTransformEditor (AdaptiveVoiceTransformProcessor&);
    ~AdaptiveVoiceTransformEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;

    AdaptiveVoiceTransformProcessor& processorRef;
    AvtLookAndFeel lookAndFeel;

    StyleKnob       styleKnob;
    juce::Slider    brightnessSlider, formantSlider, pitchSlider;
    juce::Label     brightnessLabel, formantLabel, pitchLabel;

    std::unique_ptr<SliderAttachment> styleAttachment, brightnessAttachment,
                                      formantAttachment, pitchAttachment;

    PresetManager   presetManager;
    SpectrumAnalyzer analyzerBefore, analyzerAfter;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AdaptiveVoiceTransformEditor)
};
