#pragma once

#include <memory>

#include <juce_audio_utils/juce_audio_utils.h>

#include "PluginProcessor.h"
#include "UI/LookAndFeel.h"
#include "UI/SpectrogramDisplay.h"

/**
    Harmonizer UI: a scrolling colour spectrogram with live chord highlighting,
    Tune / Gate / Polyphony knobs, a readout of the detected chord, and a model
    loader.
*/
class NeuralChordHarmonizerEditor : public juce::AudioProcessorEditor,
                                     private juce::Timer
{
public:
    explicit NeuralChordHarmonizerEditor (NeuralChordHarmonizerProcessor&);
    ~NeuralChordHarmonizerEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;
    void chooseModelsFolder();

    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;

    NeuralChordHarmonizerProcessor& processorRef;
    AvtLookAndFeel lookAndFeel;

    juce::Slider tuneKnob, gateKnob, polyKnob;
    juce::Label  tuneLabel, gateLabel, polyLabel, titleLabel, chordLabel, statusLabel;
    std::unique_ptr<SliderAttachment> tuneAttachment, gateAttachment, polyAttachment;

    juce::TextButton loadButton { "Load Models..." };
    std::unique_ptr<juce::FileChooser> chooser;

    SpectrogramDisplay spectrogram;
    uint32_t lastSpecRead = 0;

    int chordMask = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NeuralChordHarmonizerEditor)
};
