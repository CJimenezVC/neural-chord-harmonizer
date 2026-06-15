#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

/**
    Large rotary control for the primary "Style Shift" parameter. Rendered by
    AvtLookAndFeel::drawRotarySlider; behaves as a juce::Slider so it binds to an
    APVTS attachment.
*/
class StyleKnob : public juce::Slider
{
public:
    StyleKnob();
    ~StyleKnob() override = default;

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (StyleKnob)
};
