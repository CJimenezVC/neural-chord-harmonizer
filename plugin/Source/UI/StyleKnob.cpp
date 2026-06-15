#include "StyleKnob.h"

StyleKnob::StyleKnob()
{
    setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    setTextBoxStyle (juce::Slider::TextBoxBelow, false, 80, 20);
    setRange (0.0, 1.0, 0.0);
}

// Rendering is handled by AvtLookAndFeel::drawRotarySlider so every knob looks
// the same and nothing is drawn twice.
