#include "StyleKnob.h"

StyleKnob::StyleKnob()
{
    setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    setTextBoxStyle (juce::Slider::TextBoxBelow, false, 80, 20);
    setRange (0.0, 1.0, 0.0);
}

void StyleKnob::paint (juce::Graphics& g)
{
    auto bounds = getLocalBounds().toFloat().reduced (6.0f);
    const auto centre = bounds.getCentre();
    const float radius = juce::jmin (bounds.getWidth(), bounds.getHeight()) * 0.5f;

    const float startAngle = juce::MathConstants<float>::pi * 1.2f;
    const float endAngle   = juce::MathConstants<float>::pi * 2.8f;
    const float value = (float) valueToProportionOfLength (getValue());
    const float angle = startAngle + value * (endAngle - startAngle);

    juce::Path bg;
    bg.addCentredArc (centre.x, centre.y, radius, radius, 0.0f, startAngle, endAngle, true);
    g.setColour (juce::Colours::darkgrey);
    g.strokePath (bg, juce::PathStrokeType (4.0f));

    juce::Path arc;
    arc.addCentredArc (centre.x, centre.y, radius, radius, 0.0f, startAngle, angle, true);
    g.setColour (juce::Colour (0xff4fc3f7));
    g.strokePath (arc, juce::PathStrokeType (4.0f));

    // Pointer Slider::paint draws the text box; we only add the arc here.
    juce::Slider::paint (g);
}
