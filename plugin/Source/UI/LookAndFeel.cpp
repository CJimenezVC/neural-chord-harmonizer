#include "LookAndFeel.h"

#include <cmath>

AvtLookAndFeel::AvtLookAndFeel()
{
    setColour (juce::ResizableWindow::backgroundColourId, juce::Colour (0xff1e1e24));
    setColour (juce::Slider::thumbColourId,               juce::Colour (0xff4fc3f7));
    setColour (juce::Slider::rotarySliderFillColourId,    juce::Colour (0xff4fc3f7));
    setColour (juce::Slider::rotarySliderOutlineColourId, juce::Colour (0xff3a3a44));
    setColour (juce::Slider::textBoxTextColourId,         juce::Colours::white);
    setColour (juce::Slider::textBoxOutlineColourId,      juce::Colours::transparentBlack);
    setColour (juce::Label::textColourId,                 juce::Colours::lightgrey);
}

void AvtLookAndFeel::drawRotarySlider (juce::Graphics& g, int x, int y, int width, int height,
                                       float sliderPos, float startAngle, float endAngle,
                                       juce::Slider& slider)
{
    const auto bounds = juce::Rectangle<int> (x, y, width, height).toFloat().reduced (6.0f);
    const auto centre = bounds.getCentre();
    const float radius = juce::jmin (bounds.getWidth(), bounds.getHeight()) * 0.5f;
    const float angle  = startAngle + sliderPos * (endAngle - startAngle);
    const float lineW  = juce::jmax (2.0f, radius * 0.12f);
    const float arcR   = radius - lineW * 0.5f;

    // Background arc.
    juce::Path bg;
    bg.addCentredArc (centre.x, centre.y, arcR, arcR, 0.0f, startAngle, endAngle, true);
    g.setColour (slider.findColour (juce::Slider::rotarySliderOutlineColourId));
    g.strokePath (bg, juce::PathStrokeType (lineW, juce::PathStrokeType::curved,
                                            juce::PathStrokeType::rounded));

    // Value arc.
    juce::Path arc;
    arc.addCentredArc (centre.x, centre.y, arcR, arcR, 0.0f, startAngle, angle, true);
    g.setColour (slider.findColour (juce::Slider::rotarySliderFillColourId));
    g.strokePath (arc, juce::PathStrokeType (lineW, juce::PathStrokeType::curved,
                                             juce::PathStrokeType::rounded));

    // Pointer.
    juce::Point<float> tip (centre.x + arcR * std::cos (angle - juce::MathConstants<float>::halfPi),
                            centre.y + arcR * std::sin (angle - juce::MathConstants<float>::halfPi));
    g.setColour (slider.findColour (juce::Slider::thumbColourId));
    g.drawLine ({ centre, tip }, lineW);
    g.fillEllipse (juce::Rectangle<float> (lineW, lineW).withCentre (centre));
}
