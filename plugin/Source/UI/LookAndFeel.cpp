#include "LookAndFeel.h"

AvtLookAndFeel::AvtLookAndFeel()
{
    setColour (juce::ResizableWindow::backgroundColourId, juce::Colour (0xff1e1e24));
    setColour (juce::Slider::thumbColourId,               juce::Colour (0xff4fc3f7));
    setColour (juce::Slider::rotarySliderFillColourId,    juce::Colour (0xff4fc3f7));
    setColour (juce::Slider::textBoxTextColourId,         juce::Colours::white);
    setColour (juce::Label::textColourId,                 juce::Colours::lightgrey);
}
