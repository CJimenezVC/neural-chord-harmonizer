#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

/** Dark theme for the plugin UI. */
class AvtLookAndFeel : public juce::LookAndFeel_V4
{
public:
    AvtLookAndFeel();
    ~AvtLookAndFeel() override = default;

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AvtLookAndFeel)
};
