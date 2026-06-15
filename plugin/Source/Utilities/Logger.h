#pragma once

#include <juce_core/juce_core.h>

/**
    Thin wrapper around juce::Logger for plugin diagnostics. Writes to a log
    file under the user application data directory; no-ops in release if
    disabled. Never call from the audio thread.
*/
class AvtLogger
{
public:
    static void init();
    static void log (const juce::String& message);
    static juce::File getLogFile();
};

#if JUCE_DEBUG
    #define AVT_LOG(msg) AvtLogger::log (msg)
#else
    #define AVT_LOG(msg) juce::ignoreUnused (msg)
#endif
