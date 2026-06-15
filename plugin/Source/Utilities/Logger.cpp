#include "Logger.h"

juce::File AvtLogger::getLogFile()
{
    return juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
               .getChildFile ("AdaptiveVoiceTransform")
               .getChildFile ("avt.log");
}

void AvtLogger::init()
{
    auto file = getLogFile();
    file.getParentDirectory().createDirectory();
    juce::Logger::setCurrentLogger (nullptr);
    static std::unique_ptr<juce::FileLogger> logger (
        new juce::FileLogger (file, "Adaptive Voice Transform", 0));
    juce::Logger::setCurrentLogger (logger.get());
}

void AvtLogger::log (const juce::String& message)
{
    juce::Logger::writeToLog (message);
}
