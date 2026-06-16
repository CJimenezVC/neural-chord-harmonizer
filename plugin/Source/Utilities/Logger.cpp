#include "Logger.h"

juce::File AvtLogger::getLogFile()
{
    return juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
               .getChildFile ("NeuralChordHarmonizer")
               .getChildFile ("avt.log");
}

void AvtLogger::init()
{
    auto file = getLogFile();
    file.getParentDirectory().createDirectory();
    juce::Logger::setCurrentLogger (nullptr);
    static std::unique_ptr<juce::FileLogger> logger (
        new juce::FileLogger (file, "Neural Chord Harmonizer", 0));
    juce::Logger::setCurrentLogger (logger.get());
}

void AvtLogger::log (const juce::String& message)
{
    juce::Logger::writeToLog (message);
}
