cmake_minimum_required(VERSION 3.22)

project(AdaptiveVoiceTransform VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(AVT_BUILD_TESTS "Build unit tests" OFF)

# ---------------------------------------------------------------------------
# Dependencies: JUCE + RTNeural (fetched automatically)
# Override with -DJUCE_PATH=/path or -DRTNEURAL_PATH=/path to use local checkouts.
# ---------------------------------------------------------------------------
include(FetchContent)

if(DEFINED JUCE_PATH)
    add_subdirectory(${JUCE_PATH} JUCE)
else()
    FetchContent_Declare(JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG        8.0.4)
    FetchContent_MakeAvailable(JUCE)
endif()

if(DEFINED RTNEURAL_PATH)
    add_subdirectory(${RTNEURAL_PATH} RTNeural)
else()
    FetchContent_Declare(RTNeural
        GIT_REPOSITORY https://github.com/jatinchowdhury18/RTNeural.git
        GIT_TAG        main)
    FetchContent_MakeAvailable(RTNeural)
endif()

# ---------------------------------------------------------------------------
# Plugin target
# ---------------------------------------------------------------------------
juce_add_plugin(AdaptiveVoiceTransform
    COMPANY_NAME              "Cristobal Jimenez"
    PLUGIN_MANUFACTURER_CODE  Cjim
    PLUGIN_CODE               Avtf
    FORMATS                   VST3 AU Standalone
    PRODUCT_NAME              "Adaptive Voice Transform"
    IS_SYNTH                  FALSE
    NEEDS_MIDI_INPUT          FALSE
    COPY_PLUGIN_AFTER_BUILD   FALSE)

juce_generate_juce_header(AdaptiveVoiceTransform)

target_sources(AdaptiveVoiceTransform PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp

    Source/DSP/FeatureExtractor.cpp
    Source/DSP/YINPitchDetector.cpp
    Source/DSP/FormantAnalyzer.cpp
    Source/DSP/SpectrogramProcessor.cpp
    Source/DSP/OverlapAddBuffer.cpp

    Source/ML/ModelManager.cpp
    Source/ML/EncoderNetwork.cpp
    Source/ML/DecoderNetwork.cpp
    Source/ML/VocoderNetwork.cpp
    Source/ML/NeuralAudioProcessor.cpp

    Source/UI/StyleKnob.cpp
    Source/UI/PresetManager.cpp
    Source/UI/LookAndFeel.cpp
    Source/UI/SpectrumAnalyzer.cpp

    Source/Utilities/Performance.cpp
    Source/Utilities/Logger.cpp)

target_include_directories(AdaptiveVoiceTransform PRIVATE Source)

target_compile_definitions(AdaptiveVoiceTransform PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0)

target_link_libraries(AdaptiveVoiceTransform PRIVATE
    juce::juce_audio_utils
    juce::juce_dsp
    RTNeural
    PUBLIC
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(AVT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(Tests)
endif()
