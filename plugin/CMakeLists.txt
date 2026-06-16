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
    # Explicit reverse-DNS bundle id (no spaces — JUCE warns if derived from a
    # company name containing spaces).
    BUNDLE_ID                 com.cristobaljimenez.adaptivevoicetransform
    PLUGIN_MANUFACTURER_CODE  Cjim
    PLUGIN_CODE               Avtf
    FORMATS                   VST3 AU Standalone
    PRODUCT_NAME              "Adaptive Voice Transform"
    IS_SYNTH                  FALSE
    NEEDS_MIDI_INPUT          FALSE
    # The standalone captures live audio input; on macOS this is required or the
    # OS silently provides no input (NSMicrophoneUsageDescription + entitlement).
    MICROPHONE_PERMISSION_ENABLED TRUE
    MICROPHONE_PERMISSION_TEXT "Adaptive Voice Transform processes microphone/line input."
    COPY_PLUGIN_AFTER_BUILD   FALSE)

juce_generate_juce_header(AdaptiveVoiceTransform)

# Chord-following harmonizer. (Voice-conversion sources from the earlier
# direction remain in the tree but are no longer built.)
target_sources(AdaptiveVoiceTransform PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp

    Source/DSP/YINPitchDetector.cpp
    Source/DSP/OverlapAddBuffer.cpp

    Source/ML/NNModel.cpp
    Source/ML/ChordDetector.cpp

    Source/UI/LookAndFeel.cpp)

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
