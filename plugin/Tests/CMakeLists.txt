# Lightweight, dependency-free unit tests for the JUCE-independent DSP
# primitives. Heavier integration tests (full plugin) run via pluginval.

add_executable(avt_tests
    test_main.cpp
    test_dsp.cpp)

target_include_directories(avt_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../Source)

# These primitives are header-only / self-contained, so the test target needs a
# couple of the .cpp units but no JUCE.
target_sources(avt_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../Source/DSP/OverlapAddBuffer.cpp)

set_target_properties(avt_tests PROPERTIES CXX_STANDARD 20)

add_test(NAME avt_tests COMMAND avt_tests)
