# Plugin Build Guide

## Prerequisites

| Tool      | Version | Notes                                   |
| --------- | ------- | --------------------------------------- |
| CMake     | ≥ 3.22  | FetchContent for JUCE/RTNeural          |
| Compiler  | C++20   | MSVC 2022 / Clang 14+ / GCC 11+         |
| Git       | any     | Required for FetchContent               |

JUCE and RTNeural are fetched automatically by CMake — no submodules or system
installs required. (You can override with `-DJUCE_PATH=` / `-DRTNEURAL_PATH=`
to use local checkouts.)

### Platform extras

- **macOS:** Xcode command-line tools (`xcode-select --install`).
- **Linux:** ALSA/JACK + X11 dev packages:
  ```bash
  sudo apt install libasound2-dev libjack-jackd2-dev libx11-dev \
      libxcomposite-dev libxcursor-dev libxext-dev libfreetype6-dev \
      libcurl4-openssl-dev libwebkit2gtk-4.0-dev
  ```
- **Windows:** Visual Studio 2022 with the "Desktop development with C++" workload.

## Build

```bash
./scripts/build_plugin.sh
# or manually:
cd plugin
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

### Output

| Format     | Path (relative to `plugin/build`)                                 |
| ---------- | ----------------------------------------------------------------- |
| VST3       | `NeuralChordHarmonizer_artefacts/Release/VST3/*.vst3`            |
| AU         | `NeuralChordHarmonizer_artefacts/Release/AU/*.component` (macOS) |
| Standalone | `NeuralChordHarmonizer_artefacts/Release/Standalone/*`           |

## Installing the model

The plugin loads the detector from a folder containing:

```
models/pretrained/chordnet.rtneural
models/pretrained/chord_info.json
```

Load it at runtime via the editor's **Load Models...** button, or point the
`AVT_MODELS_DIR` environment variable at the folder to auto-load on startup.

## Routing the sidechain

The plugin declares a **sidechain input bus** in addition to the main vocal I/O.
In your DAW:

1. Insert the plugin on the **vocal** track.
2. Route a melodic instrument (guitar/bass/piano) to the plugin's sidechain
   input (the exact UI for this is host-specific — "Side-chain" in Logic/Ableton,
   an extra input bus in Reaper, etc.).
3. The instrument drives chord detection; the vocal is the signal that gets
   harmonized.

## Validating the plugin

```bash
# AU validation (macOS)
auval -v aufx Avtf Avtf

# pluginval (cross-platform)
pluginval --strictness-level 8 path/to/NeuralChordHarmonizer.vst3
```

## Unit tests

The JUCE-free DSP primitives have a standalone test target (no JUCE required):

```bash
cd plugin/Tests
clang++ -std=c++20 -I../Source -O2 \
    test_main.cpp test_dsp.cpp test_nn.cpp ../Source/DSP/OverlapAddBuffer.cpp \
    -o /tmp/avt_tests && /tmp/avt_tests
```

or via CMake/CTest if the test target is enabled.

See [`REAL_TIME_OPTIMIZATION.md`](REAL_TIME_OPTIMIZATION.md) for performance tuning.
