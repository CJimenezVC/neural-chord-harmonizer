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

| Format | Path (relative to `plugin/build`)                                  |
| ------ | ------------------------------------------------------------------ |
| VST3   | `AdaptiveVoiceTransform_artefacts/Release/VST3/*.vst3`             |
| AU     | `AdaptiveVoiceTransform_artefacts/Release/AU/*.component` (macOS)  |
| Standalone | `AdaptiveVoiceTransform_artefacts/Release/Standalone/*`        |

## Installing the models

The plugin loads RTNeural models from a known location. Place exports at:

```
models/pretrained/{encoder,decoder,vocoder}.rtneural
models/pretrained/model_info.json
```

These can be bundled into the plugin as binary data (see `CMakeLists.txt`,
`juce_add_binary_data`) or loaded from disk at runtime.

## Validating the plugin

```bash
# AU validation (macOS)
auval -v aufx Avtf Avtf

# pluginval (cross-platform)
pluginval --strictness-level 8 path/to/AdaptiveVoiceTransform.vst3
```

## Unit tests

```bash
cmake -B build -DAVT_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

See [`REAL_TIME_OPTIMIZATION.md`](REAL_TIME_OPTIMIZATION.md) for performance tuning.
