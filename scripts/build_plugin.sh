#!/usr/bin/env bash
# Configure and build the JUCE plugin (Release). JUCE + RTNeural are fetched
# automatically via CMake FetchContent.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/plugin"

BUILD_TYPE="${1:-Release}"
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

cmake -B build -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build build --config "$BUILD_TYPE" -j "$JOBS"

echo "==> Artefacts under plugin/build/NeuralChordHarmonizer_artefacts/$BUILD_TYPE/"
