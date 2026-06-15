#!/usr/bin/env bash
# Build the avtdsp Python module from the plugin's C++ DSP (JUCE-free path).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

"$PY" -m pip install -q pybind11 numpy

# Capture include dirs individually so paths containing spaces survive quoting.
EXT="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
PYBIND_INC="$("$PY" -c 'import pybind11; print(pybind11.get_include())')"
PYTHON_INC="$("$PY" -c 'import sysconfig; print(sysconfig.get_path("include"))')"

# macOS resolves Python symbols at import time -> dynamic_lookup.
LINKFLAGS="-undefined dynamic_lookup"
if [[ "$(uname)" != "Darwin" ]]; then LINKFLAGS=""; fi

c++ -O3 -Wall -shared -std=c++17 -fPIC \
    $LINKFLAGS \
    -I"$PYBIND_INC" -I"$PYTHON_INC" \
    -I"$HERE/../plugin/Source" \
    "$HERE/avtdsp.cpp" \
    "$HERE/../plugin/Source/DSP/SpectrogramProcessor.cpp" \
    -o "$HERE/avtdsp${EXT}"

echo "built $HERE/avtdsp${EXT}"
