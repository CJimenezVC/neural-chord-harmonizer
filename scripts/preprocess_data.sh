#!/usr/bin/env bash
# Resample audio, extract features, write HDF5, build splits.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/training"

python preprocess.py --config config.yaml
echo "==> Features + splits written under data/datasets/preprocessed/ and data/splits/"
