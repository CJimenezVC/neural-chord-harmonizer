#!/usr/bin/env bash
# Export trained PyTorch checkpoints to RTNeural JSON.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/training"

python export_rtneural.py --config config.yaml
echo "==> RTNeural models written to models/pretrained/"
