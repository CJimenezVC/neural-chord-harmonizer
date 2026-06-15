#!/usr/bin/env bash
# Train the full VoiceTransformModel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/training"

python train.py --config config.yaml "$@"
echo "==> Best checkpoints exported to models/pytorch/"
