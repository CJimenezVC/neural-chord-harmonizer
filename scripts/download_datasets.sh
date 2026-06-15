#!/usr/bin/env bash
# Download VCC2020 and CMU Arctic into data/datasets/.
# Note: VCC2020 may require manual acceptance of terms; URLs are placeholders.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data/datasets"
mkdir -p "$DATA/vcc2020" "$DATA/cmu_arctic"

echo "==> CMU Arctic (bdl, clb, jmk, awb)"
ARCTIC_BASE="http://festvox.org/cmu_arctic/cmu_arctic/packed"
for spk in cmu_us_bdl_arctic cmu_us_clb_arctic cmu_us_jmk_arctic cmu_us_awb_arctic; do
    out="$DATA/cmu_arctic/${spk}.tar.bz2"
    if [[ ! -f "$out" ]]; then
        echo "  downloading $spk"
        curl -fL --retry 3 -o "$out" "$ARCTIC_BASE/${spk}-0.95-release.tar.bz2" || \
            echo "  [warn] failed: $spk (check URL / network)"
    fi
    [[ -f "$out" ]] && tar -xjf "$out" -C "$DATA/cmu_arctic" || true
done

echo "==> VCC2020 (openly downloadable, ODbL/DbCL-1.0 — no registration)"
# Hosted as ZIP assets in the nii-yamagishilab/VCC2020-database repo (master).
VCC_BASE="https://raw.githubusercontent.com/nii-yamagishilab/VCC2020-database/master"
VCC_ZIPS=(
    vcc2020_database_training_source.zip
    vcc2020_database_training_target_task1.zip
    vcc2020_database_training_target_task2.zip
    vcc2020_database_evaluation.zip
    vcc2020_database_groundtruth.zip
    vcc2020_database_transcriptions.zip
)
for zip in "${VCC_ZIPS[@]}"; do
    out="$DATA/vcc2020/$zip"
    if [[ ! -f "$out" ]]; then
        echo "  downloading $zip"
        curl -fL --retry 3 -o "$out" "$VCC_BASE/$zip" || \
            { echo "  [warn] failed: $zip (check network)"; continue; }
    fi
    unzip -oq "$out" -d "$DATA/vcc2020" && echo "  extracted $zip"
done

echo "==> Done. Next: scripts/preprocess_data.sh"
