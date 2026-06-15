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

echo "==> VCC2020"
echo "  VCC2020 requires registration. Download from:"
echo "    https://github.com/nii-yamagishilab/VCC2020-database"
echo "  and extract parallel pairs into: $DATA/vcc2020/train/"

echo "==> Done. Next: scripts/preprocess_data.sh"
