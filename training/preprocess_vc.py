"""Build DTW-aligned parallel pairs for voice conversion training.

VCC2020 Task 1 is parallel (same sentences across speakers), but source and
target utterances aren't time-aligned. For each (source speaker, target speaker)
pair that shares a sentence, we DTW-align their mel sequences so the model can
learn a frame-wise source->target mapping conditioned on the target speaker.

Reads the existing train_features.h5 (mels already extracted by the plugin's
C++ DSP) and writes vc_pairs.h5 with aligned (source, target, target_id) groups.

    python preprocess_vc.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import h5py
import librosa
import numpy as np
import yaml
from tqdm import tqdm

# All 8 English speakers act as both source and target (every A->B pair over the
# sentences they share). This maximizes parallel data and source variety. The
# cross-lingual Task-2 targets are excluded: they are non-parallel (different
# languages/sentences), so they can't be DTW-aligned for parallel VC.
SPEAKERS = ["SEF1", "SEF2", "SEM1", "SEM2", "TEF1", "TEF2", "TEM1", "TEM2"]


def main(cfg: dict) -> None:
    feat_path = cfg["data"]["features_train"]
    out_path = Path(feat_path).parent / "vc_pairs.h5"

    with h5py.File(feat_path, "r") as h5in:
        keys = list(h5in.keys())
        index: dict[tuple[str, str], str] = {}
        for k in keys:
            spk, sent = k.split("_", 1)
            index[(spk, sent)] = k

        def sentences(spk: str) -> set[str]:
            return {s for (sp, s) in index if sp == spk}

        n_pairs = 0
        with h5py.File(out_path, "w") as out:
            out.attrs["targets"] = json.dumps(SPEAKERS)
            for src, tgt in product(SPEAKERS, SPEAKERS):
                if src == tgt:
                    continue
                tid = SPEAKERS.index(tgt)
                shared = sorted(sentences(src) & sentences(tgt))
                for sent in tqdm(shared, desc=f"{src}->{tgt}", leave=False):
                    sm = np.asarray(h5in[index[(src, sent)]]["mel"], np.float32)   # [Ts, M]
                    tm = np.asarray(h5in[index[(tgt, sent)]]["mel"], np.float32)   # [Tt, M]
                    # DTW on mel frames; warp path pairs source<->target frames.
                    _, wp = librosa.sequence.dtw(X=sm.T, Y=tm.T, metric="euclidean")
                    wp = wp[::-1]                                                   # forward order
                    src_aligned = sm[wp[:, 0]]
                    tgt_aligned = tm[wp[:, 1]]
                    g = out.create_group(f"{src}_{tgt}_{sent}")
                    g.create_dataset("source", data=src_aligned, compression="gzip")
                    g.create_dataset("target", data=tgt_aligned, compression="gzip")
                    g.attrs["target_id"] = tid
                    n_pairs += 1

    print(f"[ok] wrote {n_pairs} DTW-aligned pairs -> {out_path}")
    print(f"     targets: {TARGETS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(yaml.safe_load(open(args.config)))
