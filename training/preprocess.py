"""Audio preprocessing: resample, extract features, pack to HDF5, build splits.

Run via ``scripts/preprocess_data.sh`` or directly:

    python preprocess.py --config config.yaml
    python preprocess.py --make-splits --seed 1234
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import librosa
import numpy as np
import yaml
from tqdm import tqdm


def load_audio(path: Path, sr: int) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def compute_mel(y: np.ndarray, a: dict) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=a["sample_rate"], n_fft=a["n_fft"], hop_length=a["hop_length"],
        win_length=a["win_length"], n_mels=a["n_mels"], fmin=a["fmin"], fmax=a["fmax"],
    )
    return np.log(mel + 1e-6).T.astype(np.float32)   # (T, n_mels)


def compute_f0(y: np.ndarray, a: dict) -> np.ndarray:
    f0, _, _ = librosa.pyin(
        y, fmin=a["f0_min"], fmax=a["f0_max"], sr=a["sample_rate"],
        hop_length=a["hop_length"],
    )
    return np.nan_to_num(f0).astype(np.float32)      # (T,)


def iter_audio_files(root: Path):
    for ext in ("*.wav", "*.flac"):
        yield from root.rglob(ext)


def preprocess(cfg: dict) -> None:
    a = cfg["audio"]
    out_dir = Path(cfg["data"]["features_train"]).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets_root = Path("../data/datasets")
    targets = {
        "train": (datasets_root / "vcc2020" / "train", cfg["data"]["features_train"]),
        "eval": (datasets_root / "cmu_arctic", cfg["data"]["features_eval"]),
    }

    mel_acc: list[np.ndarray] = []
    for name, (src, out_path) in targets.items():
        files = list(iter_audio_files(src))
        if not files:
            print(f"[warn] no audio under {src} — run download_datasets.sh first")
            continue
        with h5py.File(out_path, "w") as h5:
            for f in tqdm(files, desc=f"features:{name}"):
                y = load_audio(f, a["sample_rate"])
                mel, f0 = compute_mel(y, a), compute_f0(y, a)
                n = min(len(mel), len(f0))
                grp = h5.create_group(f.stem)
                grp.create_dataset("mel", data=mel[:n], compression="gzip")
                grp.create_dataset("f0", data=f0[:n], compression="gzip")
                if name == "train":
                    mel_acc.append(mel[:n])

    if mel_acc:
        allmel = np.concatenate(mel_acc, axis=0)
        stats = {
            "mel_mean": allmel.mean(0).tolist(),
            "mel_std": allmel.std(0).tolist(),
            "sample_rate": a["sample_rate"],
            "n_mels": a["n_mels"],
        }
        Path(cfg["data"]["stats"]).write_text(json.dumps(stats))
        print(f"[ok] wrote stats -> {cfg['data']['stats']}")


def make_splits(cfg: dict, seed: int) -> None:
    random.seed(seed)
    splits_dir = Path(cfg["data"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(cfg["data"]["features_train"], "r") as f:
        keys = list(f.keys())
    random.shuffle(keys)
    n = len(keys)
    train, val = keys[: int(0.8 * n)], keys[int(0.8 * n): int(0.9 * n)]
    test = keys[int(0.9 * n):]
    for name, ks in [("train", train), ("val", val), ("test", test)]:
        (splits_dir / f"{name}.txt").write_text("\n".join(ks))
    print(f"[ok] splits: train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--make-splits", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.make_splits:
        make_splits(cfg, args.seed)
    else:
        preprocess(cfg)
        make_splits(cfg, cfg["training"]["seed"])
