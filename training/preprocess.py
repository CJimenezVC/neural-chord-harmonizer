"""Audio preprocessing: resample, extract features, pack to HDF5, build splits.

[LEGACY/DEPRECATED - old voice-conversion engine; not used by the chord harmonizer.]


Run via ``scripts/preprocess_data.sh`` or directly:

    python preprocess.py --config config.yaml
    python preprocess.py --make-splits --seed 1234
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import h5py
import librosa
import numpy as np
import yaml
from tqdm import tqdm

# Use the plugin's exact C++ feature extraction (built via bindings/build.sh)
# so training and inference features are identical by construction.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bindings"))
import avtdsp


def mel_filterbank(a: dict) -> np.ndarray:
    """Librosa Slaney filterbank (also baked into model_info.json for the plugin)."""
    return librosa.filters.mel(
        sr=a["sample_rate"], n_fft=a["n_fft"], n_mels=a["n_mels"],
        fmin=a["fmin"], fmax=a["fmax"],
    ).astype(np.float32)


def load_audio(path: Path, sr: int) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def compute_mel(y: np.ndarray, a: dict, fb: np.ndarray) -> np.ndarray:
    # Streaming log-mel via the plugin's SpectrogramProcessor (C++).
    return avtdsp.extract_logmel(y, a["n_fft"], a["hop_length"], a["n_mels"],
                                 fb, float(a["sample_rate"]))   # (T, n_mels)


def iter_audio_files(root: Path):
    for ext in ("*.wav", "*.flac"):
        for p in root.rglob(ext):
            # Skip macOS archive cruft (__MACOSX/ and ._ resource forks).
            if "__MACOSX" in p.parts or p.name.startswith("._"):
                continue
            yield p


def preprocess(cfg: dict, max_files: int | None = None) -> None:
    a = cfg["audio"]
    out_dir = Path(cfg["data"]["features_train"]).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets_root = Path("../data/datasets")
    targets = {
        # VCC2020 ZIPs extract to vcc2020/{source,target_task1,target_task2}/<SPK>/*.wav
        "train": (datasets_root / "vcc2020", cfg["data"]["features_train"]),
        "eval": (datasets_root / "cmu_arctic", cfg["data"]["features_eval"]),
    }

    fb = mel_filterbank(a)
    mel_acc: list[np.ndarray] = []
    for name, (src, out_path) in targets.items():
        files = sorted(iter_audio_files(src))
        if max_files is not None:
            # Stride-sample so a capped run still spans multiple speakers.
            files = files[:: max(1, len(files) // max_files)][:max_files] if files else files
        if not files:
            print(f"[warn] no audio under {src} — run download_datasets.sh first")
            continue
        print(f"[{name}] {len(files)} files -> {out_path}", flush=True)
        with h5py.File(out_path, "w") as h5:
            for f in tqdm(files, desc=f"features:{name}"):
                y = load_audio(f, a["sample_rate"])
                mel = compute_mel(y, a, fb)
                # F0 is not used by the training loss; store zeros (skips slow pyin).
                f0 = np.zeros(len(mel), np.float32)
                # Key by speaker + file: parallel utterances share a filename
                # across speakers (e.g. SEF1/E10001 vs TEF1/E10001).
                key = f"{f.parent.name}_{f.stem}"
                grp = h5.create_group(key)
                grp.create_dataset("mel", data=mel, compression="gzip")
                grp.create_dataset("f0", data=f0, compression="gzip")
                if name == "train":
                    mel_acc.append(mel)

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
    ap.add_argument("--max-files", type=int, default=None,
                    help="cap files per split (quick subset; spans speakers)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.make_splits:
        make_splits(cfg, args.seed)
    else:
        preprocess(cfg, max_files=args.max_files)
        make_splits(cfg, cfg["training"]["seed"])
        make_splits(cfg, cfg["training"]["seed"])
