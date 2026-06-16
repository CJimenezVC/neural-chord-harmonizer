"""Dataset and DataLoader for preprocessed VCC2020 / CMU Arctic features.

[LEGACY/DEPRECATED - old voice-conversion engine; not used by the chord harmonizer.]


Reads packed mel-spectrogram + F0 features from the HDF5 files produced by
``preprocess.py`` and yields fixed-length windows for training.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class VoiceFeatureDataset(Dataset):
    """Yields ``window_frames``-length mel windows from an HDF5 feature file."""

    def __init__(self, h5_path: str, split_file: str | None, window_frames: int,
                 stats_path: str | None = None, limit: int | None = None):
        self.h5_path = h5_path
        self.window_frames = window_frames
        self._h5: h5py.File | None = None

        with h5py.File(h5_path, "r") as f:
            self.keys = list(f.keys())

        if split_file and Path(split_file).exists():
            wanted = set(Path(split_file).read_text().split())
            self.keys = [k for k in self.keys if k in wanted]

        if limit is not None:
            self.keys = self.keys[:limit]

        self.mel_mean = self.mel_std = None
        if stats_path and Path(stats_path).exists():
            stats = json.loads(Path(stats_path).read_text())
            self.mel_mean = np.asarray(stats["mel_mean"], dtype=np.float32)
            self.mel_std = np.asarray(stats["mel_std"], dtype=np.float32)

    def _file(self) -> h5py.File:
        # Open lazily so the dataset is fork-safe with num_workers > 0.
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        grp = self._file()[self.keys[idx]]
        mel = np.asarray(grp["mel"], dtype=np.float32)        # (T, n_mels)
        f0 = np.asarray(grp["f0"], dtype=np.float32)          # (T,)

        # Random crop / pad to window_frames.
        t = mel.shape[0]
        w = self.window_frames
        if t >= w:
            start = random.randint(0, t - w)
            mel, f0 = mel[start:start + w], f0[start:start + w]
        else:
            pad = w - t
            mel = np.pad(mel, ((0, pad), (0, 0)))
            f0 = np.pad(f0, (0, pad))

        if self.mel_mean is not None:
            mel = (mel - self.mel_mean) / (self.mel_std + 1e-8)

        return {"mel": torch.from_numpy(mel), "f0": torch.from_numpy(f0)}


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    d, t = cfg["data"], cfg["training"]
    splits = Path(d["splits_dir"])
    limit = d.get("limit")
    # train and val are both drawn from the training HDF5 (make_splits writes
    # train/val/test manifests over its keys); the eval HDF5 is reserved for
    # the held-out test set used by evaluate.py.
    train_ds = VoiceFeatureDataset(d["features_train"], str(splits / "train.txt"),
                                   t["window_frames"], d["stats"], limit=limit)
    val_ds = VoiceFeatureDataset(d["features_train"], str(splits / "val.txt"),
                                 t["window_frames"], d["stats"], limit=limit)
    # pin_memory only helps for CUDA; MPS/CPU don't benefit and it can warn.
    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                          num_workers=t["num_workers"], drop_last=True, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False,
                        num_workers=t["num_workers"], pin_memory=pin)
    return train_dl, val_dl
