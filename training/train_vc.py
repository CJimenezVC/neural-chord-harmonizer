"""Voice conversion training: (source mel, target speaker) -> target mel.

Trains VoiceConversionModel on the DTW-aligned VCC2020 parallel pairs built by
preprocess_vc.py. Saves the decoder + the learned target-speaker embeddings.

    python train_vc.py --device mps --epochs 200
"""
from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import VoiceConversionModel, count_parameters
from train import pick_device


class VCPairDataset(Dataset):
    """Random aligned windows of (source mel, target mel, target id)."""

    def __init__(self, h5_path: str, stats_path: str, window: int):
        self.h5_path = h5_path
        self.window = window
        self._h5: h5py.File | None = None
        with h5py.File(h5_path, "r") as f:
            self.keys = list(f.keys())
            self.tids = {k: int(f[k].attrs["target_id"]) for k in self.keys}
        stats = json.loads(Path(stats_path).read_text())
        self.mean = np.asarray(stats["mel_mean"], np.float32)
        self.std = np.asarray(stats["mel_std"], np.float32)

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.keys)

    def _norm(self, mel: np.ndarray) -> np.ndarray:
        return (mel - self.mean) / (self.std + 1e-8)

    def __getitem__(self, idx: int):
        g = self._file()[self.keys[idx]]
        src = np.asarray(g["source"], np.float32)
        tgt = np.asarray(g["target"], np.float32)
        t, w = src.shape[0], self.window
        if t >= w:
            s = random.randint(0, t - w)
            src, tgt = src[s:s + w], tgt[s:s + w]
        else:
            pad = w - t
            src = np.pad(src, ((0, pad), (0, 0)))
            tgt = np.pad(tgt, ((0, pad), (0, 0)))
        return {
            "source": torch.from_numpy(self._norm(src)),
            "target": torch.from_numpy(self._norm(tgt)),
            "tid": torch.tensor(self.tids[self.keys[idx]], dtype=torch.long),
        }


def train(cfg: dict, device: str, epochs: int) -> None:
    device = pick_device(device)
    torch.manual_seed(cfg["training"]["seed"])

    vc_path = str(Path(cfg["data"]["features_train"]).parent / "vc_pairs.h5")
    with h5py.File(vc_path, "r") as f:
        targets = json.loads(f.attrs["targets"])
    n_targets = len(targets)

    model = VoiceConversionModel(cfg, n_targets).to(device)
    print(f"device: {device}  targets: {targets}  params: {count_parameters(model):,}", flush=True)

    ds = VCPairDataset(vc_path, cfg["data"]["stats"], cfg["training"]["window_frames"])
    n_val = max(1, len(ds) // 10)
    val_ds, train_ds = torch.utils.data.random_split(
        ds, [n_val, len(ds) - n_val], generator=torch.Generator().manual_seed(0))
    bs = cfg["training"]["batch_size"]
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=bs)
    print(f"pairs: {len(ds)}  train batches: {len(train_dl)}  val batches: {len(val_dl)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    autocast = (lambda: torch.autocast("cuda")) if amp else nullcontext
    writer = SummaryWriter(Path(cfg["logging"]["log_dir"]) / "vc")

    pt_out = Path(cfg["paths"]["pytorch_out"]); pt_out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    step = 0
    for epoch in range(epochs):
        model.train()
        for b in tqdm(train_dl, desc=f"epoch {epoch}", leave=False):
            src, tgt, tid = b["source"].to(device), b["target"].to(device), b["tid"].to(device)
            opt.zero_grad(set_to_none=True)
            with autocast():
                pred = model(src, tid)
                loss = F.mse_loss(pred, tgt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if step % cfg["logging"]["log_every"] == 0:
                writer.add_scalar("vc/train", loss.item(), step)
            step += 1

        val = validate(model, val_dl, device)
        writer.add_scalar("vc/val", val, epoch)
        print(f"epoch {epoch}: val={val:.4f}", flush=True)
        if val < best:
            best = val
            torch.save(model.decoder.state_dict(), pt_out / "decoder.pt")
            torch.save(model.spk_emb.weight.detach().cpu(), pt_out / "spk_emb.pt")
            (pt_out / "vc_targets.json").write_text(json.dumps(targets))
            print(f"  ✓ new best ({val:.4f}) -> {pt_out}", flush=True)

    writer.close()


@torch.no_grad()
def validate(model, dl, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for b in dl:
        src, tgt, tid = b["source"].to(device), b["target"].to(device), b["tid"].to(device)
        total += F.mse_loss(model(src, tid), tgt).item()
        n += 1
    return total / max(n, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    train(yaml.safe_load(open(args.config)), args.device, args.epochs)
