"""Train the polyphonic pitch-class detector on synthetic chords.

    python train_chord.py --device mps --epochs 30

Saves the detector + the log-frequency filterbank (needed by the plugin to
build the same feature).
"""
from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import chord_synth as cs
from model import ChordNet, count_parameters


def pick_device(requested: str = "auto") -> str:
    """Prefer Apple MPS, fall back to CUDA, then CPU."""
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def evaluate(model, dl, device) -> dict:
    model.eval()
    tp = fp = fn = 0
    for b in dl:
        logits = model(b["feat"].to(device))
        pred = (torch.sigmoid(logits) > 0.5).float().cpu()
        lab = b["label"]
        tp += float(((pred == 1) & (lab == 1)).sum())
        fp += float(((pred == 1) & (lab == 0)).sum())
        fn += float(((pred == 0) & (lab == 1)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1}


def train(device: str, epochs: int) -> None:
    device = pick_device(device)
    torch.manual_seed(0)
    model = ChordNet(cs.N_PITCH).to(device)
    print(f"device: {device}  in_dim: {cs.N_PITCH}  params: {count_parameters(model):,}", flush=True)

    train_dl = DataLoader(cs.ChordDataset(20000), batch_size=128, shuffle=False, num_workers=4)
    val_dl = DataLoader(cs.ChordDataset(2000), batch_size=128, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    out = Path("../models/pytorch"); out.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for epoch in range(epochs):
        model.train()
        for b in train_dl:
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(b["feat"].to(device)),
                                                      b["label"].to(device))
            loss.backward()
            opt.step()
        m = evaluate(model, val_dl, device)
        print(f"epoch {epoch}: f1={m['f1']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f}",
              flush=True)
        if m["f1"] > best:
            best = m["f1"]
            torch.save(model.state_dict(), out / "chordnet.pt")
            np.save(out / "logfreq_fb.npy", cs.build_logfreq_fb())
            (out / "chord_info.json").write_text(json.dumps({
                "in_dim": cs.N_PITCH, "n_fft": cs.N_FFT, "sample_rate": cs.SR,
                "midi_lo": cs.MIDI_LO, "midi_hi": cs.MIDI_HI,
            }))
            print(f"  ✓ new best f1={best:.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    train(args.device, args.epochs)
