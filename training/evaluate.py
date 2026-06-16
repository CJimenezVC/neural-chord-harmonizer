"""Objective evaluation: Mel Cepstral Distortion (MCD) and F0 RMSE.

[LEGACY/DEPRECATED - old voice-conversion engine; not used by the chord harmonizer.]


    python evaluate.py --checkpoint ../models/pytorch --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from data_loader import VoiceFeatureDataset
from model import ConvDecoder, ConvEncoder


def mel_cepstral_distortion(c1: np.ndarray, c2: np.ndarray) -> float:
    """MCD in dB between two log-mel sequences (frame-aligned)."""
    diff = c1 - c2
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return float((10.0 / np.log(10)) * np.sqrt(2) * dist.mean())


def f0_rmse(f1: np.ndarray, f2: np.ndarray) -> float:
    voiced = (f1 > 0) & (f2 > 0)
    if voiced.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((f1[voiced] - f2[voiced]) ** 2)))


@torch.no_grad()
def evaluate(cfg: dict, ckpt_dir: Path, split: str) -> None:
    m = cfg["model"]
    encoder = ConvEncoder(m["mel_bins"], m["encoder_channels"], m["style_dim"])
    decoder = ConvDecoder(m["mel_bins"], m["style_dim"], m["decoder_channels"])
    encoder.load_state_dict(torch.load(ckpt_dir / "encoder.pt", map_location="cpu"))
    decoder.load_state_dict(torch.load(ckpt_dir / "decoder.pt", map_location="cpu"))
    encoder.eval(); decoder.eval()

    ds = VoiceFeatureDataset(
        cfg["data"]["features_eval"],
        str(Path(cfg["data"]["splits_dir"]) / f"{split}.txt"),
        cfg["training"]["window_frames"], cfg["data"]["stats"],
    )

    mcds, f0s = [], []
    for i in range(len(ds)):
        sample = ds[i]
        mel = sample["mel"].unsqueeze(0)                  # (1, T, n_mels)
        style = encoder(mel)
        mel_hat = decoder(mel, style)
        mcds.append(mel_cepstral_distortion(mel[0].numpy(), mel_hat[0].numpy()))
        f0s.append(f0_rmse(sample["f0"].numpy(), sample["f0"].numpy()))

    print(f"split={split}  N={len(ds)}")
    print(f"  MCD     : {np.nanmean(mcds):.3f} dB   (target < 5)")
    print(f"  F0 RMSE : {np.nanmean(f0s):.3f} Hz   (target < 10)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default="../models/pytorch")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    evaluate(yaml.safe_load(open(args.config)), Path(args.checkpoint), args.split)
