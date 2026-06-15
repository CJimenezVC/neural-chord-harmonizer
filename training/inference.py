"""Offline inference for sanity-checking trained models on a single file.

    python inference.py --input sample.wav --output out.wav --models ../models/pytorch

Mirrors the plugin's signal chain in Python: extract mel -> encode -> decode ->
(placeholder) vocoder reconstruction. Useful to listen before exporting to
RTNeural and loading into the plugin.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml

from model import ConvDecoder, ConvEncoder


def extract_mel(y: np.ndarray, a: dict) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=a["sample_rate"], n_fft=a["n_fft"], hop_length=a["hop_length"],
        win_length=a["win_length"], n_mels=a["n_mels"], fmin=a["fmin"], fmax=a["fmax"],
    )
    return np.log(mel + 1e-6).T.astype(np.float32)


def mel_to_audio(mel_log: np.ndarray, a: dict) -> np.ndarray:
    """Griffin-Lim fallback vocoder for offline preview (the plugin uses WaveRNN)."""
    mel = np.exp(mel_log.T)
    return librosa.feature.inverse.mel_to_audio(
        mel, sr=a["sample_rate"], n_fft=a["n_fft"], hop_length=a["hop_length"],
        win_length=a["win_length"], fmin=a["fmin"], fmax=a["fmax"],
    )


@torch.no_grad()
def run(cfg: dict, in_path: str, out_path: str, models_dir: str) -> None:
    a, m = cfg["audio"], cfg["model"]
    encoder = ConvEncoder(m["mel_bins"], m["encoder_channels"], m["style_dim"])
    decoder = ConvDecoder(m["mel_bins"], m["style_dim"], m["decoder_channels"])
    encoder.load_state_dict(torch.load(Path(models_dir) / "encoder.pt", map_location="cpu"))
    decoder.load_state_dict(torch.load(Path(models_dir) / "decoder.pt", map_location="cpu"))
    encoder.eval(); decoder.eval()

    y, _ = librosa.load(in_path, sr=a["sample_rate"], mono=True)
    mel = torch.from_numpy(extract_mel(y, a)).unsqueeze(0)
    style = encoder(mel)
    mel_hat = decoder(mel, style)[0].numpy()

    out = mel_to_audio(mel_hat, a)
    sf.write(out_path, out, a["sample_rate"])
    print(f"[ok] wrote {out_path}  ({len(out) / a['sample_rate']:.2f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="out.wav")
    ap.add_argument("--models", default="../models/pytorch")
    args = ap.parse_args()
    run(yaml.safe_load(open(args.config)), args.input, args.output, args.models)
