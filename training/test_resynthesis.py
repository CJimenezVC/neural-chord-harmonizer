"""Offline check of the plugin's mel-inversion resynthesis path.

Mirrors the C++ audio chain exactly (per 24 kHz frame):
  STFT(Hann) -> mag, phase
  mel = log(melfb @ mag + 1e-6)
  normalize -> encode -> decode -> denormalize
  melLin = exp(mel');  mag2[k] = Σ_m melfb[m,k]·melLin[m] / colNorm[k]
  spectrum = mag2 · e^{j·phase_in};  ISTFT -> frame
  frame *= Hann (synthesis);  overlap-add;  *= 1/1.5

Writes out.wav and reports levels. Use --identity to skip the neural step
(pure DSP round-trip sanity / gain check).

    python test_resynthesis.py --wav <file> [--identity]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml

from model import ConvDecoder, ConvEncoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--wav", default=None)
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--identity", action="store_true", help="skip encoder/decoder")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    a, m = cfg["audio"], cfg["model"]
    sr, n_fft, hop = a["sample_rate"], a["n_fft"], a["hop_length"]

    wav = args.wav or str(next(Path("../data/datasets/vcc2020/source/SEF1").glob("*.wav")))
    y, _ = librosa.load(wav, sr=sr, mono=True)
    print(f"input: {wav}  ({len(y)/sr:.2f}s)")

    melfb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=a["n_mels"],
                                fmin=a["fmin"], fmax=a["fmax"])          # [n_mels, n_bins]
    colnorm = (melfb ** 2).sum(0) + 1e-8                                  # [n_bins]
    win = np.hanning(n_fft).astype(np.float32)
    # librosa.feature.melspectrogram uses a POWER spectrum (|STFT|^2).

    stats = json.loads(Path(cfg["data"]["stats"]).read_text())
    mean = np.asarray(stats["mel_mean"], np.float32)
    std = np.asarray(stats["mel_std"], np.float32)

    enc = dec = None
    if not args.identity:
        enc = ConvEncoder(m["mel_bins"], m["encoder_channels"], m["style_dim"]).eval()
        dec = ConvDecoder(m["mel_bins"], m["style_dim"], m["decoder_channels"]).eval()
        enc.load_state_dict(torch.load("../models/pytorch/encoder.pt", map_location="cpu"))
        dec.load_state_dict(torch.load("../models/pytorch/decoder.pt", map_location="cpu"))

    out = np.zeros(len(y) + n_fft, np.float32)
    n_frames = 1 + (len(y) - n_fft) // hop
    with torch.no_grad():
        for f in range(n_frames):
            s = f * hop
            frame = y[s:s + n_fft] * win
            spec = np.fft.rfft(frame)
            mag, phase = np.abs(spec), np.angle(spec)
            mel = np.log(melfb @ (mag ** 2) + 1e-6).astype(np.float32)   # power

            if enc is not None:
                mn = ((mel - mean) / (std + 1e-8)).astype(np.float32)
                t = torch.from_numpy(mn).view(1, 1, -1)
                style = enc(t)
                mel_hat = dec(t, style).numpy().ravel()
                mel = mel_hat * (std + 1e-8) + mean        # denormalize

            mel_lin = np.exp(mel)                          # power mel energies
            power = np.maximum(0.0, (melfb.T @ mel_lin) / colnorm)   # inverse-mel
            mag2 = np.sqrt(power)                           # power -> magnitude
            spec2 = mag2 * np.exp(1j * phase)
            rec = np.fft.irfft(spec2, n=n_fft).astype(np.float32)
            out[s:s + n_fft] += rec * win                  # synthesis window + OLA

    out *= 3.0  # 1/1.5 COLA x ~4.5 makeup (matches plugin)
    sf.write(args.out, out[:len(y)], sr)
    rms_in, rms_out = np.sqrt((y ** 2).mean()), np.sqrt((out[:len(y)] ** 2).mean())
    print(f"identity={args.identity}")
    print(f"  RMS  in={rms_in:.4f}  out={rms_out:.4f}  ratio={rms_out/ (rms_in+1e-9):.3f}")
    print(f"  peak in={np.abs(y).max():.4f}  out={np.abs(out).max():.4f}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
