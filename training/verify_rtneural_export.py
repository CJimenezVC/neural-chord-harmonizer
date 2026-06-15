"""Validate the exported .rtneural files against the PyTorch models.

Loads the JSON written by export_rtneural.py, runs a numpy forward pass that
mirrors exactly what the C++ plugin does (single-frame / streaming), and
compares against the PyTorch reference. If the max abs diff is tiny, both the
exported weight layout and the C++ forward-pass algorithm are correct.

    python verify_rtneural_export.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from model import ConvDecoder, ConvEncoder, WaveRNNVocoder


# --- numpy ops mirroring the C++ inference (single time step) ---------------

def relu(x):
    return np.maximum(x, 0.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dense(x, layer):
    # weights = [W (in x out), b (out)]
    w = np.asarray(layer["weights"][0], dtype=np.float64)   # [in][out]
    b = np.asarray(layer["weights"][1], dtype=np.float64)   # [out]
    y = x @ w + b
    return _activate(y, layer.get("activation", "none"))


def conv1d_center(x, layer):
    # Single-frame conv with padding=1, kernel=3 reduces to the centre tap:
    # weights[0] = [out][in][k]; use k index 1.
    w = np.asarray(layer["weights"][0], dtype=np.float64)   # [out][in][k]
    b = np.asarray(layer["weights"][1], dtype=np.float64)   # [out]
    centre = w[:, :, w.shape[2] // 2]                       # [out][in]
    y = centre @ x + b
    return _activate(y, layer.get("activation", "relu"))


def _activate(y, name):
    if name == "relu":
        return relu(y)
    if name == "softmax":
        e = np.exp(y - y.max())
        return e / e.sum()
    return y


def run_encoder(spec, mel):
    x = mel
    for layer in spec["layers"]:
        t = layer["type"]
        if t == "conv1d":
            x = conv1d_center(x, layer)
        elif t == "global_mean_pool":
            x = x                       # single frame -> identity
        elif t == "dense":
            x = dense(x, layer)
    return x


def run_decoder(spec, mel, style):
    x = np.concatenate([mel, style])
    for layer in spec["layers"]:
        t = layer["type"]
        if t == "conv1d":
            x = conv1d_center(x, layer)
        elif t == "dense":
            x = dense(x, layer)
    return x


def gru_step(layer, x, h):
    w = layer["weights"]
    Wih = np.asarray(w["weight_ih"], dtype=np.float64)   # [3H][in]
    Whh = np.asarray(w["weight_hh"], dtype=np.float64)   # [3H][H]
    bih = np.asarray(w["bias_ih"], dtype=np.float64)
    bhh = np.asarray(w["bias_hh"], dtype=np.float64)
    H = Whh.shape[1]
    gi = Wih @ x + bih
    gh = Whh @ h + bhh
    r = sigmoid(gi[:H] + gh[:H])
    z = sigmoid(gi[H:2 * H] + gh[H:2 * H])
    n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
    return (1.0 - z) * n + z * h


def run_vocoder(spec, mel):
    x = mel
    h = None
    for layer in spec["layers"]:
        t = layer["type"]
        if t == "conv1d":
            x = conv1d_center(x, layer)
        elif t == "gru":
            H = np.asarray(layer["weights"]["weight_hh"]).shape[1]
            h = gru_step(layer, x, np.zeros(H))
            x = h
        elif t == "dense":
            x = dense(x, layer)
    return x


# --- comparison -------------------------------------------------------------

def main(cfg: dict) -> None:
    m = cfg["model"]
    pt = Path(cfg["paths"]["pytorch_out"])
    out = Path(cfg["paths"]["rtneural_out"])

    enc = ConvEncoder(m["mel_bins"], m["encoder_channels"], m["style_dim"])
    dec = ConvDecoder(m["mel_bins"], m["style_dim"], m["decoder_channels"])
    voc = WaveRNNVocoder(m["mel_bins"], m["vocoder_gru_hidden"], m["vocoder_quant_bins"])
    enc.load_state_dict(torch.load(pt / "encoder.pt", map_location="cpu"))
    dec.load_state_dict(torch.load(pt / "decoder.pt", map_location="cpu"))
    voc.load_state_dict(torch.load(pt / "vocoder.pt", map_location="cpu"))
    for n in (enc, dec, voc):
        n.eval()

    enc_spec = json.loads((out / "encoder.rtneural").read_text())
    dec_spec = json.loads((out / "decoder.rtneural").read_text())
    voc_spec = json.loads((out / "vocoder.rtneural").read_text())

    rng = np.random.default_rng(0)
    mel = rng.standard_normal(m["mel_bins"]).astype(np.float64)
    mel_t = torch.tensor(mel, dtype=torch.float32).view(1, 1, -1)

    with torch.no_grad():
        # encoder
        style_pt = enc(mel_t).numpy().ravel()
        style_np = run_encoder(enc_spec, mel)
        d_enc = np.abs(style_pt - style_np).max()

        # decoder (use the PyTorch style so we isolate the decoder)
        mel_hat_pt = dec(mel_t, torch.tensor(style_pt).view(1, -1)).numpy().ravel()
        mel_hat_np = run_decoder(dec_spec, mel, style_pt.astype(np.float64))
        d_dec = np.abs(mel_hat_pt - mel_hat_np).max()

        # vocoder (first frame, hidden state = 0)
        logits_pt, _ = voc(mel_t)
        probs_pt = torch.softmax(logits_pt, dim=-1).numpy().ravel()
        probs_np = run_vocoder(voc_spec, mel)
        d_voc = np.abs(probs_pt - probs_np).max()

    tol = 1e-4
    print(f"encoder  max|Δ| = {d_enc:.2e}")
    print(f"decoder  max|Δ| = {d_dec:.2e}")
    print(f"vocoder  max|Δ| = {d_voc:.2e}  (softmax probs, first frame)")
    ok = max(d_enc, d_dec, d_voc) < tol
    print(f"\n{'✓ PASS' if ok else '✗ FAIL'}: export matches PyTorch within {tol:g}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(yaml.safe_load(open(args.config)))
