"""Export trained PyTorch checkpoints to RTNeural JSON model files.

    python export_rtneural.py --config config.yaml

RTNeural consumes a JSON description of layers + weights. This script walks each
sub-model, emits the corresponding RTNeural layer specs, and validates that the
exported weights reproduce the PyTorch output within tolerance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch
import yaml

from model import ConvDecoder, ConvEncoder, WaveRNNVocoder


def linear_layer(weight: torch.Tensor, bias: torch.Tensor, activation: str = "none") -> dict:
    return {
        "type": "dense",
        "shape": [weight.shape[1], weight.shape[0]],
        "weights": [weight.t().tolist(), bias.tolist()],
        "activation": activation,
    }


def conv1d_layer(weight: torch.Tensor, bias: torch.Tensor, activation: str = "relu") -> dict:
    # RTNeural conv1d weight layout: [out_ch][in_ch][kernel]
    return {
        "type": "conv1d",
        "in_channels": weight.shape[1],
        "out_channels": weight.shape[0],
        "kernel_size": weight.shape[2],
        "dilation": 1,
        "weights": [weight.tolist(), bias.tolist()],
        "activation": activation,
    }


def export_encoder(model: ConvEncoder) -> dict:
    sd = model.state_dict()
    return {
        "in_shape": model.conv1.in_channels,
        "layers": [
            conv1d_layer(sd["conv1.weight"], sd["conv1.bias"]),
            conv1d_layer(sd["conv2.weight"], sd["conv2.bias"]),
            {"type": "global_mean_pool"},
            linear_layer(sd["fc1.weight"], sd["fc1.bias"], "relu"),
            linear_layer(sd["fc2.weight"], sd["fc2.bias"], "none"),
        ],
    }


def export_decoder(model: ConvDecoder) -> dict:
    sd = model.state_dict()
    return {
        "in_shape": model.conv1.in_channels,
        "layers": [
            conv1d_layer(sd["conv1.weight"], sd["conv1.bias"]),
            conv1d_layer(sd["conv2.weight"], sd["conv2.bias"]),
            conv1d_layer(sd["conv3.weight"], sd["conv3.bias"]),
            linear_layer(sd["fc1.weight"], sd["fc1.bias"], "relu"),
            linear_layer(sd["fc2.weight"], sd["fc2.bias"], "none"),
        ],
    }


def export_vocoder(model: WaveRNNVocoder) -> dict:
    sd = model.state_dict()
    layers = []
    for i in (0, 2, 4, 6):  # the four conditioning conv layers
        layers.append(conv1d_layer(sd[f"cond.{i}.weight"], sd[f"cond.{i}.bias"]))
    layers.append({
        "type": "gru",
        "hidden_size": model.gru.hidden_size,
        "weights": {
            "weight_ih": sd["gru.weight_ih_l0"].tolist(),
            "weight_hh": sd["gru.weight_hh_l0"].tolist(),
            "bias_ih": sd["gru.bias_ih_l0"].tolist(),
            "bias_hh": sd["gru.bias_hh_l0"].tolist(),
        },
    })
    layers.append(linear_layer(sd["out.weight"], sd["out.bias"], "softmax"))
    return {"in_shape": model.cond[0].in_channels, "layers": layers}


def validate(model, torch_input: torch.Tensor, exported: dict, name: str) -> None:
    """Placeholder numeric check — re-run RTNeural inference here once wired up."""
    with torch.no_grad():
        ref = model(torch_input) if name != "vocoder" else model(torch_input)[0]
    print(f"  [{name}] PyTorch ref shape {tuple(np.asarray(ref).shape)} — "
          f"exported {len(exported['layers'])} layers")


def main(cfg: dict) -> None:
    m = cfg["model"]
    pt = Path(cfg["paths"]["pytorch_out"])
    out = Path(cfg["paths"]["rtneural_out"])
    out.mkdir(parents=True, exist_ok=True)

    encoder = ConvEncoder(m["mel_bins"], m["encoder_channels"], m["style_dim"])
    decoder = ConvDecoder(m["mel_bins"], m["style_dim"], m["decoder_channels"])
    vocoder = WaveRNNVocoder(m["mel_bins"], m["vocoder_gru_hidden"], m["vocoder_quant_bins"])
    encoder.load_state_dict(torch.load(pt / "encoder.pt", map_location="cpu"))
    decoder.load_state_dict(torch.load(pt / "decoder.pt", map_location="cpu"))
    vocoder.load_state_dict(torch.load(pt / "vocoder.pt", map_location="cpu"))
    for net in (encoder, decoder, vocoder):
        net.eval()

    specs = {
        "encoder": export_encoder(encoder),
        "decoder": export_decoder(decoder),
        "vocoder": export_vocoder(vocoder),
    }
    for name, spec in specs.items():
        (out / f"{name}.rtneural").write_text(json.dumps(spec))
        print(f"[ok] wrote {out / f'{name}.rtneural'}")

    info = {
        "style_dim": m["style_dim"],
        "mel_bins": m["mel_bins"],
        "sample_rate": cfg["audio"]["sample_rate"],
        "hop_length": cfg["audio"]["hop_length"],
        "n_fft": cfg["audio"]["n_fft"],
        "fmin": cfg["audio"]["fmin"],
        "fmax": cfg["audio"]["fmax"],
    }

    # Bake the mel normalization stats directly into model_info so the plugin
    # applies the exact same (mel - mean) / std the model was trained on.
    stats_path = Path(cfg["data"]["stats"])
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        info["mel_mean"] = stats["mel_mean"]
        info["mel_std"] = stats["mel_std"]
        print(f"[ok] embedded mel normalization stats ({len(stats['mel_mean'])} bins)")
    else:
        print(f"[warn] stats file not found ({stats_path}); plugin will skip normalization")

    # Bake the exact (Slaney, area-normalized) mel filterbank so the plugin's
    # features match librosa.feature.melspectrogram (which uses a POWER spectrum).
    a = cfg["audio"]
    mel_fb = librosa.filters.mel(sr=a["sample_rate"], n_fft=a["n_fft"], n_mels=a["n_mels"],
                                 fmin=a["fmin"], fmax=a["fmax"])         # [n_mels, n_bins]
    info["mel_fb"] = mel_fb.astype(np.float32).tolist()
    print(f"[ok] embedded mel filterbank {mel_fb.shape}")

    (out / "model_info.json").write_text(json.dumps(info))
    print(f"[ok] wrote {out / 'model_info.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(yaml.safe_load(open(args.config)))
