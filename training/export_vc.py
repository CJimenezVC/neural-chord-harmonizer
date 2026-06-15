"""Export the voice-conversion model (decoder + target embeddings) for the plugin.

Writes decoder.rtneural and a model_info.json with mode="conversion", the target
names, the learned speaker embeddings, and the usual stats/filterbank/pinv.

    python export_vc.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch
import yaml

from export_rtneural import export_decoder
from model import ConvDecoder


def main(cfg: dict) -> None:
    m, a = cfg["model"], cfg["audio"]
    pt = Path(cfg["paths"]["pytorch_out"])
    out = Path(cfg["paths"]["rtneural_out"])
    out.mkdir(parents=True, exist_ok=True)

    targets = json.loads((pt / "vc_targets.json").read_text())
    dec = ConvDecoder(m["mel_bins"], m["style_dim"], m["decoder_channels"])
    dec.load_state_dict(torch.load(pt / "decoder.pt", map_location="cpu"))
    dec.eval()
    emb = torch.load(pt / "spk_emb.pt", map_location="cpu").numpy()   # [n_targets, style_dim]

    (out / "decoder.rtneural").write_text(json.dumps(export_decoder(dec)))
    print(f"[ok] wrote {out / 'decoder.rtneural'}")

    info = {
        "mode": "conversion",
        "style_dim": m["style_dim"],
        "mel_bins": m["mel_bins"],
        "sample_rate": a["sample_rate"],
        "hop_length": a["hop_length"],
        "n_fft": a["n_fft"],
        "fmin": a["fmin"],
        "fmax": a["fmax"],
        "targets": targets,
        "speaker_embeddings": emb.astype(np.float32).tolist(),   # [n_targets][style_dim]
    }
    stats = json.loads(Path(cfg["data"]["stats"]).read_text())
    info["mel_mean"], info["mel_std"] = stats["mel_mean"], stats["mel_std"]
    fb = librosa.filters.mel(sr=a["sample_rate"], n_fft=a["n_fft"], n_mels=a["n_mels"],
                             fmin=a["fmin"], fmax=a["fmax"])
    info["mel_fb"] = fb.astype(np.float32).tolist()
    info["inv_mel_fb"] = np.linalg.pinv(fb).astype(np.float32).tolist()

    (out / "model_info.json").write_text(json.dumps(info))
    print(f"[ok] wrote {out / 'model_info.json'}  (conversion; targets={targets})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(yaml.safe_load(open(args.config)))
