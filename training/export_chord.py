"""Export the chord/pitch detector + log-freq filterbank for the plugin.

    python export_chord.py

Writes chordnet.rtneural (dense layers, sigmoid output) and chord_info.json
(with the baked log-frequency filterbank so the plugin builds the same feature).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from export_rtneural import linear_layer
from model import ChordNet


def main() -> None:
    pt = Path("../models/pytorch")
    out = Path("../models/pretrained")
    out.mkdir(parents=True, exist_ok=True)

    info = json.loads((pt / "chord_info.json").read_text())
    net = ChordNet(info["in_dim"])
    net.load_state_dict(torch.load(pt / "chordnet.pt", map_location="cpu"))
    net.eval()
    sd = net.state_dict()

    spec = {
        "in_shape": info["in_dim"],
        "layers": [
            linear_layer(sd["net.0.weight"], sd["net.0.bias"], "relu"),
            linear_layer(sd["net.2.weight"], sd["net.2.bias"], "relu"),
            linear_layer(sd["net.4.weight"], sd["net.4.bias"], "sigmoid"),
        ],
    }
    (out / "chordnet.rtneural").write_text(json.dumps(spec))
    print(f"[ok] wrote {out / 'chordnet.rtneural'}")

    fb = np.load(pt / "logfreq_fb.npy")                       # [n_pitch, n_bins]
    info_out = {**info, "logfreq_fb": fb.astype(np.float32).tolist()}
    (out / "chord_info.json").write_text(json.dumps(info_out))
    print(f"[ok] wrote {out / 'chord_info.json'}  (filterbank {fb.shape})")


if __name__ == "__main__":
    main()
