"""PyTorch model definition for Neural Chord Harmonizer.

ChordNet: a per-frame polyphonic pitch-class detector. It maps the
log-frequency feature (61 semitone bins) to 12 pitch-class logits. A small
dense, multi-label classifier (61 -> 256 -> 256 -> 12); sigmoid is applied at
the loss / inference. Dense-only so it exports cleanly to the plugin's NNModel.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChordNet(nn.Module):
    """Per-frame polyphonic pitch-class detector: log-freq feature -> 12 classes.

    The hard, learned part is harmonic suppression: mapping a harmonic-rich
    instrument spectrum back to the fundamental pitch classes (a chromagram
    smears energy onto harmonics; this learns to undo that).
    """

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 12),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)          # logits (sigmoid applied at loss/inference)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
