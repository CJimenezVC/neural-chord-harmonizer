"""PyTorch model definitions for Adaptive Voice Transform.

Three modules trained jointly:
  - ConvEncoder:    mel-spectrogram -> compact style vector
  - ConvDecoder:    (mel, style)    -> transformed mel-spectrogram
  - WaveRNNVocoder: mel             -> waveform (mu-law, autoregressive)

The architectures are intentionally lightweight (~500K params total) so they
export cleanly to RTNeural and run in real time inside the plugin.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """Mel-spectrogram -> style vector (~150K params)."""

    def __init__(self, mel_bins: int = 128, channels: int = 256, style_dim: int = 64):
        super().__init__()
        # kernel_size=1 (pointwise): the plugin runs these per-frame, so a 1-tap
        # conv makes single-frame inference exactly match training.
        self.conv1 = nn.Conv1d(mel_bins, channels, kernel_size=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.fc1 = nn.Linear(channels, 128)
        self.fc2 = nn.Linear(128, style_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, T, mel_bins) -> (B, mel_bins, T) for Conv1d
        x = mel.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=-1)            # global mean pooling -> (B, channels)
        x = F.relu(self.fc1(x))
        style = self.fc2(x)          # (B, style_dim)
        return style


class ConvDecoder(nn.Module):
    """(mel, style) -> transformed mel (~200K params)."""

    def __init__(self, mel_bins: int = 128, style_dim: int = 64, channels: int = 256):
        super().__init__()
        in_ch = mel_bins + style_dim
        # kernel_size=1 (pointwise) for exact per-frame inference in the plugin.
        self.conv1 = nn.Conv1d(in_ch, channels, kernel_size=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=1)
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, mel_bins)

    def forward(self, mel: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # mel: (B, T, mel_bins), style: (B, style_dim)
        b, t, _ = mel.shape
        style_seq = style.unsqueeze(1).expand(b, t, -1)        # broadcast frame-wise
        x = torch.cat([mel, style_seq], dim=-1)                # (B, T, mel+style)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.transpose(1, 2)
        x = F.relu(self.fc1(x))
        mel_out = self.fc2(x)                                  # (B, T, mel_bins)
        return mel_out


class WaveRNNVocoder(nn.Module):
    """Mel-spectrogram -> waveform via a conditioned GRU (~150K params).

    Single-sample autoregressive generation at inference. During training we
    teacher-force the GRU over the frame for efficiency.
    """

    def __init__(self, mel_bins: int = 128, gru_hidden: int = 64, quant_bins: int = 256):
        super().__init__()
        self.cond = nn.Sequential(
            nn.Conv1d(mel_bins, mel_bins, 1), nn.ReLU(),
            nn.Conv1d(mel_bins, mel_bins, 1), nn.ReLU(),
            nn.Conv1d(mel_bins, mel_bins, 1), nn.ReLU(),
            nn.Conv1d(mel_bins, mel_bins, 1), nn.ReLU(),
        )
        self.gru = nn.GRU(mel_bins, gru_hidden, batch_first=True)
        self.out = nn.Linear(gru_hidden, quant_bins)

    def forward(self, mel: torch.Tensor, h0: torch.Tensor | None = None):
        # mel: (B, T, mel_bins)
        c = self.cond(mel.transpose(1, 2)).transpose(1, 2)     # conditioning stack
        y, hn = self.gru(c, h0)
        logits = self.out(y)                                   # (B, T, quant_bins)
        return logits, hn


class VoiceTransformModel(nn.Module):
    """Full pipeline: encoder -> decoder -> vocoder."""

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.encoder = ConvEncoder(m["mel_bins"], m["encoder_channels"], m["style_dim"])
        self.decoder = ConvDecoder(m["mel_bins"], m["style_dim"], m["decoder_channels"])
        self.vocoder = WaveRNNVocoder(m["mel_bins"], m["vocoder_gru_hidden"], m["vocoder_quant_bins"])

    def forward(self, mel: torch.Tensor):
        style = self.encoder(mel)
        mel_hat = self.decoder(mel, style)
        logits, _ = self.vocoder(mel_hat)
        return {"style": style, "mel_hat": mel_hat, "audio_logits": logits}


class VoiceConversionModel(nn.Module):
    """Target-speaker voice conversion: (source mel, target id) -> target mel.

    A learned embedding per target speaker replaces the autoencoder's style
    vector, so the decoder learns a non-uniform, target-specific source->target
    spectral mapping (this is what distinguishes it from a fixed formant warp).
    The decoder is the same kernel=1 ConvDecoder used elsewhere, so the plugin's
    inference path is unchanged apart from where the style vector comes from.
    """

    def __init__(self, cfg: dict, n_targets: int):
        super().__init__()
        m = cfg["model"]
        self.spk_emb = nn.Embedding(n_targets, m["style_dim"])
        self.decoder = ConvDecoder(m["mel_bins"], m["style_dim"], m["decoder_channels"])

    def forward(self, source_mel: torch.Tensor, target_id: torch.Tensor) -> torch.Tensor:
        emb = self.spk_emb(target_id)              # (B, style_dim)
        return self.decoder(source_mel, emb)       # (B, T, mel_bins)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load(open("config.yaml"))
    model = VoiceTransformModel(cfg)
    for name, sub in [("encoder", model.encoder), ("decoder", model.decoder), ("vocoder", model.vocoder)]:
        print(f"{name:8s}: {count_parameters(sub):,} params")
    print(f"{'total':8s}: {count_parameters(model):,} params")
