"""Main training loop for the full VoiceTransformModel.

    python train.py --config config.yaml

Optimizes a reconstruction objective through the whole pipeline:

    L = || vocoder(decoder(mel, encoder(mel))) - target ||^2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader import build_dataloaders
from model import VoiceTransformModel, count_parameters


def mu_law_encode(x: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """Map mel-reconstruction targets into quantized bins for the vocoder head."""
    mu = bins - 1
    x = torch.clamp(x, -1.0, 1.0)
    enc = torch.sign(x) * torch.log1p(mu * x.abs()) / torch.log1p(torch.tensor(float(mu)))
    return ((enc + 1) / 2 * mu).long().clamp(0, mu)


def reconstruction_loss(out: dict, mel: torch.Tensor, cfg: dict) -> dict:
    mel_loss = F.mse_loss(out["mel_hat"], mel)
    # Vocoder target derived from the (normalized) mel frame energy as a proxy.
    target = mu_law_encode(mel.mean(-1, keepdim=True).tanh(), cfg["model"]["vocoder_quant_bins"])
    logits = out["audio_logits"]
    voc_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
    total = cfg["loss"]["recon_weight"] * mel_loss + cfg["loss"]["vocoder_weight"] * voc_loss
    return {"total": total, "mel": mel_loss, "voc": voc_loss}


def train(cfg: dict) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["training"]["seed"])

    model = VoiceTransformModel(cfg).to(device)
    print(f"model params: {count_parameters(model):,}")

    train_dl, val_dl = build_dataloaders(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["amp"] and device == "cuda")
    writer = SummaryWriter(cfg["logging"]["log_dir"])

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pt_out = Path(cfg["paths"]["pytorch_out"])
    pt_out.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    step = 0
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        for batch in tqdm(train_dl, desc=f"epoch {epoch}"):
            mel = batch["mel"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=scaler.is_enabled()):
                out = model(mel)
                losses = reconstruction_loss(out, mel, cfg)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            scaler.step(opt)
            scaler.update()
            if step % cfg["logging"]["log_every"] == 0:
                writer.add_scalar("train/total", losses["total"].item(), step)
                writer.add_scalar("train/mel", losses["mel"].item(), step)
                writer.add_scalar("train/voc", losses["voc"].item(), step)
            step += 1

        val = validate(model, val_dl, cfg, device)
        writer.add_scalar("val/total", val, epoch)
        print(f"epoch {epoch}: val={val:.4f}")

        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")
        if val < best_val:
            best_val = val
            save_submodules(model, pt_out)
            print(f"  ✓ new best ({val:.4f}) -> {pt_out}")

    writer.close()


@torch.no_grad()
def validate(model, dl, cfg, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in dl:
        mel = batch["mel"].to(device)
        out = model(mel)
        total += reconstruction_loss(out, mel, cfg)["total"].item()
        n += 1
    return total / max(n, 1)


def save_submodules(model: VoiceTransformModel, out_dir: Path) -> None:
    torch.save(model.encoder.state_dict(), out_dir / "encoder.pt")
    torch.save(model.decoder.state_dict(), out_dir / "decoder.pt")
    torch.save(model.vocoder.state_dict(), out_dir / "vocoder.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    train(yaml.safe_load(open(args.config)))
