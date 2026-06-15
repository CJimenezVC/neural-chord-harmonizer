# Models

| Subdir          | Contents                                                       |
| --------------- | -------------------------------------------------------------- |
| `pretrained/`   | RTNeural exports (`*.rtneural`) + `model_info.json` (plugin loads these) |
| `pytorch/`      | Training checkpoints (`encoder.pt`, `decoder.pt`, `vocoder.pt`) |
| `quantization/` | Quantization accuracy reports                                  |

Model binaries are **gitignored** (see root `.gitignore`). Produce them with:

```bash
./scripts/train.sh          # -> models/pytorch/*.pt
./scripts/export_models.sh  # -> models/pretrained/*.rtneural + model_info.json
```

`model_info.json` records the dims and normalization statistics the plugin
needs to match training (style_dim, mel_bins, sample_rate, hop_length).
