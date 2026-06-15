# Training Guide

End-to-end instructions for training the encoder, decoder, and vocoder and
exporting them to RTNeural.

## 1. Environment

```bash
cd training
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10+ and PyTorch 2.x (CUDA optional but recommended).

## 2. Data

```bash
../scripts/download_datasets.sh     # fetch VCC2020 + CMU Arctic
../scripts/preprocess_data.sh       # resample, extract features, build splits
```

See [`../DATASET.md`](../DATASET.md) for details. After this you should have:

```
data/datasets/preprocessed/train_features.h5
data/datasets/preprocessed/eval_features.h5
data/datasets/preprocessed/data_stats.json
data/splits/{train,val,test}.txt
```

## 3. Configure

All hyperparameters live in [`../training/config.yaml`](../training/config.yaml).
Key knobs:

```yaml
model:
  style_dim: 64
  mel_bins: 128
training:
  batch_size: 32
  lr: 3.0e-4
  epochs: 200
  window_frames: 32
```

## 4. Train

```bash
../scripts/train.sh          # wraps: python train.py --config config.yaml
```

Checkpoints are written to `training/checkpoints/` and the best model is
symlinked to `models/pytorch/{encoder,decoder,vocoder}.pt`.

Monitor:
- Reconstruction loss (should decrease steadily)
- Validation MCD / F0 RMSE (logged each epoch)

## 5. Evaluate

```bash
python evaluate.py --checkpoint ../models/pytorch --split test
```

Reports MCD (target < 5 dB), F0 RMSE (target < 10 Hz), and vocoder loss.

## 6. Export to RTNeural

```bash
../scripts/export_models.sh  # wraps: python export_rtneural.py
```

Produces `models/pretrained/{encoder,decoder,vocoder}.rtneural` and
`model_info.json`. The exporter validates that RTNeural output matches PyTorch
within tolerance before writing.

## 7. Offline Sanity Check

```bash
python inference.py --input sample.wav --output out.wav \
    --models ../models/pretrained
```

Listen to `out.wav` before loading the exported models into the plugin.

## Troubleshooting

| Symptom                         | Likely cause / fix                              |
| ------------------------------- | ----------------------------------------------- |
| Loss plateaus high              | LR too high/low; check `data_stats.json` norm   |
| Vocoder buzzy / noisy           | Increase vocoder epochs; verify mel alignment   |
| Export mismatch vs PyTorch      | Unsupported layer; check `export_rtneural.py`   |
| Plugin output differs from offline | Normalization stats mismatch (`model_info.json`) |
