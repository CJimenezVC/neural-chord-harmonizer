# Quantization Report

Tracks accuracy impact of post-training quantization for RTNeural deployment.
Populated by `training/export_rtneural.py` once a trained model is available.

## Method

- **Baseline:** FP32 inference (matches PyTorch).
- **Candidate:** INT8 weights/activations with per-channel scales.
- **Acceptance:** measured MCD increase < 1 dB vs FP32.

## Results

| Model    | FP32 MCD (dB) | INT8 MCD (dB) | Δ (dB) | Size FP32 | Size INT8 |
| -------- | ------------- | ------------- | ------ | --------- | --------- |
| encoder  | _TBD_         | _TBD_         | _TBD_  | _TBD_     | _TBD_     |
| decoder  | _TBD_         | _TBD_         | _TBD_  | _TBD_     | _TBD_     |
| vocoder  | _TBD_         | _TBD_         | _TBD_  | _TBD_     | _TBD_     |

See `accuracy_loss_analysis.csv` for per-layer detail.

> Current default ships **FP32** (no quantization loss). INT8 is an optional
> optimization path documented in `docs/REAL_TIME_OPTIMIZATION.md`.
