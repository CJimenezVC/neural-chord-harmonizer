# Quantization Report

Tracks the accuracy impact of post-training quantization of **ChordNet** for
RTNeural deployment. Optional — ChordNet is small enough that FP32 is the
default and quantization is rarely needed.

## Method

- **Baseline:** FP32 inference (matches PyTorch to ~1.8e-7).
- **Candidate:** INT8 weights/activations with per-channel scales.
- **Acceptance:** detection **F1** drop < 0.005 vs FP32 on held-out synthetic
  chords.

## Results

| Model    | FP32 F1 | INT8 F1 | Δ F1  | Size FP32 | Size INT8 |
| -------- | ------- | ------- | ----- | --------- | --------- |
| chordnet | ~0.996  | _TBD_   | _TBD_ | _TBD_     | _TBD_     |

> Current default ships **FP32** (no quantization loss). The dense ChordNet is
> tiny, so INT8 is generally unnecessary; it remains an optional path documented
> in `docs/REAL_TIME_OPTIMIZATION.md`.
