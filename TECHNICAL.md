# Technical Architecture

This document describes the end-to-end design of Adaptive Voice Transform: the
audio processing pipeline, the neural model architecture, the streaming /
latency strategy, and the training objective.

## 1. Design Thesis

End-to-end neural voice conversion is powerful but hard to make real-time,
controllable, and stable. Adaptive Voice Transform instead uses a **hybrid
DSP-neural chain**:

- **Deterministic DSP** handles what classical signal processing does well:
  STFT, mel projection, F0 estimation (YIN), and formant analysis.
- **Lightweight neural networks** handle what is hard to specify by hand:
  learning a compact *style* representation and mapping it back to a
  transformed mel-spectrogram.
- **User parameters** modulate the style vector continuously, so the plugin
  exposes musical controls (style blend, brightness, formant shift, pitch
  shift) rather than discrete preset switches.

The whole chain is designed to be differentiable end-to-end during training and
streaming-safe during inference.

## 2. Audio Processing Pipeline

```
Input Audio (host rate, e.g. 48 kHz)
  │
  ├─ [Downsample]       host rate → 24 kHz (Lagrange resampler)
  │
  ├─ [Feature Extraction DSP]   @ 24 kHz
  │    STFT (512, 75% overlap) → mel (128 bins) → YIN F0 → formants
  │
  ├─ [Neural Encoder]   mel → style vector (64-d)
  │
  ├─ [Style Modulation] user params interpolate / shift the style vector
  │
  ├─ [Neural Decoder]   (mel ⊕ style) → transformed mel (128 bins)
  │
  ├─ [WaveRNN Vocoder]  mel → waveform @ 24 kHz
  │
  ├─ [Post-DSP]         overlap-add, spectral smoothing, artifact suppression
  │
  └─ [Upsample]         24 kHz → host rate (Lagrange resampler)
  │
Output Audio (host rate, ~40 ms latency)
```

### Dual sample-rate design

The models are trained and run **at 24 kHz** (VCC2020's native rate) to halve
compute and memory versus 48 kHz. The plugin presents the host's rate (any rate;
typically 48 kHz) and brackets the neural chain with resamplers:

- **Downsample** host → 24 kHz before feature extraction.
- **Upsample** 24 kHz → host after the vocoder.

Both use `juce::LagrangeInterpolator` (see `plugin/Source/DSP/Resampler.h`), fed
from small FIFOs so fractional sample-rate ratios (e.g. 44.1 kHz hosts) don't
drift. Speech energy lives well below the 12 kHz Nyquist at 24 kHz, so the
perceptual cost is negligible while compute drops ~2×.

### Frame parameters

| Parameter        | Value                                  |
| ---------------- | -------------------------------------- |
| Host sample rate | any (typ. 48 kHz)                      |
| Model sample rate| 24 kHz (fixed)                         |
| STFT window      | 512 samples @ 24 kHz (~21.3 ms)        |
| Overlap          | 75 % (hop = 128 samples, ~5.3 ms)      |
| Mel bins         | 128 (20 Hz – 8 kHz)                    |
| Frame rate       | ~187 frames/s (hop 128 @ 24 kHz)       |
| F0 range         | 50 – 500 Hz (YIN)                      |
| Lookahead        | 4 frames (vocoder stability)           |

## 3. Model Architecture

### 3.1 Encoder (~150 K params)

```
mel (T, 128)
  → Conv1D(128→256, k=3) + ReLU
  → Conv1D(256→256, k=3) + ReLU
  → GlobalMeanPool → (256,)
  → Dense(256→128) + ReLU
  → Dense(128→64)            # style vector
```

The bottleneck forces the encoder to compress speaker/style identity into a
64-dim vector. Trained on random ~680 ms windows (32 frames), one style vector
per utterance.

### 3.2 Decoder (~200 K params)

```
mel (T, 128) ⊕ style (T, 64)  →  (T, 192)
  → Conv1D(192→256, k=3) + ReLU
  → Conv1D(256→256, k=3) + ReLU
  → Conv1D(256→256, k=3) + ReLU
  → Dense(256→256) + ReLU
  → Dense(256→128)           # transformed mel
```

The style vector is broadcast frame-wise and concatenated to each mel frame.

### 3.3 WaveRNN Vocoder (~150 K params)

```
mel (T, 128)
  → Conditioning: Conv1D(128→128) × 4
  → GRU(128→64) with mel conditioning
  → Dense(64→256) → Softmax     # 256-way μ-law quantization
  → waveform (T, 1)
```

Single-sample autoregressive generation (no beam search), fixed quantization,
pre-computed conditioning stack for real-time use.

## 4. Style Modulation

User-facing parameters and how they map into the chain:

| Parameter          | Range            | Effect                                        |
| ------------------ | ---------------- | --------------------------------------------- |
| `StyleShift`       | 0 – 1            | Interpolate style vector source ↔ target      |
| `TimbralBrightness`| −1 – +1          | High-frequency emphasis (EQ tilt)             |
| `FormantShift`     | −12 – +12 st     | Multiply formant frequencies                  |
| `PitchShift`       | −24 – +24 st     | Transposition (PSOLA-like time-stretch)       |

All parameters are smoothed (see `Parameters/ParameterSmoothing.h`) to avoid
zipper noise, then applied before the decoder.

## 5. Streaming & Latency

### Strategy

- **Resampler FIFOs** bridge host rate ↔ 24 kHz without drift.
- **Circular buffers** for continuous input/output.
- **Overlap-add** reconstruction rather than batch processing.
- **RNN state management** so the GRU carries across frames.
- **4-frame lookahead** absorbs vocoder transients.
- **Latency compensation** reported to the host via
  `AudioProcessor::setLatencySamples` (frame + lookahead, converted to host
  samples, plus resampler group delay).

### Latency budget (target 40–50 ms)

| Stage                | Budget |
| -------------------- | ------ |
| Input buffering      | 10 ms  |
| Down/up resampling   | 2 ms   |
| STFT analysis        | 5 ms   |
| Feature norm         | 2 ms   |
| Encoder              | 8 ms   |
| Decoder              | 9 ms   |
| Vocoder              | 9 ms   |
| Overlap-add          | 3 ms   |
| Output delay         | 2 ms   |
| **Total**            | **50 ms** |

> Running inference at 24 kHz roughly halves the encoder/decoder/vocoder cost
> versus 48 kHz; the resampler adds a small fixed overhead and group delay.

See [`docs/REAL_TIME_OPTIMIZATION.md`](docs/REAL_TIME_OPTIMIZATION.md).

## 6. Training Objective

Reconstruction loss through the full pipeline:

```
L = || vocoder(decoder(mel, encoder(mel))) − target ||²
```

This jointly encourages (1) the encoder to learn meaningful style, (2) the
decoder to reconstruct faithfully, and (3) the vocoder to produce clean audio.

**Evaluation metrics:**

| Metric            | Target   | Meaning                       |
| ----------------- | -------- | ----------------------------- |
| Mel Cepstral Dist | < 5 dB   | Spectral similarity           |
| F0 RMSE           | < 10 Hz  | Pitch contour preservation    |
| Vocoder loss      | < 0.5 MSE| Audio reconstruction quality  |

## 7. Export to RTNeural

1. Load trained PyTorch checkpoints.
2. Trace to ONNX.
3. Optimize (layer fusion, memory layout).
4. Emit RTNeural JSON model files.
5. Validate: numerically match PyTorch, measure latency, profile memory.

See `training/export_rtneural.py` and [`docs/PLUGIN_BUILD.md`](docs/PLUGIN_BUILD.md).
