# Technical Architecture

This document describes the end-to-end design of the **Chord Harmonizer**: the
two signal paths (detector and voice), the neural detector architecture, the
formant-preserving pitch shifter, and the training objective.

## 1. Design Thesis

A musically useful vocal harmonizer needs to know *which notes to sing*. Rather
than hand-tuning a polyphonic pitch tracker, the Chord Harmonizer uses a
**hybrid DSP-neural split**:

- **A neural network listens.** A lightweight dense net (ChordNet) maps a
  log-frequency spectrum of the sidechain instrument to 12 pitch-class
  probabilities. Polyphony, harmonic overtones, and timbre variation are exactly
  what's hard to specify by hand and easy to learn.
- **Deterministic DSP sings.** A bank of formant-preserving phase-vocoder pitch
  shifters re-voices the singer onto the detected chord tones, preserving vocal
  timbre (formants) so the result still sounds like the singer.
- **User parameters** shape the result continuously: how tightly the voice
  snaps (`Tune`), how loud the instrument must be to register (`Gate`), and how
  many harmony voices to stack (`Polyphony`).

The detector is trained on synthesized chords and validated against the exact
C++ feature, so there is no train/inference drift.

## 2. Signal Paths

```
                         ┌─────────────────── Detector path (24 kHz) ───────────────────┐
Sidechain instrument ──► │ Downsample → Log-freq feature → ChordNet → peak-hold + gate │ ──► chord (12-pc mask)
(host rate)              └──────────────────────────────────────────────────────────────┘             │
                                                                                                        │
                         ┌─────────────────── Voice path (host rate) ──────────────────┐                │
Main voice ───────────►  │ YIN F0 → for each chord tone: formant-preserving pitch shift │ ◄──────────────┘
(host rate)              │            → equal-power sum (choir)                         │
                         └──────────────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
                                            Output (host rate)
```

### Dual sample-rate design

The detector runs at a fixed **24 kHz** — the rate its log-frequency feature and
ChordNet were trained at. The plugin downsamples only the **sidechain** to
24 kHz (`juce::LagrangeInterpolator` via `DSP/Resampler.h`, fed from a
`SampleFifo`). The **voice path** stays at the host rate end-to-end, so the
pitch shifter introduces no resampling artifacts on the audible signal.

At 24 kHz, `N_FFT = 2048` gives ~11.7 Hz/bin — enough to resolve fundamentals
down to ~C2, which covers bass, guitar, and piano usefully.

### Frame parameters

| Parameter             | Value                                     |
| --------------------- | ----------------------------------------- |
| Host sample rate      | any (typ. 48 kHz)                         |
| Detector sample rate  | 24 kHz (fixed)                            |
| Detector FFT / hop    | 2048 / 512 (~85 ms window, ~21 ms hop)    |
| Log-freq bins         | 61 (one per semitone, MIDI 36–96 / C2–C7) |
| Voice pitch-shift FFT | 1024, hop 256 (host rate)                 |
| Voice F0              | YIN, ~60–500 Hz                           |
| Harmony voices        | up to 6                                   |

## 3. Detector: ChordNet

### 3.1 Log-frequency feature (`DSP/LogFreqFeature.h`)

```
frame (2048 @ 24 kHz)
  → unit-RMS normalize          # level-invariant: detects at any input gain
  → Hann window → real FFT → |·|
  → semitone filterbank (61 × nbins, baked from chord_info.json)
  → log(· + 1e-6)               # 61-d feature
```

The per-frame RMS normalization makes detection independent of how loud the
instrument is recorded — a quiet pick and a clipping strum produce the same
feature. The filterbank is baked at training time and shipped in
`chord_info.json`, so the C++ feature is bit-for-bit comparable to Python
(validated to ~3.4e-5).

### 3.2 ChordNet (`ML/ChordDetector` + `ML/NNModel`)

```
feature (61)
  → Dense(61→256) + ReLU
  → Dense(256→256) + ReLU
  → Dense(256→12) → Sigmoid     # 12 independent pitch-class probabilities
```

A small dense, multi-label classifier: each of the 12 outputs is an independent
"is this pitch class sounding?" probability (sigmoid, not softmax — chords are
polyphonic). Dense-only so it exports cleanly to the self-contained `NNModel`
engine (validated vs PyTorch to ~1.8e-7). Trained F1 ≈ 0.996.

### 3.3 Chord stabilization

Raw per-frame activations flicker. Two stages stabilize them
(`PluginProcessor::runDetector`):

- **Noise gate.** The 24 kHz frame's RMS is compared to the `Gate` threshold; if
  the instrument is below it, the activations are zeroed (so the chord decays
  away instead of reading notes out of noise).
- **Peak-hold with decay.** Each pitch class holds its peak and decays at
  `0.97`/hop (~21 ms), so a strummed chord sustains smoothly instead of
  stuttering as the note rings down.

## 4. Voice: formant-preserving pitch shift

### 4.1 Pitch shifter (`DSP/PitchShifter.h`)

A streaming phase vocoder (1024 FFT, 256 hop):

1. STFT the voice; estimate **instantaneous frequency** per bin from the phase
   difference between hops.
2. **Whiten** the spectral envelope (divide out a smoothed magnitude envelope)
   so pitch can be changed independently of timbre.
3. Re-bin to the target ratio, accumulate synthesis phase, then **re-apply the
   original envelope** at the shifted pitch — this preserves formants, so the
   shifted voice still sounds like the singer rather than a chipmunk.

Latency = FFT size = 1024 samples at the host rate (~21 ms @ 48 kHz), reported
to the host via `setLatencySamples`. Validated against the Python reference at
corr ≈ 0.996.

### 4.2 Choir mapping (`PluginProcessor::collectTargets`)

For each detected pitch class (strongest `Polyphony` of them), the target MIDI
note is placed in the octave nearest the singer's current F0, ordered lead-first.
Per voice, the pitch ratio is

```
ratio = (targetHz / voiceHz) ^ Tune          # Tune: 0 = dry, 1 = full snap
```

smoothed with a one-pole glide. Active voices are summed and equal-power
normalized (`1/√N`). No chord (or unvoiced input) → silent output.

## 5. Parameters

| Parameter   | Range          | Effect                                            |
| ----------- | -------------- | ------------------------------------------------- |
| `Tune`      | 0 – 1          | Natural blend → tight snap (hard auto-tune)       |
| `Gate`      | −80 – −10 dB   | Instrument RMS threshold for the detector         |
| `Polyphony` | 1 – 6          | Max simultaneous harmony voices (e.g. 6 = guitar) |

All are exposed via `AudioProcessorValueTreeState` and smoothed on the audio
thread to avoid zipper noise.

## 6. Streaming & Latency

- **Sidechain FIFO** bridges host rate → 24 kHz without drift (`SampleFifo`).
- **Detector** drains complete 2048-sample frames every 512-sample hop.
- **Pitch shifters** are overlap-add streaming (no batch processing).
- **No allocations or locks** on the audio thread; all scratch buffers are
  pre-allocated in `prepareToPlay`.
- **Reported latency** = pitch-shifter FFT size (1024 host-rate samples).

The detector path is *not* in the audible latency budget — the voice is shifted
continuously and simply follows the most recent detected chord.

## 7. Training Objective

ChordNet is trained as a multi-label classifier on synthesized chords
(`training/chord_synth.py` renders chords with randomized timbre/voicing/level;
`train_chord.py` optimizes):

```
L = BCE( sigmoid(ChordNet(feature)), chord_pitch_class_targets )
```

**Evaluation:** per-pitch-class precision / recall / F1 on held-out synthetic
chords (current best F1 ≈ 0.996).

## 8. Export to RTNeural

1. Train ChordNet (PyTorch).
2. Export dense weights to the `.rtneural` JSON consumed by `NNModel`.
3. Write `chord_info.json` (feature dims, `n_fft`, `sample_rate`, `midi_lo/hi`,
   baked `logfreq_fb`).
4. Validate: C++ `NNModel` output matches PyTorch within tolerance (~1.8e-7).

See `training/export_chord.py` and [`docs/PLUGIN_BUILD.md`](docs/PLUGIN_BUILD.md).
