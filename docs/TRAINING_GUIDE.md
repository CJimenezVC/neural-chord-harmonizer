# Training Guide

End-to-end instructions for training the **ChordNet** polyphonic pitch-class
detector and exporting it to RTNeural.

No external dataset is required: ChordNet trains on **synthesized** chords, so
training is fast and fully reproducible.

## 1. Environment

```bash
cd training
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10+ and PyTorch 2.x. Apple Silicon uses MPS with a CPU
fallback (`PYTORCH_ENABLE_MPS_FALLBACK=1`).

## 2. How the data is made

`training/chord_synth.py` synthesizes labeled chords on the fly:

- Pick a root, quality (major, minor, 7th, sus, …), and octave.
- Render the notes as a sum of harmonics with **randomized timbre** (harmonic
  rolloff) and **randomized level**, so the detector generalizes across
  instruments and gains.
- Compute the **log-frequency feature** — the exact same one the plugin uses
  (`build_logfreq_fb` / `frame_feature`), including per-frame RMS normalization.
- Label = the 12-d pitch-class vector of the chord.

Key constants (kept in sync with `chord_info.json`):

```
SR        = 24000
N_FFT     = 2048
MIDI_LO   = 36      # C2
MIDI_HI   = 96      # C7
N_PITCH   = 61      # one log-freq bin per semitone
```

## 3. Model

`training/model.py:ChordNet` — a dense multi-label classifier:

```
Linear(61 → 256) → ReLU → Linear(256 → 256) → ReLU → Linear(256 → 12)
```

Trained with BCE-with-logits against the 12-d pitch-class target (sigmoid at
inference; chords are polyphonic, so this is multi-label, not softmax).

## 4. Train

```bash
python train_chord.py        # synthesizes chords and trains ChordNet
```

The script logs per-epoch precision / recall / F1 on held-out synthetic chords
and keeps the best checkpoint (current best F1 ≈ 0.996):

```
epoch 14: f1=0.996 prec=0.997 rec=0.995
  ✓ new best f1=0.996
```

## 5. Export to RTNeural

```bash
python export_chord.py
```

Produces:

```
models/pretrained/chordnet.rtneural    # dense weights for NNModel
models/pretrained/chord_info.json      # in_dim, n_fft, sample_rate, midi_lo/hi, logfreq_fb
```

`export_chord.py` writes the dense weights in the layout the C++ `NNModel`
expects; the self-contained `NNModel` engine is itself unit-tested against
independent reference formulas (`plugin/Tests/test_nn.cpp`), and the streaming
forward pass matches PyTorch to ~1.8e-7.

## 6. Feature parity (no train/inference drift)

The plugin's C++ `LogFreqFeature` is a direct port of
`chord_synth.py:frame_feature` (same STFT, same baked filterbank, same per-frame
RMS normalization), validated to ~3.4e-5 — so what the model trains on is what
the plugin feeds it.

## 7. Sanity check in the plugin

Load `models/pretrained/` via the editor's **Load Models...** button (or
`AVT_MODELS_DIR`), play a chord into the sidechain, and watch the 12-note chord
readout light up.

## Troubleshooting

| Symptom                               | Likely cause / fix                                |
| ------------------------------------- | ------------------------------------------------- |
| Low F1 / confuses chords              | Increase synthesis variety (timbre, octaves)      |
| Plugin detects nothing                | Wrong `chord_info.json`, or sidechain not routed  |
| Phantom notes when not playing        | Raise the **Gate**; verify RMS normalization      |
| Export mismatch vs PyTorch            | Unsupported layer; ChordNet must stay dense-only  |
| Plugin feature differs from Python    | Filterbank mismatch — re-export `chord_info.json` |
