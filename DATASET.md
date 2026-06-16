# Data Guide

The Chord Harmonizer's detector (ChordNet) trains on **synthesized chords** —
there is **no external dataset to download**. This keeps training fast, fully
reproducible, and free of licensing constraints.

## Synthetic chord generation

`training/chord_synth.py` renders labeled chords on the fly:

1. **Choose a chord** — a root (12 pitch classes), a quality (major, minor,
   dominant 7th, sus2/sus4, …, from the `QUALITIES` table), and an octave within
   MIDI 36–96 (C2–C7).
2. **Render audio** — sum the chord's notes as harmonic series with a
   **randomized harmonic rolloff** (timbre) and a **randomized overall level**,
   so the model sees a wide range of instruments and gains.
3. **Extract the feature** — the log-frequency feature (61 semitone bins) with
   per-frame RMS normalization, identical to the plugin's `LogFreqFeature`.
4. **Label** — the 12-d pitch-class vector of the chord (multi-label).

Because both the audio and the labels are generated, the "dataset" is infinite
and perfectly balanced; an epoch is simply a batch of freshly synthesized chords.

### Key parameters

| Constant   | Value | Meaning                              |
| ---------- | ----- | ------------------------------------ |
| `SR`       | 24000 | detector sample rate                 |
| `N_FFT`    | 2048  | ~11.7 Hz/bin (resolves down to ~C2)  |
| `MIDI_LO`  | 36    | lowest note (C2)                     |
| `MIDI_HI`  | 96    | highest note (C7)                    |
| `N_PITCH`  | 61    | log-freq bins (one per semitone)     |

These are mirrored into `models/pretrained/chord_info.json` (with the baked
`logfreq_fb` filterbank) so training and the plugin share the exact same feature.

## Testing with real audio

A small vocal sample is committed for offline demos:

```
data/audio_samples/vocals.wav      # dry vocal for harmonizer demos
```

For live use, no files are needed — route a real instrument to the plugin's
sidechain and sing into the main input. Real instrument recordings are **not**
required for training; the synthetic generator covers the detector's needs.

## Directory layout

```
data/
└── audio_samples/        # small committed demo audio (vocals, etc.)
```

> There is no external dataset to download — the detector trains entirely on
> synthesized chords.
