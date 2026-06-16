# Chord Harmonizer

**Real-time chord-following vocal harmonizer / auto-tune plugin.** A sidechain
instrument (guitar, bass, or piano) drives a neural polyphonic pitch-class
detector; the main vocal input is pitch-shifted — formant-preserving — onto the
tones of the detected chord, producing a live, chord-aware **choir**. Built on
[JUCE](https://juce.com/) and [RTNeural](https://github.com/jatinchowdhury18/RTNeural).

The neural network is the *listening* half: it transcribes whatever the
instrument is playing into 12 pitch-class activations in real time. The DSP is
the *singing* half: a bank of formant-preserving phase-vocoder pitch shifters
re-voices the singer onto those notes. A single **Tune** control sweeps from
natural harmony to hard-snap (T-Pain) auto-tune.

## Features

- Real-time, chord-following vocal harmony driven by a sidechain instrument
- Neural **polyphonic** pitch-class detector (ChordNet) — guitar/bass/piano chords
- Formant-preserving phase-vocoder pitch shifting (the voice stays the voice)
- **Choir** mode: one harmony voice per detected chord tone (up to 6)
- **Tune** (natural → tight), **Gate** (instrument noise gate), **Polyphony** (1–6)
- VST3 / AU / Standalone for DAW integration

## Architecture at a Glance

```
Sidechain instrument (host rate)        Main voice (host rate)
  → Downsample to 24 kHz                  → YIN F0 (pitch tracking)
  → Log-freq feature (61 semitone bins)   │
  → ChordNet (→ 12 pitch-class probs)     │
  → Peak-hold + noise gate                │
  → detected chord ───────────────────────┤
                                          ▼
                          Choir: for each chord tone, a
                          formant-preserving phase-vocoder
                          pitch shift of the voice → sum
                                          ▼
                              Output (host rate)
```

The **detector** runs at a fixed **24 kHz** (its trained feature rate); the
**voice path** runs at the host rate. When no chord is detected (or the
instrument is below the gate), the output is silent.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`TECHNICAL.md`](TECHNICAL.md)
for the detailed design.

## Repository Layout

| Path          | Contents                                                |
| ------------- | ------------------------------------------------------- |
| `training/`   | Python training pipeline (synthetic chords → RTNeural)  |
| `plugin/`     | JUCE VST3/AU/Standalone plugin (C++)                    |
| `models/`     | Trained ChordNet + RTNeural export                      |
| `bindings/`   | pybind11 binding exposing the C++ DSP feature to Python |
| `benchmarks/` | Latency / CPU / memory measurements                     |
| `docs/`       | Architecture, training, build, and DSP documentation    |

## Quick Start

### Build the plugin

```bash
# Prerequisites: CMake >= 3.22, a C++20 compiler
git clone https://github.com/CJimenezVC/adaptive-voice-transform.git
cd adaptive-voice-transform

# JUCE and RTNeural are pulled automatically via CMake FetchContent
./scripts/build_plugin.sh
```

### Run it

1. Load the plugin on your **vocal** track.
2. Route a melodic instrument (guitar/bass/piano) to the plugin's **sidechain**.
3. Click **Load Models...** and select `models/pretrained/`.
4. Play a chord on the instrument and sing — the voice follows the chord.

Controls: **Tune** (0 = natural blend, 1 = tight snap), **Gate** (instrument
detection threshold), **Polyphony** (max simultaneous harmony notes, e.g. 6 for
a full guitar chord).

### Train your own detector

No external dataset required — the detector trains on **synthesized** chords:

```bash
cd training
pip install -r requirements.txt
python train_chord.py     # trains ChordNet on synthetic chords (Apple MPS / CPU)
python export_chord.py    # -> models/pretrained/chordnet.rtneural + chord_info.json
```

See [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) for the full walkthrough.

## Validation

Every piece of the chain is validated against its Python reference:

| Component                  | Check                              | Result        |
| -------------------------- | ---------------------------------- | ------------- |
| Log-freq feature (C++)     | vs `training/chord_synth.py`       | ~3.4e-5       |
| ChordNet RTNeural export   | vs PyTorch                         | ~1.8e-7       |
| Formant-preserving shifter | corr vs Python reference           | ~0.996        |
| ChordNet detection         | F1 on held-out synthetic chords    | ~0.996        |

The training feature is computed by the plugin's **exact C++ DSP** via a
pybind11 binding (`bindings/`), so there is no Python/C++ feature drift.

> **Legacy:** an earlier neural voice-conversion engine ("Adaptive Voice
> Transform") lives in the tree (`ML/Encoder|Decoder|VocoderNetwork`,
> `DSP/FeatureExtractor`, etc.). It is **deprecated** and not built into the
> current plugin target; the harmonizer is the active product.

## References

- YIN pitch detection — de Cheveigné & Kawahara (2002)
- Phase-vocoder pitch shifting — Laroche & Dolson (1999)
- [RTNeural](https://github.com/jatinchowdhury18/RTNeural)
- [JUCE](https://juce.com/)

## License

GNU GPL v3.0 with the **Commons Clause** — see [`LICENSE`](LICENSE). You may
use, modify, and redistribute the software under the GPLv3; the Commons Clause
adds one restriction: you may **not Sell** it (i.e. provide it, or a service
whose value derives substantially from it, for a fee). For commercial licensing,
contact the author.

> Note: the Commons Clause makes this **source-available**, not OSI
> open-source — the "no Sell" condition is a restriction GPLv3 alone does not
> impose.
