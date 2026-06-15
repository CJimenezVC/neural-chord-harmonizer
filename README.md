# Adaptive Voice Transform

**Real-time neural style transfer voice plugin.** Continuously control timbre,
pitch, and resonance with a hybrid DSP-neural architecture built on
[JUCE](https://juce.com/) and [RTNeural](https://github.com/jatinchowdhury18/RTNeural).

Rather than end-to-end voice conversion, Adaptive Voice Transform implements a
**differentiable DSP chain**: deterministic DSP extracts pitch and formants,
lightweight neural networks learn and apply voice style transformations, and
real-time parameters modulate the style continuously — all at sub-50 ms latency
suitable for DAW use.

## Features

- Real-time voice transformation (~40–50 ms latency)
- Independent, continuous control over style, timbre, formants, and pitch
- VST3 / AU plugin for DAW integration
- Multi-speaker conversion trained on VCC2020 + CMU Arctic
- Streaming WaveRNN vocoder optimized for RTNeural

## Architecture at a Glance

```
Input (host rate, e.g. 48 kHz)
  → Downsample to 24 kHz
  → Feature Extraction (STFT, mel-spectrogram, YIN F0, formants)
  → Neural Encoder (mel → 64-dim style vector)
  → Style Modulation (user parameter interpolation)
  → Neural Decoder ((mel, style) → transformed mel)
  → WaveRNN Vocoder (mel → waveform)
  → Post-DSP refinement (overlap-add, artifact suppression)
  → Upsample to host rate
Output (host rate, ~40 ms latency)
```

The neural models run at **24 kHz** (VCC2020's native rate) to halve compute;
the plugin resamples to/from the host rate around the neural chain.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`TECHNICAL.md`](TECHNICAL.md)
for the detailed design.

## Repository Layout

| Path          | Contents                                              |
| ------------- | ----------------------------------------------------- |
| `training/`   | Python training pipeline (PyTorch → RTNeural export)  |
| `plugin/`     | JUCE VST3/AU plugin (C++)                              |
| `models/`     | Trained checkpoints and RTNeural exports              |
| `benchmarks/` | Latency / CPU / memory measurements                   |
| `data/`       | Datasets, preprocessed features, splits               |
| `scripts/`    | Automation (download, preprocess, train, export, build) |
| `docs/`       | Architecture, training, build, and DSP documentation  |

## Quick Start

### Build the plugin

```bash
# Prerequisites: CMake >= 3.22, a C++20 compiler
git clone https://github.com/yourusername/adaptive-voice-transform.git
cd adaptive-voice-transform

# JUCE and RTNeural are pulled automatically via CMake FetchContent
./scripts/build_plugin.sh
```

Output artifacts:
- `adaptive-voice-transform.vst3`
- `adaptive-voice-transform.component` (macOS AU)

### Train your own model

```bash
cd training
pip install -r requirements.txt
../scripts/download_datasets.sh
../scripts/preprocess_data.sh
../scripts/train.sh
../scripts/export_models.sh
```

See [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) for the full walkthrough.

## Performance Targets

| Metric  | Target                          |
| ------- | ------------------------------- |
| Latency | 40–50 ms                        |
| CPU     | < 25 % (modern 4-core)          |
| Memory  | ~200 MB resident                |
| Params  | ~500 K (encoder + decoder + vocoder) |

## Status

🚧 **Scaffolding stage.** The repository structure, interfaces, and documentation
are in place. Implementations are marked with `TODO` and are being filled in.
See individual module headers for the intended API.

## References

- YIN pitch detection — de Cheveigné & Kawahara (2002)
- WaveRNN vocoder — Kalchbrenner et al. (2018)
- [RTNeural](https://github.com/jatinchowdhury18/RTNeural)
- [JUCE](https://juce.com/)
- [Voice Conversion Challenge 2020 database](https://github.com/nii-yamagishilab/VCC2020-database)

## License

MIT — see [`LICENSE`](LICENSE).
