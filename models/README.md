# Models

| Subdir          | Contents                                                          |
| --------------- | ---------------------------------------------------------------- |
| `pretrained/`   | RTNeural export the plugin loads: `chordnet.rtneural` + `chord_info.json` |
| `pytorch/`      | ChordNet training checkpoint(s) and `chord_info.json`            |
| `quantization/` | Quantization accuracy reports (optional)                         |

The plugin loads two files from `pretrained/`:

- **`chordnet.rtneural`** — the dense ChordNet weights (61→256→256→12), read by
  the self-contained `NNModel` engine.
- **`chord_info.json`** — feature metadata: `in_dim` (61), `n_fft` (2048),
  `sample_rate` (24000), `midi_lo` (36), `midi_hi` (96), and the baked
  `logfreq_fb` filterbank used by `LogFreqFeature`.

Produce them with:

```bash
cd training
python train_chord.py     # -> models/pytorch/ checkpoint
python export_chord.py    # -> models/pretrained/chordnet.rtneural + chord_info.json
```

Load into the plugin via the editor's **Load Models...** button or the
`AVT_MODELS_DIR` environment variable.

> Legacy `encoder.rtneural` / `decoder.rtneural` / `vocoder.rtneural` and
> `model_info.json` may still be present from the deprecated voice-conversion
> engine; the harmonizer does not use them.
