# Data

See [`../DATASET.md`](../DATASET.md) for the full guide.

The Chord Harmonizer's detector trains on **synthesized** chords
(`training/chord_synth.py`) — there is no external dataset to download. This
folder only holds small committed demo audio:

```
data/
└── audio_samples/     # small committed demo audio (e.g. vocals.wav)
```

For live use, route a real instrument to the plugin's sidechain and sing into
the main input — no data files required.

> The legacy voice-conversion pipeline used a `datasets/` tree (VCC2020 + CMU
> Arctic). That path is deprecated and no longer built or fetched.
