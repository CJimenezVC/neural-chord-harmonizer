# Notebooks

Exploratory and reporting notebooks for the Chord Harmonizer. Keep heavy
computation in the `.py` modules; notebooks should call into them.

| Notebook                          | Purpose                                                  |
| --------------------------------- | -------------------------------------------------------- |
| `chord_feature_eda.ipynb`         | Inspect synthesized chords: log-freq features, label balance, timbre variety |
| `model_training.ipynb`            | Interactive ChordNet runs, loss/F1 curves, ablations     |
| `evaluation_metrics.ipynb`        | Per-pitch-class precision / recall / F1, confusion across chord qualities |

> Notebooks are gitignored at the checkpoint level (`.ipynb_checkpoints/`).
> Commit cleared-output notebooks to keep diffs small.
