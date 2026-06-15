# Notebooks

Exploratory and reporting notebooks. Keep heavy computation in the `.py`
modules; notebooks should call into them.

| Notebook                          | Purpose                                          |
| --------------------------------- | ------------------------------------------------ |
| `exploratory_data_analysis.ipynb` | Inspect VCC2020/CMU Arctic: durations, F0 ranges, mel stats |
| `model_training.ipynb`            | Interactive training runs, loss curves, ablations |
| `evaluation_metrics.ipynb`        | MCD / F0 RMSE analysis, before/after spectrograms |

> Notebooks are gitignored at the checkpoint level (`.ipynb_checkpoints/`).
> Commit cleared-output notebooks to keep diffs small.
