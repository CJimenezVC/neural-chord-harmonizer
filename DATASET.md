# Dataset Preparation Guide

Adaptive Voice Transform is trained on **VCC2020** (parallel conversion pairs)
and evaluated on **CMU Arctic** (high-quality reference speakers).

## Datasets

### VCC2020 — Voice Conversion Challenge 2020 (training)

- Parallel pairs: `P225↔P226`, `P227↔P228`, `P229↔P230`, `P231↔P232`
- ~10 hours per speaker pair
- 16 kHz, mono, already time-aligned
- Source: https://www.voiceconversionchallenge.org/

### CMU Arctic (evaluation / generalization)

- 4 professional speakers: `bdl`, `clb`, `jmk`, `awb`
- ~1 hour each, high quality
- Used for out-of-domain generalization testing
- Source: http://www.festvox.org/cmu_arctic/

## Data Splits

| Split      | Source                              | Fraction |
| ---------- | ----------------------------------- | -------- |
| Training   | VCC2020 pairs                       | 80 %     |
| Validation | VCC2020 held-out speakers           | 10 %     |
| Test       | CMU Arctic + out-of-domain voices   | 10 %     |

Split manifests live in `data/splits/{train,val,test}.txt`.

## Directory Layout

```
data/
├── datasets/
│   ├── vcc2020/
│   │   ├── train/
│   │   │   ├── speaker1_source/
│   │   │   ├── speaker1_target/
│   │   │   └── ...
│   │   └── eval/
│   ├── cmu_arctic/
│   │   ├── bdl/  clb/  jmk/  awb/
│   └── preprocessed/
│       ├── train_features.h5
│       ├── eval_features.h5
│       └── data_stats.json
├── audio_samples/        # before/after demos (committed, small)
└── splits/
    ├── train.txt  val.txt  test.txt
```

> **Note:** Raw audio and the `datasets/` tree are **not** committed (see
> `.gitignore`). Use the scripts below to fetch and build them locally.

## Pipeline

### 1. Download

```bash
./scripts/download_datasets.sh
```

Downloads VCC2020 and CMU Arctic into `data/datasets/`.

### 2. Preprocess

```bash
./scripts/preprocess_data.sh
```

This runs `training/preprocess.py`, which:

1. Resamples all audio to 48 kHz mono.
2. Computes mel-spectrograms (128 bins) and F0 (YIN) per utterance.
3. Computes dataset normalization statistics (`data_stats.json`).
4. Writes packed feature tensors to `train_features.h5` / `eval_features.h5`.

### 3. Build splits

Split manifests are generated during preprocessing. To regenerate manually:

```bash
python training/preprocess.py --make-splits --seed 1234
```

## Normalization

Features are normalized using global mean/std stored in `data_stats.json`:

```json
{
  "mel_mean": [...],   "mel_std": [...],
  "f0_mean":  123.4,   "f0_std":  45.6,
  "sample_rate": 48000, "n_mels": 128
}
```

The plugin loads the same statistics at inference time to match training.
