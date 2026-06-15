# Dataset Preparation Guide

Adaptive Voice Transform is trained on **VCC2020** (parallel conversion pairs)
and evaluated on **CMU Arctic** (high-quality reference speakers).

## Datasets

### VCC2020 — Voice Conversion Challenge 2020 (training)

- **4 source speakers** (English): `SEF1`, `SEF2`, `SEM1`, `SEM2`
- **10 target speakers**: `TEF1`, `TEF2`, `TEM1`, `TEM2` (English, Task 1) and
  `TFF1`, `TFM1`, `TGF1`, `TGM1`, `TMF1`, `TMM1` (Finnish/German/Mandarin, Task 2)
- Speaker code: `S`/`T` = source/target, 2nd letter = language
  (`E`/`F`/`G`/`M` = English/Finnish/German/Mandarin), 3rd = `M`/`F` (male/female)
- 70 sentences per speaker; 24 kHz, 16-bit mono WAV
- **Parallel pairs share a filename** (e.g. `SEF1/E10051.wav` ≡ `TEF1/E10051.wav`);
  IDs 20001–20050 are nonparallel
- **Task 1** (intra-lingual, 16 source→target pairs) is the parallel-data task
  used here; Task 2 is cross-lingual
- Openly licensed (ODbL with DbCL 1.0 — commercial use permitted; no registration)
- Source: https://github.com/nii-yamagishilab/VCC2020-database

> Our pipeline resamples the 24 kHz source audio to 48 kHz during preprocessing
> (`audio.sample_rate` in `config.yaml`).

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
│   ├── vcc2020/                  # extracted from the database ZIPs
│   │   ├── vcc2020_training/      # SEF1/ SEF2/ SEM1/ SEM2/ TEF1/ ... per-speaker WAVs
│   │   ├── vcc2020_evaluation/    # source-speaker eval recordings
│   │   ├── vcc2020_groundtruth/   # target-speaker English references
│   │   └── prompts/               # transcriptions
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
