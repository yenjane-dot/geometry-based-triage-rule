# Geometry-Based Triage Rule

This repository contains the code and frozen result files for the reproducibility package associated with the study:

"A Geometry-Based Triage Rule for Deployment-Stage Screening of Vision Transformer Backbones under Common Perturbations"

## What is included

- `experiments/exp_strict_protocol.py`
  Main strict-protocol experiment used for the paper.
- `experiments/exp_cross_dataset.py`
  Supplementary ADE20K cross-dataset noise check.
- `experiments/update_tables.py`
  Summary export script for archived result files.
- `experiments/run_smoke_test.py`
  Lightweight smoke test on fixed sample lists.
- `plot_figures.py`
  Figure regeneration script from archived strict-protocol results.
- `samples/`
  Fixed COCO and ADE20K sample manifests for quick environment checks.
- `run_smoke_test.ps1`
  One-command PowerShell entry for the smoke test.
- `results/exp_strict_20260623_000811.json`
  Archived strict-protocol output.
- `results/exp_ade20k_20260623_085332.json`
  Archived ADE20K supplementary output.

## Data access

The raw datasets are not redistributed in this repository. They must be obtained from the official sources:

- COCO 2017: https://cocodataset.org
- ADE20K: https://groups.csail.mit.edu/vision/datasets/ADE20K

## Environment

Python 3.10+ is recommended.

Install dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Required environment variables

Set dataset paths before running the experiments.

```powershell
$env:DATASET_COCO_PATH='path/to/COCO/train2017'
$env:COCO_ANN_PATH='path/to/COCO/annotations/instances_train2017.json'
$env:DATASET_ADE20K_PATH='path/to/ADE20K/ADEChallengeData2016/images/training'
$env:ADE20K_SCENE_FILE='path/to/ADE20K/ADEChallengeData2016/sceneCategories.txt'
```

The strict-protocol script automatically infers:

- COCO validation images from the sibling `val2017` directory
- COCO validation annotations from the sibling `instances_val2017.json` file

## Quick smoke test

The repository includes fixed sample manifests for a lightweight integration check:

- `samples/coco_train_ids.txt`: 5 COCO `train2017` images
- `samples/coco_val_ids.txt`: 5 COCO `val2017` images
- `samples/ade20k_training_files.txt`: 10 ADE20K training images

Run the smoke test:

```powershell
.\run_smoke_test.ps1
```

Or directly:

```powershell
.\.venv\Scripts\python.exe .\experiments\run_smoke_test.py --model dinov2 --batch_size 4
```

This produces:

```text
outputs/smoke_test_summary.json
```

The smoke test verifies dataset path resolution, model loading, and feature extraction. It is not a substitute for the full experiment.

## Main reproduction

Run the main strict-protocol experiment:

```powershell
.\.venv\Scripts\python.exe .\experiments\exp_strict_protocol.py --split_mode official --single_label --grid full --n_train_use 1024 --n_test_use 384 --min_per_class 5 --seeds 0,1,2,3,4 --bootstrap_n 100 --batch_size 32
```

This produces:

```text
results/exp_strict_YYYYMMDD_HHMMSS.json
```

## Supplementary ADE20K check

```powershell
.\.venv\Scripts\python.exe .\experiments\exp_cross_dataset.py
```

This produces:

```text
results/exp_ade20k_YYYYMMDD_HHMMSS.json
```

## Regenerate figures and summaries

```powershell
.\.venv\Scripts\python.exe .\plot_figures.py
.\.venv\Scripts\python.exe .\experiments\update_tables.py
```

This produces figures under `figures/` and summary files under `outputs/`.

## Reproducibility notes

- The main paper uses the official COCO `train2017 -> val2017` split.
- The filtered single-label subset is generated deterministically from the official annotations using the rules described in the manuscript.
- Perturbation randomness is deterministic per image and per seed.
- The repository includes frozen result files so reviewers can inspect the exact numerical outputs without rerunning the full pipeline.
