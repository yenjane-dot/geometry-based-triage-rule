# Geometry-Based Triage Rule

This repository contains the code and frozen result files for the reproducibility package associated with the study:

"A Geometry-Based Triage Rule for Deployment-Stage Screening of Vision Transformer Backbones under Common Perturbations"

## What is included

- `experiments/exp_strict_protocol.py`
  Main strict-protocol experiment used for the paper.
- `experiments/exp_cross_dataset.py`
  Supplementary ADE20K cross-dataset noise check.
- `experiments/update_tables.py`
  Table refresh script.
- `plot_figures.py`
  Figure refresh script.
- `results/exp_strict_20260623_000811.json`
  Archived strict-protocol output.
- `results/exp_ade20k_20260623_085332.json`
  Archived ADE20K supplementary output.

## What is not included

- Raw COCO or ADE20K images
- Manuscript source files
- Submission-only files such as cover letters or title pages
- Private local paths

The raw datasets are not redistributed. They must be obtained from the official sources:

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
$env:DATASET_ADE20K_PATH='path/to/ADE20K/images/training'
$env:ADE20K_SCENE_FILE='path/to/ADE20K/sceneCategories.txt'
```

The strict-protocol script automatically infers:

- COCO validation images from the sibling `val2017` directory
- COCO validation annotations from the sibling `instances_val2017.json` file

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

## Refresh figures and tables

```powershell
.\.venv\Scripts\python.exe .\plot_figures.py
.\.venv\Scripts\python.exe .\experiments\update_tables.py
```

## Reproducibility notes

- The main paper uses the official COCO `train2017 -> val2017` split.
- The filtered single-label subset is generated deterministically from the official annotations using the rules described in the manuscript.
- Perturbation randomness is deterministic per image and per seed.
- The repository includes frozen result files so reviewers can inspect the exact numerical outputs without rerunning the full pipeline.
