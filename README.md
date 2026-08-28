# Geometry-Based ViT Backbone Screening

This repository contains the public reproducibility package for a geometry-based Vision Transformer backbone screening study. It includes the experiment code, frozen result files, sample manifests, and rerun instructions.

## What is included

- `experiments/exp_strict_protocol.py`
  Main strict-protocol experiment: geometric diagnostics (cosine shift, decision-nearest-centroid displacement, nearest-centroid margin) and representational-similarity baselines (CKA, SVCCA), with a strict COCO train/val split, train-only centroids, a deterministic 3-family perturbation grid (noise, occlusion, blur; 15 conditions), 5 seeds, and bootstrap confidence intervals.
- `experiments/exp_offpool.py`
  Off-pool validation: splits candidate backbones into a fit cohort (used for the DSS z-score normalisation) and a disjoint hold cohort, and tests whether the screening rule ranks unseen candidates.
- `experiments/plot_offpool.py`
  Regenerates the off-pool comparison figures/tables from a frozen off-pool result.
- `experiments/exp_cross_dataset.py`
  Supplementary ADE20K cross-dataset noise check.
- `experiments/update_tables.py`
  Summary export script for archived strict-protocol result files.
- `experiments/run_smoke_test.py` and `run_smoke_test.ps1`
  Lightweight smoke test on fixed sample manifests.
- `plot_figures.py`
  Regenerates diagnostic figures from a frozen strict-protocol result.
- `samples/`
  Fixed COCO and ADE20K sample manifests for quick environment checks.
- `results/`
  Frozen outputs: the main 15-condition strict result, the ADE20K supplementary result, and the off-pool (11 held-out backbones) result.

## Data access

The raw datasets are not redistributed here. They must be obtained from the official sources:

- COCO 2017: https://cocodataset.org
- ADE20K: https://groups.csail.mit.edu/vision/datasets/ADE20K

## Environment

Python 3.10+ recommended. Install dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Required environment variables

Set the dataset paths before running the experiments:

```powershell
$env:DATASET_COCO_PATH='path/to/COCO/train2017'
$env:COCO_ANN_PATH='path/to/COCO/annotations/instances_train2017.json'
$env:DATASET_ADE20K_PATH='path/to/ADE20K/ADEChallengeData2016/images/training'
$env:ADE20K_SCENE_FILE='path/to/ADE20K/ADEChallengeData2016/sceneCategories.txt'
```

The strict-protocol script infers the COCO validation images from the sibling `val2017` directory and the validation annotations from the sibling `instances_val2017.json`.

## Quick smoke test

```powershell
.\run_smoke_test.ps1
# or
.\.venv\Scripts\python.exe .\experiments\run_smoke_test.py --model dinov2 --batch_size 4
```

Produces `outputs/smoke_test_summary.json`.

## Main reproduction

```powershell
.\.venv\Scripts\python.exe .\experiments\exp_strict_protocol.py --split_mode official --single_label --grid full --n_train_use 1024 --n_test_use 384 --min_per_class 5 --seeds 0,1,2,3,4 --bootstrap_n 100 --batch_size 32
```

Produces `results/exp_strict_YYYYMMDD_HHMMSS.json`.

## Off-pool validation reproduction

```powershell
.\.venv\Scripts\python.exe .\experiments\exp_offpool.py --fit dinov2,clip,mae,supervised_vit --hold swin_base,convnext_base,vit_base_clip,dino_vits8,vit_base_dino,resnet50,clip_vit_patch32,deit_tiny,vit_small,swin_tiny,pvt_b0 --grid full --seeds 0,1,2,3,4 --n_train_use 1024 --n_test_use 384 --min_per_class 5 --batch_size 16
```

Produces `results/exp_offpool_YYYYMMDD_HHMMSS.json`.

## Regenerate figures and summaries

```powershell
.\.venv\Scripts\python.exe .\plot_figures.py
.\.venv\Scripts\python.exe .\experiments\update_tables.py
:: off-pool figures/tables:
.\.venv\Scripts\python.exe .\experiments\plot_offpool.py --input results\exp_offpool_20260828_100548.json --out . --tag final
```

`plot_figures.py` writes figures to `outputs/figures/` by default. Set `RES_PATH` if you want to point it to a specific strict-protocol JSON.

## Reproducibility notes

- The main strict protocol uses the official COCO `train2017 -> val2017` split and a deterministic single-label subset.
- The perturbation grid is a fixed 3-family, 15-condition grid; perturbation randomness is deterministic per image and per seed.
- The off-pool split (fit vs. hold) is explicit and disjoint; the hold candidates never contribute to the DSS normalisation.
- The repository includes frozen result files so reviewers can inspect the exact numerical outputs without rerunning the full pipeline.
