"""
Project-wide configuration.
All dataset paths MUST be provided via environment variables at runtime.
No local absolute paths are committed.
"""

from pathlib import Path
import os

# Project root (inferred from this file location)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core directories
PAPER_DIR = PROJECT_ROOT / "paper"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

# Output subdirectories
EXP1_OUTPUT_DIR = OUTPUTS_DIR / "exp1_significance"
EXP2_OUTPUT_DIR = OUTPUTS_DIR / "exp2_multi_arch"
EXP3_OUTPUT_DIR = OUTPUTS_DIR / "exp3_downstream"
EXP4_OUTPUT_DIR = OUTPUTS_DIR / "exp4_dynamics"

# Ensure output directories exist
for dir_path in [OUTPUTS_DIR, EXP1_OUTPUT_DIR, EXP2_OUTPUT_DIR,
                 EXP3_OUTPUT_DIR, EXP4_OUTPUT_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset paths - MUST be set via environment variables
# No default local paths are provided to avoid accidental misuse
# Required env vars:
#   DATASET_COCO_PATH        -> COCO train2017 images directory
#   COCO_ANN_PATH            -> COCO train2017 annotations JSON
#   DATASET_ADE20K_PATH      -> ADE20K training images directory
#   ADE20K_SCENE_FILE        -> ADE20K sceneCategories.txt
# Example (PowerShell):
#   $env:DATASET_COCO_PATH='path/to/coco/train2017'
#   $env:COCO_ANN_PATH='path/to/coco/annotations/instances_train2017.json'
#   $env:DATASET_ADE20K_PATH='path/to/ADE20K/images/training'
#   $env:ADE20K_SCENE_FILE='path/to/ADE20K/sceneCategories.txt'

def get_dataset_path(dataset_name: str) -> Path:
    """Get dataset path from environment variable. Raises if not set."""
    env_var = f"DATASET_{dataset_name.upper()}_PATH"
    if env_var not in os.environ:
        raise RuntimeError(
            f"Environment variable {env_var} is not set. "
            f"Please set it to the {dataset_name} dataset directory."
        )
    path = Path(os.environ[env_var])
    if not path.exists():
        raise RuntimeError(f"Path from {env_var} does not exist: {path}")
    return path


def get_coco_ann_path() -> Path:
    """Get COCO annotation path from environment variable. Raises if not set."""
    env_var = "COCO_ANN_PATH"
    if env_var not in os.environ:
        raise RuntimeError(
            f"Environment variable {env_var} is not set. "
            f"Please set it to the COCO instances_train2017.json file."
        )
    path = Path(os.environ[env_var])
    if not path.exists():
        raise RuntimeError(f"Path from {env_var} does not exist: {path}")
    return path


def get_ade20k_scene_file() -> Path:
    """Get ADE20K sceneCategories.txt path from environment variable."""
    env_var = "ADE20K_SCENE_FILE"
    if env_var not in os.environ:
        raise RuntimeError(
            f"Environment variable {env_var} is not set. "
            f"Please set it to the ADE20K sceneCategories.txt file."
        )
    path = Path(os.environ[env_var])
    if not path.exists():
        raise RuntimeError(f"Path from {env_var} does not exist: {path}")
    return path


# Random seed configuration
RANDOM_SEED = 42

# Model configurations (HuggingFace / timm identifiers)
MODELS_CONFIG = {
    "dinov2": {"repo": "facebook/dinov2-base", "name": "dinov2"},
    "clip": {"repo": "openai/clip-vit-base-patch16", "name": "clip"},
    "mae": {"repo": "facebook/vit-mae-base", "name": "mae"},
    "dinov1": {"repo": "facebook/dino-vits8", "name": "dinov1"},
}

# Experiment defaults
DEFAULT_N_SAMPLES = 2000
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_REP = 20
PCA_N_COMPONENTS = 50


def set_random_seed(seed: int = RANDOM_SEED):
    """Set global random seed for reproducibility."""
    import numpy as np
    import torch
    import random

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Fix cudnn for full determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False