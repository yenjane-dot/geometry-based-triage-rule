"""Run a lightweight smoke test on fixed COCO and ADE20K sample lists.

This script is not intended to reproduce the full paper. It verifies that:
1. dataset paths are configured correctly,
2. the public repository can load a model,
3. feature extraction works on small fixed sample sets.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS_DIR, get_ade20k_scene_file, get_dataset_path  # noqa: E402
from exp_strict_protocol import (  # noqa: E402
    DEVICE,
    extract_features,
    load_model,
    preprocess_imagenet,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight smoke test on fixed sample lists.")
    parser.add_argument("--model", type=str, default="dinov2", choices=["dinov2", "clip", "mae", "supervised_vit"])
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def read_lines(path: Path) -> List[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def resolve_coco_sample_paths() -> List[Path]:
    train_dir = get_dataset_path("COCO")
    val_dir = train_dir.parent / "val2017"

    train_ids = read_lines(SAMPLES_DIR / "coco_train_ids.txt")
    val_ids = read_lines(SAMPLES_DIR / "coco_val_ids.txt")

    paths = [train_dir / f"{image_id}.jpg" for image_id in train_ids]
    paths.extend(val_dir / f"{image_id}.jpg" for image_id in val_ids)
    return paths


def resolve_ade20k_sample_paths() -> List[Path]:
    ade_dir = get_dataset_path("ADE20K")
    _ = get_ade20k_scene_file()
    rel_names = read_lines(SAMPLES_DIR / "ade20k_training_files.txt")
    return [ade_dir / name for name in rel_names]


def ensure_existing(paths: List[Path], label: str) -> List[Path]:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {label} sample files: {missing}")
    return paths


def summarize_features(features: np.ndarray) -> dict:
    norms = np.linalg.norm(features, axis=1)
    return {
        "n_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
    }


def main() -> None:
    args = parse_args()

    coco_paths = ensure_existing(resolve_coco_sample_paths(), "COCO")
    ade_paths = ensure_existing(resolve_ade20k_sample_paths(), "ADE20K")

    print(f"Device: {DEVICE}")
    print(f"Model: {args.model}")
    print(f"COCO smoke samples: {len(coco_paths)}")
    print(f"ADE20K smoke samples: {len(ade_paths)}")

    model = load_model(args.model)

    print("Extracting COCO sample features...")
    coco_features = extract_features(
        model,
        coco_paths,
        batch_size=int(args.batch_size),
        preprocess_fn=preprocess_imagenet,
        tag="smoke:coco",
    )

    print("Extracting ADE20K sample features...")
    ade_features = extract_features(
        model,
        ade_paths,
        batch_size=int(args.batch_size),
        preprocess_fn=preprocess_imagenet,
        tag="smoke:ade20k",
    )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(DEVICE),
        "model": args.model,
        "coco": summarize_features(coco_features),
        "ade20k": summarize_features(ade_features),
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "smoke_test_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved smoke-test summary to: {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
