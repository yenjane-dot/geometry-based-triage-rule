# -*- coding: utf-8 -*-
"""Cross-dataset validation: ADE20K scene classification (DINOv2 + noise).

Purpose: Verify that the core DNC-vs-accuracy-drop correlation generalises
beyond COCO. Runs one model (DINOv2), one perturbation (Gaussian noise,
three sigma levels), on a single-label ADE20K scene subset.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, RANDOM_SEED, set_random_seed  # noqa: E402

# Re-use core functions from the strict-protocol script (no copy-paste)
from exp_strict_protocol import (  # type: ignore[import-not-top]
    DEVICE, l2norm, cka, svcca,
    load_model, extract_features, forward_features, validate_features,
    compute_centroids, margin_nearest_centroid, dnc_along,
    fit_linear_probe, probe_accuracy, safe_corr,
    preprocess_imagenet, load_rgb_image,
    apply_noise, image_seed_index,
)

# ---------------------------------------------------------------------------
# ADE20K helpers
# ---------------------------------------------------------------------------


def get_ade20k_root() -> Path:
    env_path = os.environ.get("ADE20K_ROOT")
    if env_path:
        return Path(env_path)

    training_dir = os.environ.get("DATASET_ADE20K_PATH")
    if training_dir:
        p = Path(training_dir)
        try:
            return p.parents[1]
        except IndexError:
            return p

    return Path("path/to/ADE20K/ADEChallengeData2016")


ADE20K_ROOT = get_ade20k_root()
ADE_IMG_DIR = Path(os.environ.get("DATASET_ADE20K_PATH", str(ADE20K_ROOT / "images" / "training")))
ADE_SCENE_FILE = Path(os.environ.get("ADE20K_SCENE_FILE", str(ADE20K_ROOT / "sceneCategories.txt")))


def parse_scene_categories(path: Path) -> Dict[str, str]:
    """Return {image_stem: scene_label}."""
    mapping: Dict[str, str] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "/" not in line and " " not in line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                stem = parts[0].replace("/", "_")   # ADE_train_00000001
                label = "_".join(parts[1:])
                mapping[stem] = label
    return mapping


def build_ade_subset(
    img_dir: Path,
    scene_map: Dict[str, str],
    n_total: int = 0,
    min_per_class: int = 5,
    seed: int = 42,
) -> Tuple[List[Path], np.ndarray, List[str]]:
    """Build a balanced single-label subset from ADE20K.

    Returns (paths, labels_int, class_names).
    """
    img_paths = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])
    if n_total > 0 and len(img_paths) > n_total:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(img_paths), n_total, replace=False)
        img_paths = [img_paths[int(i)] for i in idx]

    # Map each image to its scene label
    label_per_path: List[str] = []
    valid_paths: List[Path] = []
    for p in img_paths:
        stem = p.stem  # ADE_train_00000001
        label = scene_map.get(stem)
        if label is not None:
            label_per_path.append(label)
            valid_paths.append(p)

    # Encode string labels to ints
    unique_labels = sorted(set(label_per_path))
    label2int = {lab: i for i, lab in enumerate(unique_labels)}
    int_labels = np.array([label2int[lab] for lab in label_per_path], dtype=np.int32)

    # Filter by min_per_class
    cls_counts = {i: int(np.sum(int_labels == i)) for i in range(len(unique_labels))}
    keep_cls = {i for i, c in cls_counts.items() if c >= min_per_class}
    keep_idx = [i for i, lab in enumerate(int_labels) if int(lab) in keep_cls]

    paths = [valid_paths[i] for i in keep_idx]
    labels = int_labels[keep_idx]
    class_names = [unique_labels[i] for i in sorted(keep_cls)]

    # Re-index labels (compact)
    old2new = {old: new for new, old in enumerate(sorted(keep_cls))}
    labels = np.array([old2new[int(l)] for l in labels], dtype=np.int32)

    return paths, labels, class_names


def train_test_split(
    paths: List[Path],
    labels: np.ndarray,
    n_train_per_class: int = 15,
    n_test_per_class: int = 10,
    seed: int = 42,
) -> Tuple[List[Path], np.ndarray, List[Path], np.ndarray]:
    """Stratified train/test split with per-class caps."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(labels)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for cls in uniq:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_tr = min(n_train_per_class, len(cls_idx))
        n_te = min(n_test_per_class, max(0, len(cls_idx) - n_tr))
        train_idx.extend(cls_idx[:n_tr].tolist())
        test_idx.extend(cls_idx[n_tr:n_tr + n_te].tolist())

    train_idx = sorted(train_idx)
    test_idx = sorted(test_idx)
    return (
        [paths[int(i)] for i in train_idx],
        labels[train_idx],
        [paths[int(i)] for i in test_idx],
        labels[test_idx],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_random_seed(RANDOM_SEED)

    print("Device:", DEVICE)
    print("Loading scene categories...")
    scene_map = parse_scene_categories(ADE_SCENE_FILE)
    print(f"  {len(scene_map)} entries")

    print("Building ADE20K subset...")
    paths, labels, class_names = build_ade_subset(
        ADE_IMG_DIR, scene_map,
        n_total=5000, min_per_class=10, seed=RANDOM_SEED,
    )
    print(f"  {len(paths)} images, {len(class_names)} classes")

    train_paths, y_tr, test_paths, y_te = train_test_split(
        paths, labels,
        n_train_per_class=20, n_test_per_class=15, seed=RANDOM_SEED,
    )
    n_tr_cls = len(np.unique(y_tr))
    n_te_cls = len(np.unique(y_te))
    print(f"  Train: {len(train_paths)} samples, {n_tr_cls} classes")
    print(f"  Test:  {len(test_paths)} samples, {n_te_cls} classes")

    # Load model
    print("\nLoading DINOv2...")
    model = load_model("dinov2")

    # Extract clean features
    print("Extracting clean features...")
    f_tr = extract_features(model, train_paths, batch_size=32, preprocess_fn=preprocess_imagenet, tag="ade:train")
    f_te = extract_features(model, test_paths, batch_size=32, preprocess_fn=preprocess_imagenet, tag="ade:test")

    # Centroid + margin + probe
    cent = compute_centroids(f_tr, y_tr)
    m_te = margin_nearest_centroid(f_te, y_te, cent)
    scaler, probe = fit_linear_probe(f_tr, y_tr, seed=RANDOM_SEED)
    probe_acc_clean = probe_accuracy(scaler, probe, f_te, y_te)

    out = {
        "model": "dinov2",
        "dataset": "ADE20K",
        "train_size": int(len(train_paths)),
        "test_size": int(len(test_paths)),
        "train_classes": int(n_tr_cls),
        "test_classes": int(n_te_cls),
        "clean_margin_mean": float(np.nanmean(m_te)),
        "clean_margin_neg_pct": float(np.mean(m_te[np.isfinite(m_te)] < 0.0)),
        "clean_probe_acc": probe_acc_clean,
        "perturbations": {},
    }

    # Noise perturbation
    sigmas = [0.02, 0.05, 0.10, 0.20, 0.40]
    seeds = [0]

    out["perturbations"]["noise"] = {"param": "sigma", "values": []}

    for sigma in sigmas:
        per_seed_records = []
        for seed in seeds:
            def _make_noisy(path: Path, fallback_idx: int) -> Image.Image:
                img = load_rgb_image(path)
                idx = image_seed_index(path, fallback_idx)
                return apply_noise(img, float(sigma), seed=seed, idx=idx)

            f_pert_te = extract_features(
                model, test_paths, batch_size=32,
                preprocess_fn=preprocess_imagenet,
                make_image=_make_noisy,
                tag=f"ade:noise={sigma}:seed={seed}",
            )

            cd = 1.0 - np.sum(l2norm(f_te) * l2norm(f_pert_te), axis=1)
            cka_val = cka(l2norm(f_te), l2norm(f_pert_te))
            svcca_val = svcca(l2norm(f_te), l2norm(f_pert_te))
            along = dnc_along(f_te, f_pert_te, y_te, cent)
            margin_pert = margin_nearest_centroid(f_pert_te, y_te, cent)
            probe_acc = probe_accuracy(scaler, probe, f_pert_te, y_te)

            per_seed_records.append({
                "seed": int(seed),
                "cos_dist_mean": float(np.nanmean(cd)),
                "cka": cka_val,
                "svcca": svcca_val,
                "dnc_along_mean": float(np.nanmean(along)),
                "margin_mean": float(np.nanmean(margin_pert)),
                "margin_drop_mean": float(np.nanmean(m_te) - np.nanmean(margin_pert)),
                "probe_acc": probe_acc,
                "probe_acc_drop": float(probe_acc_clean - probe_acc),
            })

        cd_means = np.array([r["cos_dist_mean"] for r in per_seed_records], dtype=np.float32)
        dnc_means = np.array([r["dnc_along_mean"] for r in per_seed_records], dtype=np.float32)
        cka_vals = np.array([r["cka"] for r in per_seed_records], dtype=np.float32)
        svcca_vals = np.array([r["svcca"] for r in per_seed_records], dtype=np.float32)
        margin_drops = np.array([r["margin_drop_mean"] for r in per_seed_records], dtype=np.float32)
        probe_drops = np.array([r["probe_acc_drop"] for r in per_seed_records], dtype=np.float32)

        out["perturbations"]["noise"]["values"].append({
            "sigma": sigma,
            "across_seed": {
                "cos_dist_mean": float(np.mean(cd_means)),
                "cka_mean": float(np.nanmean(cka_vals)),
                "svcca_mean": float(np.nanmean(svcca_vals)),
                "dnc_along_mean": float(np.mean(dnc_means)),
                "margin_drop_mean": float(np.mean(margin_drops)),
                "probe_acc_drop_mean": float(np.mean(probe_drops)),
            },
            "per_seed": per_seed_records,
        })

    # Correlation summary
    family_vals = out["perturbations"]["noise"]["values"]
    out["diagnostic_validation"] = {
        "cka_vs_probe_drop_corr": safe_corr(
            np.array([v["across_seed"]["cka_mean"] for v in family_vals], dtype=np.float32),
            np.array([v["across_seed"]["probe_acc_drop_mean"] for v in family_vals], dtype=np.float32),
        ),
        "svcca_vs_probe_drop_corr": safe_corr(
            np.array([v["across_seed"]["svcca_mean"] for v in family_vals], dtype=np.float32),
            np.array([v["across_seed"]["probe_acc_drop_mean"] for v in family_vals], dtype=np.float32),
        ),
        "cos_vs_probe_drop_corr": safe_corr(
            np.array([v["across_seed"]["cos_dist_mean"] for v in family_vals], dtype=np.float32),
            np.array([v["across_seed"]["probe_acc_drop_mean"] for v in family_vals], dtype=np.float32),
        ),
        "dnc_vs_probe_drop_corr": safe_corr(
            np.array([v["across_seed"]["dnc_along_mean"] for v in family_vals], dtype=np.float32),
            np.array([v["across_seed"]["probe_acc_drop_mean"] for v in family_vals], dtype=np.float32),
        ),
    }

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp_ade20k_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Summary
    print("\n=== Cross-dataset validation (ADE20K, DINOv2, noise) ===")
    for k, v in out["diagnostic_validation"].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
