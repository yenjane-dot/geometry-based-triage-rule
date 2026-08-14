"""
严格协议实验（建议作为 SCI 投稿主协议）

目标：
1) 使用真正的 train -> test 官方拆分，避免从同一目录内部随机切分造成样本过小。
2) 修正扰动随机性：噪声/遮挡对每张图像可复现，但不同图像不会共享同一扰动。
3) 同时报告几何诊断与线性探针准确率，验证诊断是否对应真实性能下降。
4) 对特征提取结果做显式校验，避免坏样本或零向量被静默吞掉后污染统计。

输出：
  results/exp_strict_<timestamp>.json

依赖（与你现有项目一致）：
  torch, numpy, pillow, tqdm, transformers, timm, scikit-learn, scipy
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, RANDOM_SEED, get_coco_ann_path, get_dataset_path, set_random_seed  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Protocol:
    split_mode: str = "official"      # {"official", "random"}
    n_total: int = 5000                # random split only
    n_use: int = 2000                  # random split only
    n_total_train: int = 0             # official split only, 0 means all
    n_total_test: int = 0              # official split only, 0 means all
    n_train_use: int = 8000            # official split only, 0 means no cap
    n_test_use: int = 2000             # official split only, 0 means no cap
    train_ratio: float = 0.8           # random split only
    min_per_class: int = 10
    enforce_single_label: bool = True
    single_label_max_cats: int = 1
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    bootstrap_n: int = 500
    bootstrap_alpha: float = 0.05
    grid: str = "full"                # {"full", "minimal"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict protocol experiment (train/test, multi-seed, CI).")
    p.add_argument("--split_mode", type=str, default="official", choices=["official", "random"],
                   help="official: COCO train2017 -> val2017; random: split a subset from train2017 only.")
    p.add_argument("--n_total", type=int, default=5000, help="Random split only: candidate pool size scanned from disk.")
    p.add_argument("--n_use", type=int, default=2000, help="Random split only: final used samples after filtering.")
    p.add_argument("--n_total_train", type=int, default=0, help="Official split only. 0 means use all train images.")
    p.add_argument("--n_total_test", type=int, default=0, help="Official split only. 0 means use all test images.")
    p.add_argument("--n_train_use", type=int, default=8000, help="Official split only. 0 means no cap.")
    p.add_argument("--n_test_use", type=int, default=2000, help="Official split only. 0 means no cap.")
    p.add_argument("--train_ratio", type=float, default=0.8, help="Random split only: train ratio for split.")
    p.add_argument("--min_per_class", type=int, default=10, help="Min samples per class kept in both splits.")
    p.add_argument("--single_label", action="store_true", help="Force single-label COCO subset.")
    p.add_argument("--no_single_label", action="store_true", help="Disable single-label filtering.")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seeds for perturbations.")
    p.add_argument("--bootstrap_n", type=int, default=500, help="Bootstrap replicates for CI.")
    p.add_argument("--grid", type=str, default="full", choices=["full", "minimal"], help="Perturbation grid profile.")
    p.add_argument("--models", type=str, default="dinov2,clip,mae,supervised_vit",
                   help="Comma-separated model tags: dinov2,clip,mae,supervised_vit")
    p.add_argument("--clip_preprocess", type=str, default="imagenet", choices=["imagenet", "clip"],
                   help="CLIP preprocessing: imagenet (default) or clip-native (sensitivity analysis).")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction.")
    return p.parse_args()


def list_images(img_dir: Path, n_total: int) -> List[Path]:
    paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if int(n_total) > 0:
        return paths[:n_total]
    return paths


def load_coco_mapping(ann_path: Path) -> Tuple[Dict[int, List[int]], Dict[int, str]]:
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    img_to_cats: Dict[int, set] = {}
    for ann in coco.get("annotations", []):
        iid = int(ann["image_id"])
        cid = int(ann["category_id"])
        img_to_cats.setdefault(iid, set()).add(cid)
    img_to_cats_list = {k: sorted(list(v)) for k, v in img_to_cats.items()}
    return img_to_cats_list, cat_id_to_name


def build_coco_subset(
    img_paths: List[Path],
    img_to_cats: Dict[int, List[int]],
    proto: Protocol,
    limit: int,
) -> Tuple[List[Path], np.ndarray]:
    kept_paths: List[Path] = []
    labels: List[int] = []
    for p in img_paths:
        try:
            iid = int(p.stem)
        except Exception:
            continue
        cats = img_to_cats.get(iid, [])
        if proto.enforce_single_label:
            if len(cats) == 0 or len(cats) > proto.single_label_max_cats:
                continue
            label = int(cats[0])
        else:
            if len(cats) == 0:
                continue
            label = int(cats[0])
        kept_paths.append(p)
        labels.append(label)
        if int(limit) > 0 and len(kept_paths) >= int(limit):
            break
    return kept_paths, np.asarray(labels, dtype=np.int64)


def filter_by_min_class(paths: List[Path], labels: np.ndarray, min_per_class: int) -> Tuple[List[Path], np.ndarray]:
    uniq, cnt = np.unique(labels, return_counts=True)
    keep_classes = {int(u) for u, c in zip(uniq, cnt) if int(c) >= int(min_per_class)}
    mask = np.asarray([int(y) in keep_classes for y in labels], dtype=bool)
    return [p for p, ok in zip(paths, mask) if ok], labels[mask]


def filter_to_common_classes(
    train_paths: List[Path],
    train_labels: np.ndarray,
    test_paths: List[Path],
    test_labels: np.ndarray,
    min_per_class: int,
) -> Tuple[List[Path], np.ndarray, List[Path], np.ndarray]:
    train_uniq, train_cnt = np.unique(train_labels, return_counts=True)
    test_uniq, test_cnt = np.unique(test_labels, return_counts=True)
    train_keep = {int(u) for u, c in zip(train_uniq, train_cnt) if int(c) >= int(min_per_class)}
    test_keep = {int(u) for u, c in zip(test_uniq, test_cnt) if int(c) >= int(min_per_class)}
    common = train_keep & test_keep
    tr_mask = np.asarray([int(y) in common for y in train_labels], dtype=bool)
    te_mask = np.asarray([int(y) in common for y in test_labels], dtype=bool)
    return (
        [p for p, ok in zip(train_paths, tr_mask) if ok],
        train_labels[tr_mask],
        [p for p, ok in zip(test_paths, te_mask) if ok],
        test_labels[te_mask],
    )


def balanced_cap(paths: List[Path], labels: np.ndarray, limit: int, seed: int) -> Tuple[List[Path], np.ndarray]:
    if int(limit) <= 0 or len(paths) <= int(limit):
        return paths, labels
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels)
    target_per_class = max(1, int(limit) // max(1, len(classes)))
    picked: List[int] = []
    leftovers: List[int] = []
    for cls in classes:
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        take = min(len(idx), target_per_class)
        picked.extend(idx[:take].tolist())
        leftovers.extend(idx[take:].tolist())
    if len(picked) < int(limit):
        rng.shuffle(leftovers)
        picked.extend(leftovers[: int(limit) - len(picked)])
    picked = sorted(picked[: int(limit)])
    return [paths[i] for i in picked], labels[np.asarray(picked, dtype=np.int64)]


def preprocess_imagenet(pil_img: Image.Image, size: int = 224) -> torch.Tensor:
    pil_img = pil_img.resize((256, 256), Image.BILINEAR)
    left = (256 - size) // 2
    top = (256 - size) // 2
    pil_img = pil_img.crop((left, top, left + size, top + size))
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr.transpose(2, 0, 1))


def preprocess_clip(pil_img: Image.Image) -> torch.Tensor:
    clip_mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    clip_std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    pil_img = pil_img.resize((224, 224), Image.BICUBIC)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = (arr - clip_mean) / clip_std
    return torch.from_numpy(arr.transpose(2, 0, 1))


def apply_noise(img: Image.Image, sigma: float, seed: int, idx: int) -> Image.Image:
    rng = np.random.RandomState(seed * 1000003 + idx)
    arr = np.array(img, dtype=np.float32) / 255.0
    noise = rng.randn(*arr.shape).astype(np.float32) * sigma
    return Image.fromarray((np.clip(arr + noise, 0, 1) * 255).astype(np.uint8))


def apply_occlusion(img: Image.Image, fraction: float, seed: int, idx: int) -> Image.Image:
    rng = np.random.RandomState(seed * 1000003 + idx)
    w, h = img.size
    bw, bh = int(w * math.sqrt(fraction)), int(h * math.sqrt(fraction))
    x = int(rng.randint(0, max(1, w - bw)))
    y = int(rng.randint(0, max(1, h - bh)))
    arr = np.array(img).copy()
    arr[y: y + bh, x: x + bw] = 0
    return Image.fromarray(arr)


def apply_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=float(radius)))


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def image_seed_index(path: Path, fallback_idx: int) -> int:
    try:
        return int(path.stem)
    except Exception:
        return int(fallback_idx)


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def load_model(tag: str):
    if tag == "dinov2":
        from transformers import AutoModel
        return AutoModel.from_pretrained(
            "facebook/dinov2-base",
            trust_remote_code=True,
            use_safetensors=True,
        ).to(DEVICE).eval()
    if tag == "mae":
        from transformers import AutoModel
        return AutoModel.from_pretrained(
            "facebook/vit-mae-base",
            trust_remote_code=True,
            use_safetensors=True,
        ).to(DEVICE).eval()
    if tag == "clip":
        from transformers import CLIPVisionModel
        return CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch16",
            use_safetensors=True,
        ).to(DEVICE).eval()
    if tag == "supervised_vit":
        import timm
        return timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0).to(DEVICE).eval()
    raise ValueError(f"Unknown model tag: {tag}")


def forward_features(model, batch: torch.Tensor) -> torch.Tensor:
    out = model(batch)
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0, :]
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if isinstance(out, (tuple, list)):
        t = out[0]
        return t[:, 0, :] if len(t.shape) == 3 else t
    return out


def validate_features(feats: np.ndarray, tag: str) -> np.ndarray:
    if feats.ndim != 2:
        raise ValueError(f"{tag}: expected 2D features, got shape={feats.shape}")
    finite_mask = np.all(np.isfinite(feats), axis=1)
    norm_mask = np.linalg.norm(feats, axis=1) > 1e-8
    good = finite_mask & norm_mask
    if not np.all(good):
        bad_idx = np.where(~good)[0][:10].tolist()
        raise ValueError(f"{tag}: detected invalid feature vectors at indices {bad_idx} (showing up to 10).")
    return feats.astype(np.float32, copy=False)


def extract_features(
    model,
    img_paths: List[Path],
    batch_size: int = 32,
    preprocess_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    make_image: Optional[Callable[[Path, int], Image.Image]] = None,
    tag: str = "features",
) -> np.ndarray:
    if preprocess_fn is None:
        preprocess_fn = preprocess_imagenet
    if make_image is None:
        make_image = lambda path, idx: load_rgb_image(path)
    feats = []
    for start in tqdm(range(0, len(img_paths), batch_size), desc="extract", leave=False):
        batch_paths = img_paths[start: start + batch_size]
        batch_imgs = [make_image(path, start + offset) for offset, path in enumerate(batch_paths)]
        batch = torch.stack([preprocess_fn(im) for im in batch_imgs]).to(DEVICE)
        with torch.no_grad():
            f = forward_features(model, batch).detach().cpu().numpy()
        feats.append(validate_features(f, f"{tag}[{start}:{start + len(batch_paths)}]"))
    return np.concatenate(feats, axis=0)


def compute_centroids(x: np.ndarray, y: np.ndarray) -> Dict[int, np.ndarray]:
    cent = {}
    for cls in np.unique(y):
        v = x[y == cls].mean(axis=0, keepdims=True)
        cent[int(cls)] = l2norm(v)[0]
    return cent


def dnc_along(x_clean: np.ndarray, x_pert: np.ndarray, y: np.ndarray, centroids: Dict[int, np.ndarray]) -> np.ndarray:
    cf = l2norm(x_clean)
    pf = l2norm(x_pert)
    along = np.zeros((len(cf),), dtype=np.float32)
    for i in range(len(cf)):
        cls = int(y[i])
        own = centroids.get(cls)
        if own is None:
            along[i] = np.nan
            continue
        min_d = 1e9
        nearest = None
        for c2, oc in centroids.items():
            if int(c2) == cls:
                continue
            d = 1 - float(np.dot(own, oc))
            if d < min_d:
                min_d = d
                nearest = oc
        if nearest is None:
            along[i] = np.nan
            continue
        u = nearest - own
        un = float(np.linalg.norm(u))
        if un < 1e-12:
            along[i] = np.nan
            continue
        disp = pf[i] - cf[i]
        along[i] = abs(float(np.dot(disp, u / un)))
    return along


def margin_nearest_centroid(x: np.ndarray, y: np.ndarray, centroids: Dict[int, np.ndarray]) -> np.ndarray:
    xf = l2norm(x)
    m = np.zeros((len(xf),), dtype=np.float32)
    for i in range(len(xf)):
        cls = int(y[i])
        own = centroids.get(cls)
        if own is None:
            m[i] = np.nan
            continue
        d_own = 1 - float(np.dot(xf[i], own))
        min_other = 1e9
        for c2, oc in centroids.items():
            if c2 == cls:
                continue
            d = 1 - float(np.dot(xf[i], oc))
            if d < min_other:
                min_other = d
        m[i] = float(min_other - d_own)
    return m


def bootstrap_ci(x: np.ndarray, n: int, alpha: float, seed: int) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(int(n)):
        idx = rng.randint(0, len(x), size=len(x))
        means.append(float(np.mean(x[idx])))
    means = np.asarray(means, dtype=np.float32)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def cv_accuracy(x: np.ndarray, y: np.ndarray, seed: int) -> Dict[str, float]:
    x = l2norm(x)
    _, counts = np.unique(y, return_counts=True)
    n_splits = max(2, min(5, int(np.min(counts))))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(x, y):
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x[tr])
        x_te = scaler.transform(x[te])
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(x_tr, y[tr])
        accs.append(float(clf.score(x_te, y[te])))
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs))}


def fit_linear_probe(x_train: np.ndarray, y_train: np.ndarray, seed: int):
    x_train = l2norm(x_train)
    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(x_train_std, y_train)
    return scaler, clf


def probe_accuracy(scaler: StandardScaler, clf: LogisticRegression, x: np.ndarray, y: np.ndarray) -> float:
    x_std = scaler.transform(l2norm(x))
    return float(clf.score(x_std, y))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def cka(X, Y):
    """Linear CKA (Kornblith et al. 2019, ICML)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    hsic = float(np.linalg.norm(Xc.T @ Yc, 'fro') ** 2)
    var_x = float(np.linalg.norm(Xc.T @ Xc, 'fro') ** 2)
    var_y = float(np.linalg.norm(Yc.T @ Yc, 'fro') ** 2)
    denom = (var_x * var_y) ** 0.5
    return hsic / denom if denom > 1e-12 else float('nan')


def svcca(X, Y, n_components=20):
    """SVCCA (Raghu et al. 2017, NeurIPS): SVD reduce then CCA."""
    def _svd_reduce(Z):
        Zc = Z - Z.mean(axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
        k = min(n_components, len(s))
        return U[:, :k] * s[:k]
    Xr = _svd_reduce(X)
    Yr = _svd_reduce(Y)
    k = min(Xr.shape[1], Yr.shape[1], n_components)
    if k < 2:
        return float("nan")
    Xr, Yr = Xr[:, :k], Yr[:, :k]
    n = Xr.shape[0]
    eye = np.eye(k) * 1e-6
    C_xx = Xr.T @ Xr / (n - 1) + eye
    C_yy = Yr.T @ Yr / (n - 1) + eye
    C_xy = Xr.T @ Yr / (n - 1)
    Lx = np.linalg.cholesky(C_xx)
    Ly = np.linalg.cholesky(C_yy)
    C = np.linalg.solve(Lx, C_xy @ np.linalg.inv(Ly.T))
    _, S, _ = np.linalg.svd(C)
    return float(np.mean(S))


def resolve_official_coco_paths() -> Tuple[Path, Path, Path, Path]:
    train_dir = get_dataset_path("COCO")
    if not train_dir.exists():
        raise FileNotFoundError(f"COCO train dir not found: {train_dir}")
    coco_root = train_dir.parent
    val_dir = coco_root / "val2017"
    train_ann = get_coco_ann_path()
    val_ann = train_ann.with_name("instances_val2017.json")
    if not val_dir.exists():
        raise FileNotFoundError(f"COCO val dir not found: {val_dir}")
    if not train_ann.exists():
        raise FileNotFoundError(f"COCO train ann not found: {train_ann}")
    if not val_ann.exists():
        raise FileNotFoundError(f"COCO val ann not found: {val_ann}")
    return train_dir, val_dir, train_ann, val_ann


def build_random_split(proto: Protocol, seed: int) -> Tuple[List[Path], np.ndarray, List[Path], np.ndarray]:
    coco_dir = get_dataset_path("COCO")
    ann_path = get_coco_ann_path()
    if not coco_dir.exists():
        raise FileNotFoundError(f"COCO dir not found: {coco_dir}")
    if not ann_path.exists():
        raise FileNotFoundError(f"COCO ann not found: {ann_path}")
    img_to_cats, _ = load_coco_mapping(ann_path)
    all_paths = list_images(coco_dir, proto.n_total)
    paths, labels = build_coco_subset(all_paths, img_to_cats, proto, limit=proto.n_use)
    paths, labels = filter_by_min_class(paths, labels, proto.min_per_class)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    n_tr = int(len(idx) * proto.train_ratio)
    tr_idx = idx[:n_tr]
    te_idx = idx[n_tr:]
    return (
        [paths[int(i)] for i in tr_idx],
        labels[tr_idx],
        [paths[int(i)] for i in te_idx],
        labels[te_idx],
    )


def build_official_split(proto: Protocol, seed: int) -> Tuple[List[Path], np.ndarray, List[Path], np.ndarray]:
    train_dir, val_dir, train_ann, val_ann = resolve_official_coco_paths()
    train_map, _ = load_coco_mapping(train_ann)
    test_map, _ = load_coco_mapping(val_ann)
    train_paths_all = list_images(train_dir, proto.n_total_train)
    test_paths_all = list_images(val_dir, proto.n_total_test)
    train_paths, train_labels = build_coco_subset(train_paths_all, train_map, proto, limit=0)
    test_paths, test_labels = build_coco_subset(test_paths_all, test_map, proto, limit=0)
    train_paths, train_labels = filter_by_min_class(train_paths, train_labels, proto.min_per_class)
    test_paths, test_labels = filter_by_min_class(test_paths, test_labels, proto.min_per_class)
    train_paths, train_labels, test_paths, test_labels = filter_to_common_classes(
        train_paths, train_labels, test_paths, test_labels, proto.min_per_class
    )
    train_paths, train_labels = balanced_cap(train_paths, train_labels, proto.n_train_use, seed)
    test_paths, test_labels = balanced_cap(test_paths, test_labels, proto.n_test_use, seed + 1)
    train_paths, train_labels, test_paths, test_labels = filter_to_common_classes(
        train_paths, train_labels, test_paths, test_labels, 1
    )
    return train_paths, train_labels, test_paths, test_labels


def build_perturbation_grid(grid: str) -> Dict[str, Dict[str, List[float]]]:
    if grid == "minimal":
        return {
            "noise": {"param": "sigma", "values": [0.10]},
            "occlusion": {"param": "fraction", "values": [0.50]},
            "blur": {"param": "radius", "values": [3.0]},
        }
    return {
        "noise": {"param": "sigma", "values": [0.02, 0.05, 0.10, 0.20, 0.40]},
        "occlusion": {"param": "fraction", "values": [0.10, 0.25, 0.50, 0.75, 0.90]},
        "blur": {"param": "radius", "values": [1.0, 2.0, 3.0, 5.0, 7.0]},
    }


def make_perturb_loader(pname: str, val: float, seed: int) -> Callable[[Path, int], Image.Image]:
    def _loader(path: Path, fallback_idx: int) -> Image.Image:
        img = load_rgb_image(path)
        idx = image_seed_index(path, fallback_idx)
        if pname == "noise":
            return apply_noise(img, float(val), seed=seed, idx=idx)
        if pname == "occlusion":
            return apply_occlusion(img, float(val), seed=seed, idx=idx)
        if pname == "blur":
            return apply_blur(img, float(val))
        raise ValueError(pname)
    return _loader


def main():
    args = parse_args()
    proto = Protocol(
        split_mode=str(args.split_mode),
        n_total=int(args.n_total),
        n_use=int(args.n_use),
        n_total_train=int(args.n_total_train),
        n_total_test=int(args.n_total_test),
        n_train_use=int(args.n_train_use),
        n_test_use=int(args.n_test_use),
        train_ratio=float(args.train_ratio),
        min_per_class=int(args.min_per_class),
        enforce_single_label=not bool(args.no_single_label),
        seeds=tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""),
        bootstrap_n=int(args.bootstrap_n),
        grid=str(args.grid),
    )
    if args.single_label:
        proto.enforce_single_label = True

    set_random_seed(RANDOM_SEED)

    if proto.split_mode == "official":
        train_paths, y_tr, test_paths, y_te = build_official_split(proto, seed=RANDOM_SEED)
    else:
        train_paths, y_tr, test_paths, y_te = build_random_split(proto, seed=RANDOM_SEED)

    print(f"Device: {DEVICE}")
    print(f"Split mode: {proto.split_mode}")
    print(f"Train/Test subset: {len(train_paths)}/{len(test_paths)} samples")
    print(f"Train/Test classes: {len(np.unique(y_tr))}/{len(np.unique(y_te))}")
    print(f"Single label: {proto.enforce_single_label}")
    print(f"CLIP preprocessing: {args.clip_preprocess}")
    print(f"Batch size: {args.batch_size}")

    perturb_grid = build_perturbation_grid(proto.grid)
    models = [m.strip() for m in args.models.split(",") if m.strip() != ""]
    all_out = {
        "protocol": proto.__dict__,
        "device": str(DEVICE),
        "dataset": {
            "train_size": int(len(train_paths)),
            "test_size": int(len(test_paths)),
            "train_classes": int(len(np.unique(y_tr))),
            "test_classes": int(len(np.unique(y_te))),
        },
        "models": {},
    }

    for mtag in models:
        print(f"\n{'=' * 70}\nMODEL: {mtag}\n{'=' * 70}")
        model = load_model(mtag)
        prep = preprocess_clip if (mtag == "clip" and args.clip_preprocess == "clip") else preprocess_imagenet

        print("Extract clean features (train/test)...")
        f_tr = extract_features(
            model,
            train_paths,
            batch_size=int(args.batch_size),
            preprocess_fn=prep,
            tag=f"{mtag}:clean_train",
        )
        f_te = extract_features(
            model,
            test_paths,
            batch_size=int(args.batch_size),
            preprocess_fn=prep,
            tag=f"{mtag}:clean_test",
        )

        cent = compute_centroids(f_tr, y_tr)
        m_te = margin_nearest_centroid(f_te, y_te, cent)
        scaler, probe = fit_linear_probe(f_tr, y_tr, seed=RANDOM_SEED)
        probe_acc_clean = probe_accuracy(scaler, probe, f_te, y_te)
        cv = cv_accuracy(np.concatenate([f_tr, f_te], axis=0), np.concatenate([y_tr, y_te], axis=0), seed=RANDOM_SEED)

        out = {
            "cv_acc": cv,
            "clean_margin": {
                "mean": float(np.nanmean(m_te)),
                "std": float(np.nanstd(m_te)),
                "neg_pct": float(np.mean(m_te[np.isfinite(m_te)] < 0.0)),
                "ci95": bootstrap_ci(m_te, proto.bootstrap_n, proto.bootstrap_alpha, seed=RANDOM_SEED),
                "n_test": int(len(m_te)),
            },
            "linear_probe": {
                "clean_test_acc": probe_acc_clean,
            },
            "perturbations": {},
            "diagnostic_validation": {},
        }

        for pname, spec in perturb_grid.items():
            out["perturbations"][pname] = {"param": spec["param"], "values": []}
            for val in spec["values"]:
                per_seed_records = []
                for seed in proto.seeds:
                    f_pert_te = extract_features(
                        model,
                        test_paths,
                        batch_size=int(args.batch_size),
                        preprocess_fn=prep,
                        make_image=make_perturb_loader(pname, float(val), seed),
                        tag=f"{mtag}:{pname}={val}:seed={seed}",
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
                        "cos_dist_std": float(np.nanstd(cd)),
                        "dnc_along_mean": float(np.nanmean(along)),
                        "dnc_along_ci95": bootstrap_ci(along, proto.bootstrap_n, proto.bootstrap_alpha, seed=seed + 123),
                        "margin_mean": float(np.nanmean(margin_pert)),
                        "margin_drop_mean": float(np.nanmean(m_te) - np.nanmean(margin_pert)),
                        "probe_acc": probe_acc,
                        "probe_acc_drop": float(probe_acc_clean - probe_acc),
                        "cka": cka_val,
                        "svcca": svcca_val,
                    })

                along_means = np.array([r["dnc_along_mean"] for r in per_seed_records], dtype=np.float32)
                cd_means = np.array([r["cos_dist_mean"] for r in per_seed_records], dtype=np.float32)
                margin_means = np.array([r["margin_mean"] for r in per_seed_records], dtype=np.float32)
                margin_drops = np.array([r["margin_drop_mean"] for r in per_seed_records], dtype=np.float32)
                probe_accs = np.array([r["probe_acc"] for r in per_seed_records], dtype=np.float32)
                probe_drops = np.array([r["probe_acc_drop"] for r in per_seed_records], dtype=np.float32)
                cka_vals = np.array([r["cka"] for r in per_seed_records], dtype=np.float32)
                svcca_vals = np.array([r["svcca"] for r in per_seed_records], dtype=np.float32)

                out["perturbations"][pname]["values"].append({
                    spec["param"]: float(val),
                    "across_seed": {
                        "dnc_along_mean": float(np.mean(along_means)),
                        "dnc_along_std": float(np.std(along_means)),
                        "cos_dist_mean": float(np.mean(cd_means)),
                        "cos_dist_std": float(np.std(cd_means)),
                        "margin_mean": float(np.mean(margin_means)),
                        "margin_drop_mean": float(np.mean(margin_drops)),
                        "probe_acc_mean": float(np.mean(probe_accs)),
                        "probe_acc_std": float(np.std(probe_accs)),
                        "probe_acc_drop_mean": float(np.mean(probe_drops)),
                        "cka_mean": float(np.nanmean(cka_vals)),
                        "cka_std": float(np.nanstd(cka_vals)),
                        "svcca_mean": float(np.nanmean(svcca_vals)),
                        "svcca_std": float(np.nanstd(svcca_vals)),
                    },
                    "per_seed": per_seed_records,
                })

            family_vals = out["perturbations"][pname]["values"]
            out["diagnostic_validation"][pname] = {
                "cos_vs_probe_drop_corr": safe_corr(
                    np.array([v["across_seed"]["cos_dist_mean"] for v in family_vals], dtype=np.float32),
                    np.array([v["across_seed"]["probe_acc_drop_mean"] for v in family_vals], dtype=np.float32),
                ),
                "dnc_vs_probe_drop_corr": safe_corr(
                    np.array([v["across_seed"]["dnc_along_mean"] for v in family_vals], dtype=np.float32),
                    np.array([v["across_seed"]["probe_acc_drop_mean"] for v in family_vals], dtype=np.float32),
                ),
                "margin_drop_vs_probe_drop_corr": safe_corr(
                    np.array([v["across_seed"]["margin_drop_mean"] for v in family_vals], dtype=np.float32),
                    np.array([v["across_seed"]["probe_acc_drop_mean"] for v in family_vals], dtype=np.float32),
                ),
                "cka_vs_probe_drop_corr": safe_corr(
                    np.array([v["across_seed"]["cka_mean"] for v in family_vals], dtype=np.float32),
                    np.array([v["across_seed"]["probe_acc_drop_mean"] for v in family_vals], dtype=np.float32),
                ),
                "svcca_vs_probe_drop_corr": safe_corr(
                    np.array([v["across_seed"]["svcca_mean"] for v in family_vals], dtype=np.float32),
                    np.array([v["across_seed"]["probe_acc_drop_mean"] for v in family_vals], dtype=np.float32),
                ),
            }

        all_vals = []
        for pname in perturb_grid.keys():
            all_vals.extend(out["perturbations"][pname]["values"])
        out["diagnostic_validation"]["global"] = {
            "cos_vs_probe_drop_corr": safe_corr(
                np.array([v["across_seed"]["cos_dist_mean"] for v in all_vals], dtype=np.float32),
                np.array([v["across_seed"]["probe_acc_drop_mean"] for v in all_vals], dtype=np.float32),
            ),
            "dnc_vs_probe_drop_corr": safe_corr(
                np.array([v["across_seed"]["dnc_along_mean"] for v in all_vals], dtype=np.float32),
                np.array([v["across_seed"]["probe_acc_drop_mean"] for v in all_vals], dtype=np.float32),
            ),
            "margin_drop_vs_probe_drop_corr": safe_corr(
                np.array([v["across_seed"]["margin_drop_mean"] for v in all_vals], dtype=np.float32),
                np.array([v["across_seed"]["probe_acc_drop_mean"] for v in all_vals], dtype=np.float32),
            ),
            "cka_vs_probe_drop_corr": safe_corr(
                np.array([v["across_seed"]["cka_mean"] for v in all_vals], dtype=np.float32),
                np.array([v["across_seed"]["probe_acc_drop_mean"] for v in all_vals], dtype=np.float32),
            ),
            "svcca_vs_probe_drop_corr": safe_corr(
                np.array([v["across_seed"]["svcca_mean"] for v in all_vals], dtype=np.float32),
                np.array([v["across_seed"]["probe_acc_drop_mean"] for v in all_vals], dtype=np.float32),
            ),
        }

        all_out["models"][mtag] = out
        model.cpu()
        del model
        if torch.cuda.is_available():
            try:
                print("GPU max_memory_allocated (MB):", int(torch.cuda.max_memory_allocated() / (1024 * 1024)))
            except Exception:
                pass
            torch.cuda.empty_cache()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"exp_strict_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
