"""
项目全局配置文件
集中管理所有路径和常量，消除硬编码
"""

from pathlib import Path
import os

# 项目根目录（基于本文件位置自动推导）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 核心目录
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

# 输出子目录
EXP1_OUTPUT_DIR = OUTPUTS_DIR / "exp1_significance"
EXP2_OUTPUT_DIR = OUTPUTS_DIR / "exp2_multi_arch"
EXP3_OUTPUT_DIR = OUTPUTS_DIR / "exp3_downstream"
EXP4_OUTPUT_DIR = OUTPUTS_DIR / "exp4_dynamics"

# 确保输出目录存在
for dir_path in [OUTPUTS_DIR, EXP1_OUTPUT_DIR, EXP2_OUTPUT_DIR, 
                 EXP3_OUTPUT_DIR, EXP4_OUTPUT_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 数据集路径（建议在运行前通过环境变量覆盖）
# 默认值仅作占位说明，不对应真实公开数据位置
DATASET_PATHS = {
    "COCO": Path("path/to/COCO/train2017"),
    "ADE20K": Path("path/to/ADE20K/ADEChallengeData2016/images/training"),
    "PASCAL_VOC": Path("path/to/PASCAL_VOC/VOCdevkit/VOC2012/JPEGImages"),
}

# COCO标注路径
COCO_ANN_PATH = Path("path/to/COCO/annotations/instances_train2017.json")

# 随机种子配置
RANDOM_SEED = 42

# 模型配置
MODELS_CONFIG = {
    "dinov2": {"repo": "facebook/dinov2-base", "name": "dinov2"},
    "clip": {"repo": "openai/clip-vit-base-patch16", "name": "clip"},
    "mae": {"repo": "facebook/vit-mae-base", "name": "mae"},
    "dinov1": {"repo": "facebook/dino-vits8", "name": "dinov1"},
}

# 实验参数
DEFAULT_N_SAMPLES = 2000
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_REP = 20
PCA_N_COMPONENTS = 50


def get_dataset_path(dataset_name: str) -> Path:
    """获取数据集路径，支持环境变量覆盖"""
    env_var = f"DATASET_{dataset_name.upper()}_PATH"
    if env_var in os.environ:
        return Path(os.environ[env_var])
    return DATASET_PATHS.get(dataset_name, Path())

def get_coco_ann_path() -> Path:
    """获取 COCO 标注路径，支持环境变量覆盖。"""
    env_var = "COCO_ANN_PATH"
    if env_var in os.environ:
        return Path(os.environ[env_var])
    return COCO_ANN_PATH


def set_random_seed(seed: int = RANDOM_SEED):
    """设置全局随机种子"""
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

