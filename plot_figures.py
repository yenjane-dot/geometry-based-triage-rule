import matplotlib.pyplot as plt
import numpy as np
import os
import json
from pathlib import Path

# ---------------------------------------------------------
# Public reproducibility repo paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Allow an explicit result JSON via RES_PATH; otherwise use the latest archived strict result.
default_result = None
strict_results = sorted(RESULTS_DIR.glob("exp_strict_*.json"), reverse=True)
if strict_results:
    default_result = strict_results[0]
RESULT_PATH = Path(os.environ.get("RES_PATH", default_result if default_result else ""))

if not RESULT_PATH or not RESULT_PATH.exists():
    raise FileNotFoundError(f"Result file not found: {RESULT_PATH}")

print(f"Loading results from: {RESULT_PATH}")
with open(RESULT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

models_data = data.get("models", {})

# ---------------------------------------------------------
# 模型定义及显示设置
# ---------------------------------------------------------
# 保证与结果中的键名匹配
MODEL_KEYS = ['mae', 'clip', 'dinov2', 'supervised_vit']
MODEL_LABELS = ['MAE', 'CLIP', 'DINOv2', 'Sup. ViT']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
MARKERS = ['o-', '^-', 's-', 'D-']

# 数据检查
for k in MODEL_KEYS:
    assert k in models_data, f"Model {k} missing from results JSON."

def save_fig(fig, base_name):
    """Export PNG and PDF figures inside the public reproducibility repo."""
    png_path = FIG_DIR / f"{base_name}.png"
    pdf_path = FIG_DIR / f"{base_name}.pdf"
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  -> Generated {base_name}.png / .pdf")

# ---------------------------------------------------------
# Figure 2: Decision Margins (Distribution of clean decision margins)
# ---------------------------------------------------------
print("Generating Figure 2: Decision Margins...")
margins = []
stds = []
for k in MODEL_KEYS:
    m_info = models_data[k]["clean_margin"]
    margins.append(m_info["mean"])
    stds.append(m_info["std"])

fig2 = plt.figure(figsize=(8, 5))
x = np.arange(len(MODEL_KEYS))
plt.bar(x, margins, yerr=stds, capsize=5, color=COLORS, alpha=0.8)
plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.xticks(x, MODEL_LABELS)
plt.ylabel('Decision Margin')
plt.title('Decision Margins across Models on COCO')
plt.tight_layout()
save_fig(fig2, "figure_2_decision_margins")
plt.close(fig2)

# ---------------------------------------------------------
# 抽取折线图公共函数：处理 Figure 3 (DNC) & Figure 4 (Cosine Shift)
# ---------------------------------------------------------
def plot_spectrum(metric_key, ylabel, title_prefix, base_name):
    print(f"Generating {base_name}: {title_prefix}...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    perturbations = ['noise', 'occlusion', 'blur']
    titles = ['Gaussian Noise', 'Occlusion', 'Gaussian Blur']
    param_keys = ['sigma', 'fraction', 'radius']
    xlabels = ['Sigma', 'Fraction Masked', 'Radius']

    for i, pert in enumerate(perturbations):
        ax = axes[i]
        
        # 异常检查
        assert pert in models_data[MODEL_KEYS[0]]["perturbations"], f"Missing perturbation {pert} in data."

        for j, k in enumerate(MODEL_KEYS):
            pert_data = models_data[k]["perturbations"][pert]["values"]
            x_vals = [item[param_keys[i]] for item in pert_data]
            y_vals = [item["across_seed"][metric_key] for item in pert_data]
            ax.plot(x_vals, y_vals, MARKERS[j], label=MODEL_LABELS[j], color=COLORS[j])

        ax.set_title(titles[i])
        ax.set_xlabel(xlabels[i])
        if i == 0:
            ax.set_ylabel(ylabel)
        ax.legend()

    plt.suptitle(f"{title_prefix} across Models", y=1.05)
    plt.tight_layout()
    save_fig(fig, base_name)
    plt.close(fig)

# ---------------------------------------------------------
# Figure 3: Decision-axis displacement (DNC along-axis)
# ---------------------------------------------------------
plot_spectrum("dnc_along_mean", "Along-axis Displacement", "Decision-axis Displacement", "figure_3_dnc")

# ---------------------------------------------------------
# Figure 4: Overall feature drift (Cosine Shift)
# ---------------------------------------------------------
plot_spectrum("cos_dist_mean", "Cosine Shift", "Overall Feature Drift", "figure_4_cosine_shift")

print("All figures generated successfully!")
