import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from pathlib import Path

# ---------------------------------------------------------
# Public reproducibility package paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def resolve_result_path() -> Path:
    env_path = os.environ.get("RES_PATH")
    if env_path:
        return Path(env_path)

    candidates = sorted((PROJECT_ROOT / "results").glob("exp_strict_*.json"), reverse=True)
    if candidates:
        return candidates[0]

    raise FileNotFoundError("No strict-protocol result JSON found under results/.")


RESULT_PATH = resolve_result_path()

# 期刊风格全局设置：白底不透明、无透明、可读字号
plt.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "dejavusans",
})

if not RESULT_PATH.exists():
    raise FileNotFoundError(f"Result file not found: {RESULT_PATH}")

print(f"Loading results from: {RESULT_PATH}")
with open(RESULT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
models_data = data.get("models", {})

MODEL_KEYS = ['mae', 'clip', 'dinov2', 'supervised_vit']
MODEL_LABELS = ['MAE', 'CLIP', 'DINOv2', 'Sup. ViT']
# Okabe-Ito 色盲安全色 (blue, orange, bluish green, vermillion)
COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00']
MARKERS = ['o', '^', 's', 'D']

for k in MODEL_KEYS:
    assert k in models_data, f"Model {k} missing from results JSON."


def save_fig(fig, base_name):
    """PNG(300DPI) + PDF(矢量)，白底不透明。"""
    png_path = FIG_DIR / f"{base_name}.png"
    pdf_path = FIG_DIR / f"{base_name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"  -> Generated {base_name}.png / .pdf (300dpi)")


# ---------------------------------------------------------
# Figure 1: 原理图（去坐标刻度、去图内标题、Okabe-Ito 配色）
# ---------------------------------------------------------
def gen_schematic():
    print("Generating Figure 1 (schematic)...")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.set_axis_off()
    ax.set_xlim(-0.15, 1.25)
    ax.set_ylim(-0.1, 1.05)

    # 决策轴 u（虚线）
    ax.plot([-0.05, 1.10], [0.12, 0.80], ls='--', color='#333333', lw=2)
    ax.text(0.02, 0.72, 'decision axis $u$', fontsize=10, color='#333333')

    # 两个质心
    muk = np.array([0.10, 0.22]); mukp = np.array([1.00, 0.74])
    ax.plot(*muk, 'o', ms=11, color='#0072B2', mec='black', mew=0.6)
    ax.plot(*mukp, 'o', ms=11, color='#D55E00', mec='black', mew=0.6)
    ax.text(muk[0]-0.06, muk[1]-0.10, r'$\mu_k$', fontsize=12, color='#0072B2')
    ax.text(mukp[0]+0.05, mukp[1]-0.02, r"$\mu_{k'}$", fontsize=12, color='#D55E00')

    # 干净特征 f_c 位于轴上；扰动特征 f_p 偏离轴
    fc = np.array([0.38, 0.38]); fp = np.array([0.62, 0.55])
    ax.plot(*fc, 'o', ms=10, color='#009E73', mec='black', mew=0.6)
    ax.plot(*fp, 'o', ms=10, color='#E69F00', mec='black', mew=0.6)
    ax.text(fc[0]-0.14, fc[1]-0.04, r'$f_c$', fontsize=12, color='#009E73')
    ax.text(fp[0]+0.04, fp[1]+0.06, r'$f_p$', fontsize=12, color='#E69F00')

    # 位移 δ 及其沿轴分量 δ_∥
    ax.annotate('', xy=fp, xytext=fc, arrowprops=dict(arrowstyle='->', color='#555555', lw=1.4))
    ax.text((fc[0]+fp[0])/2-0.05, (fc[1]+fp[1])/2+0.06, r'$\delta$', fontsize=12, color='#555555')
    # 沿轴分量：从 fc 沿 u 方向投影点
    u = (mukp - muk) / np.linalg.norm(mukp - muk)
    proj = fc + np.dot(fp - fc, u) * u
    ax.plot([fc[0], proj[0]], [fc[1], proj[1]], ls=':', color='#0072B2', lw=1.4)
    ax.text(proj[0]+0.03, proj[1]-0.10, r'$\delta_{\parallel}$', fontsize=12, color='#0072B2')

    save_fig(fig, "Fig1")
    plt.close(fig)


# ---------------------------------------------------------
# Figure 1: 原理图
# ---------------------------------------------------------
gen_schematic()

# ---------------------------------------------------------
# Figure 2: Decision Margins
# ---------------------------------------------------------
print("Generating Figure 2: Decision Margins...")
margins, stds = [], []
for k in MODEL_KEYS:
    m_info = models_data[k]["clean_margin"]
    margins.append(m_info["mean"]); stds.append(m_info["std"])

fig2 = plt.figure(figsize=(6.5, 4.5))
x = np.arange(len(MODEL_KEYS))
plt.bar(x, margins, yerr=stds, capsize=4, color=COLORS, edgecolor='black', linewidth=0.5, width=0.62)
plt.axhline(0, color='black', linewidth=0.9, linestyle='--')
plt.xticks(x, MODEL_LABELS)
plt.ylabel('Decision margin')
plt.ylim(-0.05, None)
save_fig(fig2, "Fig2")
plt.close(fig2)

# ---------------------------------------------------------
# Figure 3 & 4: 谱图（去 suptitle、去 axes title 冗余）
# ---------------------------------------------------------
def plot_spectrum(metric_key, ylabel, base_name):
    print(f"Generating {base_name}...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    perturbations = ['noise', 'occlusion', 'blur']
    titles = ['Gaussian Noise', 'Occlusion', 'Gaussian Blur']
    param_keys = ['sigma', 'fraction', 'radius']
    xlabels = [r'$\sigma$', 'Masked fraction', 'Radius']

    for i, pert in enumerate(perturbations):
        ax = axes[i]
        for j, k in enumerate(MODEL_KEYS):
            pert_data = models_data[k]["perturbations"][pert]["values"]
            x_vals = [item[param_keys[i]] for item in pert_data]
            y_vals = [item["across_seed"][metric_key] for item in pert_data]
            if i == 0:
                ax.plot(x_vals, y_vals, MARKERS[j]+'-', label=MODEL_LABELS[j], color=COLORS[j], ms=4)
            else:
                ax.plot(x_vals, y_vals, MARKERS[j]+'-', color=COLORS[j], ms=4)
        ax.set_xlabel(xlabels[i])
        ax.set_title(titles[i], fontsize=10)
        if i == 0:
            ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(frameon=False)
    save_fig(fig, base_name)
    plt.close(fig)

plot_spectrum("dnc_along_mean", "Along-axis displacement", "Fig3")
plot_spectrum("cos_dist_mean", "Cosine shift", "Fig4")

print("All figures generated successfully!")
