"""Export a plain-text summary from the latest strict-protocol result."""
import json, sys
from pathlib import Path

# Find the latest strict result
PROJECT_ROOT = Path(__file__).resolve().parent.parent
results_dir = PROJECT_ROOT / "results"
json_files = sorted(results_dir.glob("exp_strict_*.json"), reverse=True)
if not json_files:
    print("No result JSON found!")
    sys.exit(1)

result_path = json_files[0]
print(f"Using: {result_path.name}")

with open(result_path) as f:
    data = json.load(f)

# === Build a portable markdown summary ===
models_order = ["mae", "clip", "dinov2", "supervised_vit"]
display_names = {"mae": "MAE", "clip": "CLIP", "dinov2": "DINOv2", "supervised_vit": "Sup. ViT"}

rows = []
for mtag in models_order:
    if mtag not in data["models"]:
        continue
    g = data["models"][mtag]["diagnostic_validation"]["global"]
    name = display_names[mtag]
    cka = abs(g.get("cka_vs_probe_drop_corr", float("nan")))
    svcca = abs(g.get("svcca_vs_probe_drop_corr", float("nan")))
    cos = abs(g.get("cos_vs_probe_drop_corr", float("nan")))
    dnc = abs(g.get("dnc_vs_probe_drop_corr", float("nan")))
    margin = abs(g.get("margin_drop_vs_probe_drop_corr", float("nan")))
    row = (name, cka, svcca, cos, dnc, margin)
    rows.append(row)

out_dir = PROJECT_ROOT / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)
summary_path = out_dir / "strict_protocol_summary.md"

lines = [
    "# Strict-Protocol Summary",
    "",
    f"- source: `{result_path.name}`",
    "",
    "| Model | |CKA| | |SVCCA| | |Cosine shift| | |DNC| | |Margin drop| |",
    "|---|---:|---:|---:|---:|---:|",
]

for name, cka, svcca, cos, dnc, margin in rows:
    lines.append(f"| {name} | {cka:.3f} | {svcca:.3f} | {cos:.3f} | {dnc:.3f} | {margin:.3f} |")

ade_paths = sorted(results_dir.glob("exp_ade20k_*.json"), reverse=True)
if ade_paths:
    with open(ade_paths[0]) as f:
        ade = json.load(f)
    lines.extend(["", f"## ADE20K validation (`{ade_paths[0].name}`)", ""])
    for k, v in ade.get("diagnostic_validation", {}).items():
        lines.append(f"- {k}: {v:.4f}")

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("Summary exported!")
print(f"Markdown: {summary_path}")
print("\nSummary:")
for name, cka, svcca, cos, dnc, margin in rows:
    print(f"  {name}: CKA={cka:.3f}, SVCCA={svcca:.3f}, Cos={cos:.3f}, DNC={dnc:.3f}, Margin={margin:.3f}")
