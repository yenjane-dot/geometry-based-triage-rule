"""Update LaTeX tables with values from strict_result.json."""
import json, sys
from pathlib import Path

# Find the latest strict result
results_dir = Path(r"C:\Deepseektui\projects\stability_robustness_decoupling\results")
json_files = sorted(results_dir.glob("exp_strict_*.json"), reverse=True)
if not json_files:
    print("No result JSON found!")
    sys.exit(1)

result_path = json_files[0]
print(f"Using: {result_path.name}")

with open(result_path) as f:
    data = json.load(f)

# === Update baseline_comparison.tex ===
table_path = Path(r"C:\Deepseektui\projects\stability_robustness_decoupling\paper\tables\baseline_comparison.tex")
models_order = ["mae", "clip", "dinov2", "supervised_vit"]
display_names = {"mae": "MAE", "clip": "CLIP", "dinov2": "DINOv2", "supervised_vit": "Sup.\\ ViT"}

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
    row = f"{name} & {cka:.3f} & {svcca:.3f} & {cos:.3f} & {dnc:.3f} & {margin:.3f} \\\\"
    rows.append(row)

# Read existing table, replace placeholder rows
with open(table_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace data rows between midrule and bottomrule
import re
# Replace each model row
for i, (mtag, row) in enumerate(zip(models_order, rows)):
    old_row = rf"{display_names[mtag]}.*& X\.XXX.*\\\\"
    content = re.sub(old_row, row, content)

with open(table_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Tables updated!")
print("\nSummary:")
for row in rows:
    print(f"  {row}")

# Also print cross-dataset results if available
ade_paths = sorted(results_dir.glob("exp_ade20k_*.json"), reverse=True)
if ade_paths:
    with open(ade_paths[0]) as f:
        ade = json.load(f)
    print(f"\nADE20K validation:")
    for k, v in ade.get("diagnostic_validation", {}).items():
        print(f"  {k}: {v:.4f}")
