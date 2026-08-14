"""Export reproducibility summaries from archived result JSON files."""
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
results_dir = PROJECT_ROOT / "results"
outputs_dir = PROJECT_ROOT / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)

# Find the latest strict result
json_files = sorted(results_dir.glob("exp_strict_*.json"), reverse=True)
if not json_files:
    print("No result JSON found!")
    sys.exit(1)

result_path = json_files[0]
print(f"Using: {result_path.name}")

with open(result_path) as f:
    data = json.load(f)

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
    rows.append({
        "model": name,
        "cka_abs_corr": f"{cka:.3f}",
        "svcca_abs_corr": f"{svcca:.3f}",
        "cosine_abs_corr": f"{cos:.3f}",
        "dnc_abs_corr": f"{dnc:.3f}",
        "margin_abs_corr": f"{margin:.3f}",
    })

summary_csv = outputs_dir / "strict_diagnostic_summary.csv"
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["model", "cka_abs_corr", "svcca_abs_corr", "cosine_abs_corr", "dnc_abs_corr", "margin_abs_corr"],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Summary exported to: {summary_csv}")
print("\nSummary:")
for row in rows:
    print(
        "  {model}: CKA={cka_abs_corr}, SVCCA={svcca_abs_corr}, Cosine={cosine_abs_corr}, "
        "DNC={dnc_abs_corr}, Margin={margin_abs_corr}".format(**row)
    )

# Also print cross-dataset results if available
ade_paths = sorted(results_dir.glob("exp_ade20k_*.json"), reverse=True)
if ade_paths:
    with open(ade_paths[0]) as f:
        ade = json.load(f)
    ade_summary_txt = outputs_dir / "ade20k_validation_summary.txt"
    lines = ["ADE20K validation:"]
    for k, v in ade.get("diagnostic_validation", {}).items():
        lines.append(f"  {k}: {v:.4f}")
    with open(ade_summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print()
    for line in lines:
        print(line)
