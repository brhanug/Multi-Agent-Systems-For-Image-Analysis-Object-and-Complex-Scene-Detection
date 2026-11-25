#!/usr/bin/env python3
"""The system against its baselines on the three tasks, as one figure for "The Thesis at a Glance".

Reads the figures of Table 6.12 from their result files (baselines.json, scene_classes.json,
ablation_corrected.json) and draws three panels: approval of scene presence at alpha = 0.05,
mean AUROC over the eight object classes, balanced accuracy over the six scene answers.
Writes latex_assets/summary_baselines.png and results/signal_rebuild/summary_figure.json.

    python run_summary_figure.py [--verify]
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
OUT, FIG = SR / "summary_figure.json", BASE / "latex_assets/summary_baselines.png"


def get(f, path):
    d = json.loads((SR / f).read_text())
    for k in path.split("|"):
        d = d[k]
    return float(d)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    B, SC, AB = "baselines.json", "scene_classes.json", "ablation_corrected.json"
    panels = {
        "Scene at all: approved at 5%": {
            "The system": 100 * get(B, "scene_presence|per_system|system|approved@0.05|mean"),
            "Old system": 100 * get("triage_weight.json", "per_w|0.6|approved@0.05|mean"),  # the deployed triage score (weight 0.6)
            "Linear probe": 100 * get(B, "scene_presence|per_system|linear probe|approved@0.05|mean"),
            "Describer alone": 100 * get(AB, "per_variant|describer alone|auto_approved|mean")},
        "Objects: mean AUROC": {
            "The system": get(B, "objects|per_system|system|mean_auroc|mean"),
            "Old system": get(B, "objects|per_system|old detector (deployed)|mean_auroc|mean"),
            "Linear probe": get(B, "objects|per_system|linear probe|mean_auroc|mean"),
            "Describer alone": get("object_classes.json", "results|describer alone|macro_auroc")},
        "Scene classes: bal. accuracy": {
            "The system": get(B, "scene_classes|per_system|system|balanced_accuracy|mean"),
            "Old system": get(B, "scene_classes|per_system|old system (deployed)|balanced_accuracy|mean"),
            "Linear probe": get(B, "scene_classes|per_system|linear probe|balanced_accuracy|mean"),
            "Describer alone": get(SC, "results|one rule|variants|describer alone|balanced_accuracy|mean")},
    }
    out = {k: {n: round(v, 4) for n, v in d.items()} for k, d in panels.items()}
    if a.verify:
        same = json.loads(OUT.read_text()) == out; print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    ink, accent, grey = "#1E2733", "#2D5F8B", "#B8BEC4"
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.0))
    for ax, (title, d) in zip(axs, panels.items()):
        names = list(d)[::-1]; vals = [d[n] for n in names]
        ax.barh(names, vals, color=[accent if n == "The system" else grey for n in names], height=0.6)
        pct = "approved" in title
        for y, v in enumerate(vals):
            ax.text(v, y, f" {v:.1f}%" if pct else f" {v:.3f}", va="center", fontsize=9, color=ink)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.set_xlim(0, 100 if pct else 1.0)
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=9)
    fig.tight_layout(); fig.savefig(FIG, dpi=220)
    OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
