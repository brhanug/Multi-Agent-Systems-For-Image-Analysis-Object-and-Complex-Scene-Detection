#!/usr/bin/env python3
"""Figure for B1: F1 per class and system on the 300 human-annotated pictures.

Reads results/signal_rebuild/b1/b1_per_system.json; writes latex_assets/b1/b1_f1_by_class_and_system.png.
One sequential blue ramp for F1; every cell carries its value, so nothing depends on colour.
The always-yes column is the F1 a system would get by answering yes for every picture.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BASE = Path("/data/brhanu/thesis_project")
R = json.loads((BASE / "results/signal_rebuild/b1/b1_per_system.json").read_text())
OUT = BASE / "latex_assets/b1"; OUT.mkdir(parents=True, exist_ok=True)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SYSTEMS = [("detector", "Detector"), ("siglip", "SigLIP"), ("llava", "LLaVA"), ("kosmos", "Kosmos"), ("fusion", "Fusion")]
CLASSES = ["person", "animal", "text", "tree", "child", "furniture", "weapon", "building", "vehicle", "hat"]

cells = {(r["system"], r["class"]): r for r in R["per_class"]}
base = {r["class"]: r["always_yes_f1"] for r in R["per_class"]}
pos = {r["class"]: r["positives"] for r in R["per_class"]}
cols = SYSTEMS + [("always_yes", "Always yes")]

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
fig, ax = plt.subplots(figsize=(7.4, 5.6))
ax.set_facecolor(SURFACE)
for i, c in enumerate(CLASSES):
    for j, (key, _) in enumerate(cols):
        v = base[c] if key == "always_yes" else (cells[(key, c)]["cv"]["f1"] if (key, c) in cells else None)
        if v is None:
            ax.text(j, i, "—", ha="center", va="center", color=MUTED, fontsize=9); continue
        col = RAMP(v)
        ax.add_patch(plt.Rectangle((j - 0.5 + 0.03, i - 0.5 + 0.03), 0.94, 0.94, color=col, linewidth=0))
        lum = 0.2126 * col[0] + 0.7152 * col[1] + 0.0722 * col[2]
        degenerate = key != "always_yes" and "note" in cells.get((key, c), {})
        ax.text(j, i, f"{v:.2f}" + ("*" if degenerate else ""), ha="center", va="center", fontsize=9,
                color="white" if lum < 0.5 else INK, fontweight="bold" if key == "llava" else "normal")
ax.set_xlim(-0.5, len(cols) - 0.5); ax.set_ylim(len(CLASSES) - 0.5, -0.5)
ax.set_xticks(range(len(cols))); ax.set_xticklabels([n for _, n in cols], color=INK)
ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels([f"{c} ({pos[c]})" for c in CLASSES], color=INK)
ax.xaxis.tick_top(); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
mac = R["macro"]
ax.set_xlabel(f"F1 at a cut-off chosen by 5-fold cross-validation, {R['pictures']} pictures\n(pictures with the class in brackets).\n"
              f"Mean over the 8 shared classes: detector {mac['detector']['shared8_cv_f1']:.2f}, SigLIP {mac['siglip']['shared8_cv_f1']:.2f}, "
              f"LLaVA {mac['llava']['shared8_cv_f1']:.2f},\nfusion {mac['fusion']['shared8_cv_f1']:.2f}, always yes {mac['always_yes_baseline']['shared8_cv_f1']:.2f}.\n"
              f"* the cut-off says yes to every picture, so the value is the always-yes baseline.\n— the system cannot score this class.",
              color=INK2, fontsize=8.5, labelpad=10)
fig.savefig(OUT / "b1_f1_by_class_and_system.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "b1_f1_by_class_and_system.png")
