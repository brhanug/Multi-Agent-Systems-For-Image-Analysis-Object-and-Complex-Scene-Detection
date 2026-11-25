#!/usr/bin/env python3
"""Figures for experiment 1: each Tier 1 model, and the fusions, on the 300 human-annotated pictures.

Reads results/signal_rebuild/per_model_performance_300.json (written by run_per_model_performance_300.py)
and writes four PNGs to latex_assets/per_model_300/. Every figure carries its values as text as well,
so none depends on colour alone.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE = Path("/data/brhanu/thesis_project")
R = json.loads((BASE / "results/signal_rebuild/per_model_performance_300.json").read_text())
OUT = BASE / "latex_assets/per_model_300"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3 (validated, light surface)
SEQ_LIGHT, SEQ_DARK = "#86b6ef", "#1c5cab"            # one hue, two shades for the seen/unseen dumbbell

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.edgecolor": AXIS, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False})


def frame(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(1)
    ax.grid(axis="x", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# 1. scene presence: ROC-AUC with bootstrap interval, single models vs fusions
sp = R["scene_presence"]
fig, ax = plt.subplots(figsize=(7.2, 0.38 * len(sp) + 1.0))
for k, r in enumerate(sp):
    y = len(sp) - 1 - k
    c = S2 if "fusion" in r["label"].lower() else S1
    ax.plot(r["ci95"], [y, y], color=c, linewidth=2, solid_capstyle="round")
    ax.scatter([r["roc_auc"]], [y], s=46, color=c, edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.text(r["ci95"][1] + 0.012, y, f"{r['roc_auc']:.3f}", va="center", fontsize=8, color=INK2)
ax.axvline(0.5, color=AXIS, linewidth=1, zorder=1)
ax.set_yticks(range(len(sp)))
ax.set_yticklabels([r["label"] for r in reversed(sp)])
ax.set_xlim(0.2, 1.0)
ax.set_ylim(-0.6, len(sp) - 0.4)
ax.set_xlabel(f"ROC-AUC for “the picture shows a scene”, 95% bootstrap interval; line at 0.5 = chance\n"
              f"({R['n_pictures']} pictures: {R['scene_yes']} with a scene, {R['scene_no']} without)")
frame(ax)
ax.legend(handles=[Line2D([], [], color=S1, marker="o", linewidth=2, label="single model"),
                   Line2D([], [], color=S2, marker="o", linewidth=2, label="fusion of the models")],
          loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
save(fig, "scene_presence_auroc.png")

# 2. object presence: F1 per class and model
op = sorted([r for r in R["object_presence"] if "note" not in r], key=lambda r: -r["pictures_with_class"])
series = [("detector (YOLO)", S1, "yolo_all"), ("LLaVA names the class", S2, "llava"),
          ("Kosmos reads ≥3 words (text only)", S3, "kosmos_three_or_more_words")]
bh = 0.27
fig, ax = plt.subplots(figsize=(7.2, 0.66 * len(op) + 1.0))
for k, r in enumerate(op):
    y0 = len(op) - 1 - k
    for j, (_, c, key) in enumerate(series):
        y = y0 + (1 - j) * bh
        m = r.get(key)
        if m is None:
            if key == "yolo_all":
                ax.text(0.005, y, "not one of the detector's classes", va="center", fontsize=7.5, color=MUTED)
            continue
        ax.barh(y, m["f1"], height=bh * 0.82, color=c)
        ax.text(m["f1"] + 0.01, y, f"{m['f1']:.2f}", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(op)))
ax.set_yticklabels([f"{r['class']} ({r['pictures_with_class']})" for r in reversed(op)])
ax.set_xlim(0, 1.0)
ax.set_ylim(-0.6, len(op) - 0.4)
ax.set_xlabel("F1 for “this class is in the picture” against the human boxes\n(number of pictures with the class in brackets)")
frame(ax)
ax.legend(handles=[Patch(color=c, label=n) for n, c, _ in series], loc="lower right", fontsize=8)
save(fig, "object_presence_f1.png")

# 3. detector boxes: AP50 on pictures it trained on vs pictures it did not
det = R.get("object_boxes_detector")
if det:
    order = [r["class"] for r in op]
    cls = [c for c in order if c in det["seen"]["per_class"] and c in det["unseen"]["per_class"]]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(cls) + 1.1))
    for k, c in enumerate(cls):
        y = len(cls) - 1 - k
        a, b = det["seen"]["per_class"][c]["ap50"], det["unseen"]["per_class"][c]["ap50"]
        ax.plot([a, b], [y, y], color=AXIS, linewidth=2, solid_capstyle="round", zorder=1)
        ax.scatter([a], [y], s=46, color=SEQ_LIGHT, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.scatter([b], [y], s=46, color=SEQ_DARK, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.text(max(a, b) + 0.015, y, f"{a:.2f} → {b:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(cls)))
    ax.set_yticklabels(list(reversed(cls)))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, len(cls) - 0.4)
    s, u = det["seen"], det["unseen"]
    ax.set_xlabel(f"Box AP50 per class. In the detector's training split: {s['pictures']} pictures, mAP50 {s['map50']:.3f}.\n"
                  f"Not in it: {u['pictures']} pictures, mAP50 {u['map50']:.3f}")
    frame(ax)
    ax.legend(handles=[Line2D([], [], color=SEQ_LIGHT, marker="o", linestyle="", markersize=7, label="picture was in training"),
                       Line2D([], [], color=SEQ_DARK, marker="o", linestyle="", markersize=7, label="picture was not")],
              loc="lower right", fontsize=8)
    save(fig, "detector_seen_unseen_ap50.png")

# 4. scene category: SigLIP recall per category
sc = R["scene_category"]
cats = sorted(sc["recall_per_category"], key=lambda c: -sc["support"][c])
fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(cats) + 1.0))
for k, c in enumerate(cats):
    y = len(cats) - 1 - k
    v = sc["recall_per_category"][c]
    ax.barh(y, v, height=0.5, color=S1)
    ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(cats)))
ax.set_yticklabels([f"{c} ({sc['support'][c]})" for c in reversed(cats)])
ax.set_xlim(0, 1.0)
ax.set_ylim(-0.6, len(cats) - 0.4)
ax.set_xlabel(f"Share of each category SigLIP gets right\n({sc['n']} pictures with a scene; accuracy {sc['accuracy']:.2f}, "
              f"balanced {sc['balanced_accuracy']:.2f};\nalways answering “drawing” would score {sc['always_guess_most_common_accuracy']:.2f})")
frame(ax)
save(fig, "scene_category_recall.png")
