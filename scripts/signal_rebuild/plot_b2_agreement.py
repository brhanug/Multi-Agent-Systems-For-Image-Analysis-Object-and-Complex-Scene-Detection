#!/usr/bin/env python3
"""Figure for B2: how many independent systems agree with each pseudo-label.

Reads results/signal_rebuild/class_scores/teacher_label_votes.json; writes
latex_assets/b2/b2_agreement_by_class.png. SigLIP and LLaVA are the independent voters
(the detector trained on one of these label sets), so each pseudo-label has none, one or
both of them agreeing. One ordered blue ramp, none -> both, with the share labelled.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/data/brhanu/thesis_project")
R = json.loads((BASE / "results/signal_rebuild/class_scores/teacher_label_votes.json").read_text())
OUT = BASE / "latex_assets/b2"; OUT.mkdir(parents=True, exist_ok=True)
SURFACE, INK, INK2, MUTED, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
STEPS = ["#86b6ef", "#256abf", "#0d366b"]           # none, one, both -- one hue, ordered, validated
TITLES = {"final_dataset": "The set named for this question (4,879 pictures)",
          "v1_refresh": "The labels the detector was trained on (8,212 pictures)"}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "figure.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "legend.frameon": False})
sets = [(k, [e for e in v["per_class"] if "share_no_independent_system_agrees" in e]) for k, v in R["sets"].items()]
heights = [len(rows) for _, rows in sets]
fig, axes = plt.subplots(len(sets), 1, figsize=(7.2, 0.42 * sum(heights) + 2.2),
                         gridspec_kw={"height_ratios": heights, "hspace": 0.55})
for ax, (name, rows) in zip(axes, sets):
    ax.set_facecolor(SURFACE)
    rows = sorted(rows, key=lambda e: -e["pseudo_labels"])
    for i, e in enumerate(rows):
        none = e["share_no_independent_system_agrees"]; both = e["share_all_independent_systems_agree"]
        one = max(0.0, 1 - none - both)
        left = 0.0
        for share, col in zip((none, one, both), STEPS):
            ax.barh(i, share, left=left, height=0.62, color=col)
            left += share
        ax.text(1.01, i, f"{both:.0%} both", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{e['class']} ({e['pseudo_labels']:,})" for e in rows], color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(rows) - 0.4); ax.invert_yaxis()
    ax.set_title(TITLES.get(name, name), fontsize=10, color=INK, loc="left", pad=6)
    ax.tick_params(length=0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=MUTED)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in STEPS],
               labels=["neither agrees", "one agrees", "both agree"], loc="lower left",
               bbox_to_anchor=(0, 1.22), ncol=3, fontsize=8.5)
axes[-1].set_xlabel("Share of that class's pseudo-labels, by how many of the two independent systems\n"
                    "(SigLIP, LLaVA) also place the class in the picture. Number of pseudo-labels in brackets.",
                    color=INK2, fontsize=8.5, labelpad=10)
fig.savefig(OUT / "b2_agreement_by_class.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "b2_agreement_by_class.png")
