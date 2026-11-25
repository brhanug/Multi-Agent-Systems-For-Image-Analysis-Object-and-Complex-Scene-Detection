#!/usr/bin/env python3
"""Figure: share auto-approved against the error actually delivered, old against corrected pipeline.

Reads matched_risk_comparison.json (run_matched_risk_comparison.py): one point per risk target, the
mean over 200 book-disjoint redraws of approval and realised risk. Writes
latex_assets/matched_risk_curve.png.

    python plot_matched_risk.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/data/brhanu/thesis_project")
D = json.loads((BASE / "results/signal_rebuild/matched_risk_comparison.json").read_text())["promised_curve"]
OUT = BASE / "latex_assets/matched_risk_curve.png"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
SERIES = [("corrected + absence (confidence)", "Corrected design", "#2a78d6", "o", (6, 0)),
          ("old (disagreement)", "Deployed design", "#eb6834", "s", None),
          ("old (confidence)", "Deployed design, confidence triage", "#1baf7a", "^", (6, 0))]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300)
for key, label, col, mk, off in SERIES:
    pts = sorted((100 * c["realised_risk"], 100 * c["approved"]) for c in D[key])
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=6, mec="white", mew=1.2, label=label, zorder=3)
    if off is None:                               # below the orange segment, clear of the green line
        ax.text(4.6, 60.5, label.replace(", ", ",\n"), ha="center", va="center", fontsize=8.5, color=INK)
    else:
        ax.annotate(label, (xs[-1], ys[-1]), xytext=off, textcoords="offset points", va="center", fontsize=8.5, color=INK)
for x in (5, 10):
    ax.axvline(x, color=GRID, lw=1, zorder=1)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_xlim(0, 18); ax.set_xticks(range(0, 19, 2)); ax.set_ylim(20, 100)
ax.set_xlabel("Error actually delivered (% of all pictures wrong and auto-approved)")
ax.set_ylabel("Pictures auto-approved (%)")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
fig.tight_layout(); fig.savefig(OUT, facecolor="white"); print("wrote", OUT)
