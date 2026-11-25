import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Theme colors
DARK_BG   = "#0f0f1a"
PANEL_BG  = "#1a1a2e"
GRID_COL  = "#2a2a4a"
TEXT_COL  = "#e0e0e0"
ACCENT_COL = "#4FC3F7"

agents = ['VLM', 'Agreement', 'Scene', 'Object']
phi_score = [0.2580, 0.2209, 0.1528, 0.1501]
phi_roc = [0.0186, 0.0171, -0.0027, 0.0092]
phi_ece = [0.0630, 0.1186, 0.0613, 0.0988]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=DARK_BG)

def style_ax(ax, title, ylabel):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.set_title(title, color=TEXT_COL, fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=9)
    ax.grid(axis="y", color=GRID_COL, alpha=0.6, linewidth=0.6)
    ax.set_xticklabels(agents, fontsize=10, color=TEXT_COL)

# Panel A: Score
ax = axes[0]
bars = ax.bar(agents, phi_score, color=ACCENT_COL, alpha=0.88, width=0.55)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', color=TEXT_COL, fontsize=9, fontweight='bold')
style_ax(ax, "A — SAA Score Shapley Value\n(Grand Coalition v = 0.7818)", "Shapley Value (φ)")
ax.set_xticks(range(len(agents)))

# Panel B: ROC-AUC
ax = axes[1]
colors = [ACCENT_COL if v >= 0 else '#ef5350' for v in phi_roc]
bars = ax.bar(agents, phi_roc, color=colors, alpha=0.88, width=0.55)
for bar in bars:
    yval = bar.get_height()
    va_dir = 'bottom' if yval >= 0 else 'top'
    offset = 0.0005 if yval >= 0 else -0.0005
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + offset, f"{yval:+.4f}", ha='center', va=va_dir, color=TEXT_COL, fontsize=9, fontweight='bold')
style_ax(ax, "B — Uncertainty ROC-AUC Shapley Value\n(Grand Coalition Δ = +0.0423)", "Shapley Value (φ)")
ax.set_xticks(range(len(agents)))
ax.axhline(0, color=GRID_COL, linewidth=0.8)

# Panel C: ECE Reduction
ax = axes[2]
bars = ax.bar(agents, phi_ece, color=ACCENT_COL, alpha=0.88, width=0.55)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.002, f"{yval:.4f}", ha='center', va='bottom', color=TEXT_COL, fontsize=9, fontweight='bold')
style_ax(ax, "C — ECE Reduction Shapley Value\n(Grand Coalition Δ = +0.3417)", "Shapley Value (φ)")
ax.set_xticks(range(len(agents)))

plt.suptitle("Game-Theoretic Agent Contribution: Multi-Metric Shapley Attribution\n(n = 12,110 unconstrained images & n = 2,000 human-annotated images)", color=TEXT_COL, fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()

out_path = Path("latex_assets/shapley_attribution.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"✅ Shapley plot saved: {out_path}")
