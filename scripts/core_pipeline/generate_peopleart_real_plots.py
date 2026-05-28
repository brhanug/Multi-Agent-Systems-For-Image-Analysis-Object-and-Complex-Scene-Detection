#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
BASE = Path("/data/brhanu/thesis_project")
SUMMARY_JSON = BASE / "results" / "multi_agent" / "peopleart_real_summary.json"
ASSETS_DIR = BASE / "latex_assets"

def main():
    if not SUMMARY_JSON.exists():
        print(f"❌ Summary JSON not found at {SUMMARY_JSON}")
        return

    with open(SUMMARY_JSON, "r") as f:
        data = json.load(f)
        
    bins = data["style_bin_summary"]
    
    # Extract data for plotting
    style_bins = [b["style_bin"].capitalize() for b in bins]
    f1_scores = [b["mean_f1"] for b in bins]
    saa_scores = [b["mean_saa_score"] for b in bins]
    uncertainties = [b["mean_uncertainty"] for b in bins]
    hitl_rates = [b["hitl_rate"] for b in bins]
    
    # Style configuration
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16
    })
    
    # 1. Plot F1 Score vs. Depiction Styles
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Blues_d", len(bins))
    
    # Sort by F1 score descending
    sorted_indices = sorted(range(len(f1_scores)), key=lambda k: f1_scores[k], reverse=True)
    sorted_styles = [style_bins[i] for i in sorted_indices]
    sorted_f1 = [f1_scores[i] for i in sorted_indices]
    
    bars = ax.bar(sorted_styles, sorted_f1, color=colors, edgecolor='grey', width=0.6)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    ax.set_title("YOLOv11 Bounding Box F1-Score across Depiction Styles (PeopleArt)", pad=15)
    ax.set_ylabel("F1-Score")
    ax.set_xlabel("Depiction Style Group")
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plot_f1_path = ASSETS_DIR / "peopleart_style_f1.png"
    plt.savefig(plot_f1_path, dpi=300)
    plt.close()
    print(f"📊 Saved {plot_f1_path}")
    
    # 2. Plot SAA Score vs. Uncertainty by Style Group
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Sort styles for consistent order
    sorted_styles_unc = [style_bins[i] for i in sorted_indices]
    sorted_saa = [saa_scores[i] for i in sorted_indices]
    sorted_unc = [uncertainties[i] for i in sorted_indices]
    
    x = range(len(sorted_styles_unc))
    width = 0.35
    
    rects1 = ax.bar([pos - width/2 for pos in x], sorted_saa, width, label='SAA Agreement Score', color='#1A56A0', edgecolor='grey')
    rects2 = ax.bar([pos + width/2 for pos in x], sorted_unc, width, label='SAA Uncertainty (Std)', color='#D9534F', edgecolor='grey')
    
    ax.set_title("SAA Agreement vs. Epistemic Uncertainty across Styles", pad=15)
    ax.set_ylabel("Metric Value")
    ax.set_xlabel("Depiction Style Group")
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_styles_unc)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)
    
    # Add values on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                    
    plt.tight_layout()
    plot_unc_path = ASSETS_DIR / "peopleart_calibration_real.png"
    plt.savefig(plot_unc_path, dpi=300)
    plt.close()
    print(f"📊 Saved {plot_unc_path}")

if __name__ == '__main__':
    main()
