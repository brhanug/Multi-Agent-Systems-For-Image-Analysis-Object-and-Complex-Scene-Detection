#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    print("🎨 Generating SOTA benchmarking comparison plots...")
    
    # Data
    systems = ['DCV-MAC (Ours)', 'Qwen2.5-VL-72B', 'InternVL3-78B']
    ece = [0.1423, 0.4913, 0.5055]
    roc_auc = [0.9724, 0.4949, 0.3745]
    recall_20 = [0.9561, 0.2667, 0.1333]
    
    colors = ['#4F46E5', '#F59E0B', '#F43F5E']
    
    # Setup figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.suptitle('SOTA Benchmarking Comparison: Multi-Agent System vs. Monolithic Models (n=100)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Subplot 1: ECE
    axes[0].bar(systems, ece, color=colors, width=0.5, edgecolor='black', alpha=0.9)
    axes[0].set_title('Expected Calibration Error (ECE)\n(Lower is Better)', fontsize=11, fontweight='semibold', pad=10)
    axes[0].set_ylim(0, 0.65)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(ece):
        axes[0].text(i, v + 0.015, f"{v:.4f}", ha='center', fontweight='bold', fontsize=10)
    
    # Subplot 2: ROC-AUC
    axes[1].bar(systems, roc_auc, color=colors, width=0.5, edgecolor='black', alpha=0.9)
    axes[1].set_title('Uncertainty ROC-AUC\n(Higher is Better)', fontsize=11, fontweight='semibold', pad=10)
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(roc_auc):
        axes[1].text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold', fontsize=10)
        
    # Subplot 3: Recall
    axes[2].bar(systems, recall_20, color=colors, width=0.5, edgecolor='black', alpha=0.9)
    axes[2].set_title('Error Recall @ 20% Budget\n(Higher is Better)', fontsize=11, fontweight='semibold', pad=10)
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(recall_20):
        axes[2].text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold', fontsize=10)
        
    # Styling and layout
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xticklabels(systems, fontweight='semibold')
        
    plt.tight_layout()
    
    # Save paths
    base_dir = Path("/data/brhanu/thesis_project")
    out_path_1 = base_dir / "latex_assets/sota_comparison_plot.png"
    out_path_2 = base_dir / "results/figures/sota_comparison_plot.png"
    
    # Ensure directories exist
    out_path_1.parent.mkdir(parents=True, exist_ok=True)
    out_path_2.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(out_path_1, dpi=300, bbox_inches='tight')
    plt.savefig(out_path_2, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Success! Plot saved successfully to:\n- {out_path_1}\n- {out_path_2}")

if __name__ == "__main__":
    main()
