#!/usr/bin/env python3
"""
exp24_scene_adaptive_weights.py
---------------------------------
Experiment 24: Scene-Type-Adaptive Weight Optimization

Performs grid search to optimize SAA weight allocations (alpha, beta, gamma)
separately for each of the 5 historical scene categories:
  - teaching
  - family
  - playing
  - landscape
  - drawing

Objective: Minimize classification error rate (SAA prediction != human gold label)
           relative to human expert ground truth.

Expected output:
  - results/multi_agent/exp24_scene_weights_results.json
  - results/figures/exp24_scene_weights_bars.png
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE         = Path("/data/brhanu/thesis_project")
BATCH1_CSV   = BASE / "human_spatial_audit/user_annotations_800.csv"
MANIFEST_CSV = BASE / "final_dataset/metadata/manifest.csv"
AGENTS_CSV   = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT_DIR      = BASE / "results/multi_agent"
FIG_DIR      = BASE / "results/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def extract_ppn_id(raw_name):
    name = str(raw_name).replace("images/", "").rsplit(".", 1)[0]
    parts = name.split("_")
    return f"{parts[0]}/{'_'.join(parts[1:])}"

def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total_ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        total_ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(total_ece / max(len(y_true), 1))

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("EXP 24: Scene-Type-Adaptive Weight Optimization")
    print("=" * 70)

    # ── Load and match datasets ──
    batch1 = pd.read_csv(BATCH1_CSV)
    batch1["has_scene"] = (batch1["n_scene_labels"] > 0).astype(int)
    batch1["ppn_id"]    = batch1["raw_name"].apply(extract_ppn_id)
    
    agents = pd.read_csv(AGENTS_CSV).drop_duplicates("image_id")
    manifest = pd.read_csv(MANIFEST_CSV)
    
    # Merge agent scores
    df = batch1.merge(agents, left_on="ppn_id", right_on="image_id", how="inner")
    
    # Merge predicted scene type from manifest
    df = df.merge(manifest[["image_id", "vqa_primary_scene"]], left_on="ppn_id", right_on="image_id", suffixes=("", "_manifest"))
    
    print(f"  Matched expert-reviewed records with manifest: {len(df)}")
    print("  Scene category distribution (predicted by VLM):")
    print(df["vqa_primary_scene"].value_counts().to_string())

    categories = ["drawing", "landscape", "playing", "family", "teaching"]
    
    # Baseline heuristic: alpha=0.6, beta=0.2, gamma=0.2
    alpha_base, beta_base, gamma_base = 0.6, 0.2, 0.2
    
    results = {}
    
    # ── Grid Search Optimization ──
    for cat in categories:
        df_cat = df[df["vqa_primary_scene"] == cat].copy()
        n_samples = len(df_cat)
        if n_samples == 0:
            print(f"  ⚠️  No samples for category: {cat}")
            continue
        
        # YOLO (object), SigLIP (scene), LLaVA (vlm)
        x_yolo  = df_cat["existing_pipeline_agent"].values
        x_scene = df_cat["scene_agent"].values
        x_vlm   = df_cat["vlm_agent"].values
        y_gold  = df_cat["has_scene"].values
        
        # Compute baseline performance
        base_fusion = alpha_base * x_yolo + beta_base * x_scene + gamma_base * x_vlm
        base_preds  = (base_fusion >= 0.5).astype(int)
        base_errors = (base_preds != y_gold).astype(int)
        base_error_rate = base_errors.mean()
        base_ece = ece(y_gold, base_fusion)
        
        # Search parameters
        best_error_rate = 1.0
        best_weights    = (alpha_base, beta_base, gamma_base)
        best_ece        = base_ece
        
        # Grid search with step 0.02
        for a in np.linspace(0.0, 1.0, 51):
            for b in np.linspace(0.0, 1.0 - a, 51):
                c = 1.0 - a - b
                
                fusion = a * x_yolo + b * x_scene + c * x_vlm
                preds  = (fusion >= 0.5).astype(int)
                errors = (preds != y_gold).astype(int)
                err_rate = errors.mean()
                
                if err_rate < best_error_rate:
                    best_error_rate = err_rate
                    best_weights    = (float(a), float(b), float(c))
                    best_ece        = ece(y_gold, fusion)
                elif err_rate == best_error_rate:
                    # Tie-breaker: choose lower ECE
                    curr_ece = ece(y_gold, fusion)
                    if curr_ece < best_ece:
                        best_weights = (float(a), float(b), float(c))
                        best_ece     = curr_ece
        
        results[cat] = {
            "n_samples": n_samples,
            "baseline": {
                "weights": [alpha_base, beta_base, gamma_base],
                "error_rate": round(float(base_error_rate), 4),
                "ece": round(float(base_ece), 4),
            },
            "optimized": {
                "weights": [round(w, 4) for w in best_weights],
                "error_rate": round(float(best_error_rate), 4),
                "ece": round(float(best_ece), 4),
            },
            "improvement": {
                "delta_error_rate": round(float(base_error_rate - best_error_rate), 4),
                "delta_ece": round(float(base_ece - best_ece), 4),
            }
        }
        
        print(f"\n  --- Category: {cat} (n={n_samples}) ---")
        print(f"    Baseline:  weights={results[cat]['baseline']['weights']} -> err={results[cat]['baseline']['error_rate']:.4f}, ECE={results[cat]['baseline']['ece']:.4f}")
        print(f"    Optimized: weights={results[cat]['optimized']['weights']} -> err={results[cat]['optimized']['error_rate']:.4f}, ECE={results[cat]['optimized']['ece']:.4f}")

    # ── Plot Visualizations ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Exp 24: Scene-Type-Adaptive Weight Optimization", fontsize=14, fontweight="bold")
    
    # 1. Plot weight distribution per scene category
    x_pos = np.arange(len(categories))
    width = 0.25
    
    yolo_w  = [results[cat]["optimized"]["weights"][0] for cat in categories]
    scene_w = [results[cat]["optimized"]["weights"][1] for cat in categories]
    vlm_w   = [results[cat]["optimized"]["weights"][2] for cat in categories]
    
    axes[0].bar(x_pos - width, yolo_w, width, label="YOLO (Object)", color="#1A56A0")
    axes[0].bar(x_pos, scene_w, width, label="SigLIP (Scene)", color="#2E7D32")
    axes[0].bar(x_pos + width, vlm_w, width, label="LLaVA (VLM)", color="#E65100")
    
    axes[0].set_ylabel("Optimal Weight Coefficient", fontsize=11)
    axes[0].set_title("Optimal Weight Distribution by Category", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(categories, fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].set_ylim(0, 1.1)

    # 2. Compare error rates (Baseline vs Optimized)
    base_errs = [results[cat]["baseline"]["error_rate"] for cat in categories]
    opt_errs  = [results[cat]["optimized"]["error_rate"] for cat in categories]
    
    axes[1].bar(x_pos - width/2, base_errs, width, label="Heuristic Baseline (0.6/0.2/0.2)", color="#B71C1C", alpha=0.8)
    axes[1].bar(x_pos + width/2, opt_errs, width, label="Scene-Adaptive Optimized", color="#2E7D32", alpha=0.8)
    
    axes[1].set_ylabel("Classification Error Rate", fontsize=11)
    axes[1].set_title("Error Rate Reduction Comparison", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(categories, fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].set_ylim(0, max(base_errs + opt_errs) * 1.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "exp24_scene_weights_bars.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Visualizations saved: {fig_path}")

    # ── Save Results JSON ──
    out_path = OUT_DIR / "exp24_scene_weights_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✅ Results saved: {out_path}")

if __name__ == "__main__":
    main()
