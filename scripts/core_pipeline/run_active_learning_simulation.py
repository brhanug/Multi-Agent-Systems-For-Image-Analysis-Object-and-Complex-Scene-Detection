#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE = Path("/data/brhanu/thesis_project")
EXPERT_CSV = BASE / "human_baseline_gold_kit" / "gold_labels_human.csv"
SCORES_CSV = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT_DIR = BASE / "latex_assets"

def normalize_id(name):
    p = str(name).replace("images/", "").replace("\\", "/")
    p = p.split("/")[-1].rsplit(".", 1)[0]
    parts = p.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 and parts[0].startswith("PPN") else p

def main():
    print("🚀 Initiating ACTIVE LEARNING HITL SIMULATION...")
    
    # 1. Load expert gold labels and restrict to reviewed set (n=801)
    expert = pd.read_csv(EXPERT_CSV)
    expert['cvat_id'] = expert.index
    expert_reviewed = expert[expert['cvat_id'] <= 800].copy()
    expert_reviewed["gold_has_scene"] = (expert_reviewed["n_scene_labels"] > 0).astype(int)
    
    # 2. Load agent predictions
    agents = pd.read_csv(SCORES_CSV)
    agents['image_id'] = agents['image_id'].apply(normalize_id)
    agents = agents.drop_duplicates(subset=['image_id'])
    
    # 3. Join
    joined = expert_reviewed.merge(agents, on='image_id', how='inner')
    print(f"  Matched exactly {len(joined)} reviewed images with agent scores.")
    
    # 4. Define errors (Classification failure of monolithic model against human expert)
    gold_label = joined["gold_has_scene"].values
    joined["mono_error"] = (joined["monolithic_pipeline_agent"] >= 0.5).astype(int) != gold_label
    total_errors = joined["mono_error"].sum()
    print(f"  Total errors in uncleaned dataset: {total_errors} / {len(joined)}")
    
    # 5. Define selection metrics
    # Strategy A: SAA Unified Uncertainty (U)
    agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    joined["sigma_agents"] = joined[agent_cols].std(axis=1).fillna(0)
    joined["c_bar"] = joined[agent_cols].mean(axis=1).fillna(0)
    joined["unified_U"] = 0.6 * joined["sigma_agents"] + 0.4 * (1.0 - joined["c_bar"])
    
    # Strategy B: Monolithic Confidence Inverse
    joined["mono_conf_inv"] = 1.0 - joined["monolithic_pipeline_agent"].values
    
    # 6. Simulate cumulative error capture
    x_pct = np.linspace(0, 100, 101)
    
    # SAA Sort
    df_saa = joined.sort_values(by="unified_U", ascending=False).reset_index(drop=True)
    saa_captured = [0.0]
    for pct in x_pct[1:]:
        n = int(pct / 100 * len(df_saa))
        captured = df_saa.loc[:n-1, "mono_error"].sum()
        saa_captured.append(captured / total_errors)
        
    # Heuristic Sort
    df_heu = joined.sort_values(by="mono_conf_inv", ascending=False).reset_index(drop=True)
    heu_captured = [0.0]
    for pct in x_pct[1:]:
        n = int(pct / 100 * len(df_heu))
        captured = df_heu.loc[:n-1, "mono_error"].sum()
        heu_captured.append(captured / total_errors)
        
    # Random Sort (average of 100 runs)
    random_captured_runs = []
    for run in range(100):
        df_rand = joined.sample(frac=1.0, random_state=run).reset_index(drop=True)
        rand_captured = [0.0]
        for pct in x_pct[1:]:
            n = int(pct / 100 * len(df_rand))
            captured = df_rand.loc[:n-1, "mono_error"].sum()
            rand_captured.append(captured / total_errors)
        random_captured_runs.append(rand_captured)
    rand_captured_mean = np.mean(random_captured_runs, axis=0)
    
    # 7. Model downstream F1 score
    # F1(x) = F1_clean - (F1_clean - F1_noisy) * (1 - y(x))
    # where F1_clean = 0.9970 (AWLF), F1_noisy = 0.7870 (raw Monolithic)
    f1_clean = 0.997
    f1_noisy = 0.787
    f1_diff = f1_clean - f1_noisy
    
    f1_saa = [f1_noisy + f1_diff * y for y in saa_captured]
    f1_heu = [f1_noisy + f1_diff * y for y in heu_captured]
    f1_rand = [f1_noisy + f1_diff * y for y in rand_captured_mean]
    
    # 8. Plot results
    plt.figure(figsize=(10, 6))
    
    # Style customization for ultra-premium look
    plt.plot(x_pct, f1_saa, color="green", lw=3, label="SAA Uncertainty Triage (Our MAS)")
    plt.plot(x_pct, f1_heu, color="red", lw=2, linestyle="--", label="Model Confidence Inverse (Heuristic)")
    plt.plot(x_pct, f1_rand, color="gray", lw=1.5, linestyle=":", label="Random Sampling (Baseline)")
    
    plt.axvline(x=20.0, color="purple", linestyle="-.", alpha=0.7)
    plt.annotate("20% Auditing Limit", xy=(21, 0.81), color="purple", fontweight="bold")
    
    plt.title("Active Learning Simulation: Downstream Model F1 vs. Percentage of Images Audited", fontsize=13, pad=15)
    plt.xlabel("Percentage of Archive Audited (%)", fontsize=11)
    plt.ylabel("Downstream Model F1-Score", fontsize=11)
    plt.xlim(-2, 102)
    plt.ylim(0.77, 1.01)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right", fontsize=10.5)
    
    plt.tight_layout()
    out_path = OUT_DIR / "active_learning_efficiency.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Saved active learning simulation plot to {out_path}")
    
    # Print metrics
    print("\n" + "="*50)
    print("📈 HITL TRIAGE EFFICIENCY COMPARISON:")
    print("="*50)
    for pct in [10, 20, 30, 50]:
        idx = int(pct)
        print(f"At {pct}% audited:")
        print(f"  SAA Triage F1       : {f1_saa[idx]:.4f} (Captured {saa_captured[idx]*100:.1f}% of errors)")
        print(f"  Heuristic Triage F1 : {f1_heu[idx]:.4f} (Captured {heu_captured[idx]*100:.1f}% of errors)")
        print(f"  Random Baseline F1  : {f1_rand[idx]:.4f} (Captured {rand_captured_mean[idx]*100:.1f}% of errors)")
        print("-"*30)
    print("="*50)

if __name__ == "__main__":
    main()
