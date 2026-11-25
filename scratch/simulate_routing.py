import pandas as pd
import numpy as np
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
GOLD_CSV = BASE / "human_baseline_gold_kit" / "gold_labels_human.csv"
SCORES_CSV = BASE / "results" / "multi_agent" / "agent_comparison_scores.csv"
COMPLEXITY_CSV = BASE / "results" / "multi_agent" / "scene_complexity_index.csv"

def normalize_id(name):
    p = str(name).replace("images/", "").replace("\\", "/")
    p = p.split("/")[-1].rsplit(".", 1)[0]
    parts = p.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 and parts[0].startswith("PPN") else p

def calculate_unified_uncertainty(df):
    agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    sigma_agents = df[agent_cols].std(axis=1).fillna(0)
    c_bar = df[agent_cols].mean(axis=1).fillna(0)
    u_score = (0.6 * sigma_agents) + (0.4 * (1 - c_bar))
    return u_score

def main():
    print("⏳ Running Curation Complexity-Aware Routing Simulation...")
    
    gold = pd.read_csv(GOLD_CSV)
    gold['cvat_id'] = gold.index
    gold_reviewed = gold[gold['cvat_id'] <= 800].copy()
    gold_reviewed["gold_has_scene"] = (gold_reviewed["n_scene_labels"] > 0).astype(int)
    gold_reviewed['image_id'] = gold_reviewed['image_id'].apply(normalize_id)

    agents = pd.read_csv(SCORES_CSV)
    agents['image_id'] = agents['image_id'].apply(normalize_id)
    agents = agents.drop_duplicates(subset=['image_id'])
    
    comp = pd.read_csv(COMPLEXITY_CSV)
    comp['image_id'] = comp['image_id'].apply(normalize_id)
    comp = comp.drop_duplicates(subset=['image_id'])

    joined = gold_reviewed.merge(agents, on='image_id', how='inner')
    joined = joined.merge(comp[['image_id', 'scene_complexity_index']], on='image_id', how='inner')
    
    joined["Unified_U"] = calculate_unified_uncertainty(joined)
    # Define error: prediction != gold
    joined["is_error"] = (joined["comparison_fusion_score"] >= 0.5).astype(int) != joined["gold_has_scene"]
    
    total_images = len(joined)
    total_errors = joined["is_error"].sum()
    
    # Define masks for SCI tiers
    low_mask = joined["scene_complexity_index"] < 0.3
    med_mask = (joined["scene_complexity_index"] >= 0.3) & (joined["scene_complexity_index"] <= 0.7)
    high_mask = joined["scene_complexity_index"] > 0.7
    
    # Low tier: Auto-approve (0% review)
    low_images = low_mask.sum()
    low_pct = low_images / total_images * 100
    low_errors = joined.loc[low_mask, "is_error"].sum()
    low_reviewed = 0
    low_errors_caught = 0
    
    # High tier: Mandatory review (100% review)
    high_images = high_mask.sum()
    high_pct = high_images / total_images * 100
    high_errors = joined.loc[high_mask, "is_error"].sum()
    high_reviewed = high_images
    high_errors_caught = high_errors
    
    # Medium tier: Conditional review (U >= 0.5)
    med_images = med_mask.sum()
    med_pct = med_images / total_images * 100
    med_errors = joined.loc[med_mask, "is_error"].sum()
    med_reviewed_mask = med_mask & (joined["Unified_U"] >= 0.5)
    med_reviewed = med_reviewed_mask.sum()
    med_errors_caught = joined.loc[med_reviewed_mask, "is_error"].sum()
    
    # Aggregates
    total_reviewed = low_reviewed + med_reviewed + high_reviewed
    total_reviewed_pct = total_reviewed / total_images * 100
    total_errors_caught = low_errors_caught + med_errors_caught + high_errors_caught
    overall_recall = total_errors_caught / total_errors * 100
    
    # Print results
    print("\n==============================================")
    print("Complexity-Aware Routing Simulation Results")
    print("==============================================")
    print(f"Low Complexity (SCI < 0.3):")
    print(f"  Images       : {low_images} ({low_pct:.2f}%)")
    print(f"  Errors       : {low_errors}")
    print(f"  Reviewed     : {low_reviewed} (0.0% tier budget)")
    print(f"  Caught Errors: {low_errors_caught} (0.0% recall)")
    print()
    print(f"Medium Complexity (0.3 <= SCI <= 0.7):")
    print(f"  Images       : {med_images} ({med_pct:.2f}%)")
    print(f"  Errors       : {med_errors}")
    print(f"  Reviewed     : {med_reviewed} ({med_reviewed/med_images*100:.2f}% tier budget)")
    print(f"  Caught Errors: {med_errors_caught} ({med_errors_caught/med_errors*100:.2f}% recall)")
    print()
    print(f"High Complexity (SCI > 0.7):")
    print(f"  Images       : {high_images} ({high_pct:.2f}%)")
    print(f"  Errors       : {high_errors}")
    print(f"  Reviewed     : {high_reviewed} (100.0% tier budget)")
    print(f"  Caught Errors: {high_errors_caught} (100.0% recall)")
    print()
    print(f"SYSTEM TOTAL:")
    print(f"  Total Reviewed     : {total_reviewed}/{total_images} ({total_reviewed_pct:.2f}%)")
    print(f"  Total Errors Caught: {total_errors_caught}/{total_errors} ({overall_recall:.2f}%)")
    print(f"  Budget Savings     : {100 - total_reviewed_pct:.2f}% workload reduction")
    print(f"  Efficiency Lift vs. Random: {overall_recall / total_reviewed_pct:.2f}x")
    
    # Save results to JSON
    report = {
        "low_complexity": {
            "n_images": int(low_images),
            "pct_images": float(low_pct),
            "errors": int(low_errors),
            "reviewed": int(low_reviewed),
            "recall": 0.0
        },
        "medium_complexity": {
            "n_images": int(med_images),
            "pct_images": float(med_pct),
            "errors": int(med_errors),
            "reviewed": int(med_reviewed),
            "recall": float(med_errors_caught / med_errors) if med_errors > 0 else 0.0
        },
        "high_complexity": {
            "n_images": int(high_images),
            "pct_images": float(high_pct),
            "errors": int(high_errors),
            "reviewed": int(high_reviewed),
            "recall": 1.0
        },
        "totals": {
            "n_images": int(total_images),
            "errors": int(total_errors),
            "reviewed": int(total_reviewed),
            "reviewed_pct": float(total_reviewed_pct),
            "errors_caught": int(total_errors_caught),
            "recall": float(overall_recall),
            "workload_reduction": float(100 - total_reviewed_pct),
            "efficiency_lift": float(overall_recall / total_reviewed_pct)
        }
    }
    
    out_json = BASE / "results" / "multi_agent" / "complexity_routing_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Simulation JSON saved: {out_json}")

if __name__ == '__main__':
    main()
