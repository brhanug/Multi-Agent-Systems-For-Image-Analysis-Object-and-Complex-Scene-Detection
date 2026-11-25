import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")
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
    print(f"Total images: {total_images}, Total errors: {total_errors}")

    # Bins
    low_mask = joined["scene_complexity_index"] < 0.3
    med_mask = (joined["scene_complexity_index"] >= 0.3) & (joined["scene_complexity_index"] <= 0.7)
    high_mask = joined["scene_complexity_index"] > 0.7

    print(f"\nLow Complexity (SCI < 0.3): {low_mask.sum()} images ({low_mask.sum()/total_images*100:.1f}%), errors: {joined.loc[low_mask, 'is_error'].sum()}")
    print(f"Medium Complexity (0.3-0.7): {med_mask.sum()} images ({med_mask.sum()/total_images*100:.1f}%), errors: {joined.loc[med_mask, 'is_error'].sum()}")
    print(f"High Complexity (SCI > 0.7): {high_mask.sum()} images ({high_mask.sum()/total_images*100:.1f}%), errors: {joined.loc[high_mask, 'is_error'].sum()}")

    # Try different thresholds of Unified_U on Medium complexity images
    print("\nMedium Tier routing by Unified_U threshold:")
    for thresh in np.linspace(0.1, 0.9, 9):
        reviewed_med = med_mask & (joined["Unified_U"] >= thresh)
        med_errors_caught = joined.loc[reviewed_med, "is_error"].sum()
        total_med_errors = joined.loc[med_mask, "is_error"].sum()
        med_recall = med_errors_caught / total_med_errors if total_med_errors > 0 else 0
        med_budget = reviewed_med.sum() / med_mask.sum()
        print(f"  Thresh {thresh:.1f}: Review budget: {med_budget*100:.1f}% ({reviewed_med.sum()} images), Caught errors: {med_errors_caught}/{total_med_errors} ({med_recall*100:.1f}%)")

if __name__ == "__main__":
    main()
