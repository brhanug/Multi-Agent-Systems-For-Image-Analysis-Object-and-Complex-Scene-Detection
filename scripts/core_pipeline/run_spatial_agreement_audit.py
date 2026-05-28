#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
import scipy.stats as stats
import json

# Paths
BASE = Path("/data/brhanu/thesis_project")
USER_CSV = BASE / "human_spatial_audit/user_annotations_100.csv"
GOLD_CSV = BASE / "human_baseline_gold_kit/gold_labels_human.csv"
AGENT_CSV = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUTPUT_JSON = BASE / "results/spatial_agreement_report.json"

def get_basename(path_str):
    return str(path_str).split('/')[-1].rsplit('.', 1)[0]

def main():
    print("🧪 Running INTER-ANNOTATOR AGREEMENT & AMBIGUITY CORRELATION AUDIT...")
    
    # 1. Load Data
    user_df = pd.read_csv(USER_CSV)
    gold_df = pd.read_csv(GOLD_CSV)
    agent_df = pd.read_csv(AGENT_CSV)
    
    user_df['basename'] = user_df['filename'].apply(get_basename)
    gold_df['basename'] = gold_df['raw_name'].apply(get_basename)
    agent_df['basename'] = agent_df['image_id'].apply(lambda x: str(x).replace('/', '_'))
    
    # 2. Merge Data
    merged = user_df.merge(gold_df, on='basename', suffixes=('_user', '_gold'))
    merged = merged.merge(agent_df, on='basename')
    
    print(f"  Successfully aligned all {len(merged)} images across user, gold, and multi-agent scores.")
    
    # 3. Compute Cohen's Kappa for Scene Classification
    # Map user scene labels
    merged['scene_user'] = merged['scene'].str.strip().str.lower()
    
    # Determine gold primary scene
    scene_cols = ['label_teaching', 'label_family', 'label_playing', 'label_landscape', 'label_drawing']
    gold_scenes = []
    for idx, row in merged.iterrows():
        # find the column with value 1
        active = [c.replace('label_', '') for c in scene_cols if row[c] == 1]
        if active:
            gold_scenes.append(active[0])
        else:
            # default fallback to max or drawing
            gold_scenes.append('drawing')
    merged['scene_gold'] = gold_scenes
    
    # Category mappings to numeric for sklearn
    categories = ['teaching', 'family', 'playing', 'landscape', 'drawing']
    cat_to_id = {cat: i for i, cat in enumerate(categories)}
    
    y_user = merged['scene_user'].map(cat_to_id).fillna(cat_to_id['drawing']).astype(int)
    y_gold = merged['scene_gold'].map(cat_to_id).fillna(cat_to_id['drawing']).astype(int)
    
    # Multi-class Kappa
    multi_kappa = cohen_kappa_score(y_user, y_gold)
    
    # Category-specific binary Kappas
    binary_kappas = {}
    for cat in categories:
        user_bin = (merged['scene_user'] == cat).astype(int)
        gold_bin = (merged['scene_gold'] == cat).astype(int)
        kappa_val = cohen_kappa_score(user_bin, gold_bin)
        binary_kappas[cat] = float(kappa_val)
        
    mean_bin_kappa = float(np.mean(list(binary_kappas.values())))
    
    print("\n" + "="*50)
    print("👥 INTER-ANNOTATOR AGREEMENT (COHEN'S KAPPA)")
    print("="*50)
    print(f"  Multi-class Cohen's Kappa : {multi_kappa:.4f}")
    for cat, val in binary_kappas.items():
        print(f"    - {cat.capitalize():<12}: {val:.4f}")
    print(f"  Mean Category Kappa       : {mean_bin_kappa:.4f}")
    print("="*50)
    
    # 4. Compute Human Ambiguity vs AI Uncertainty Correlation
    # Map confidence to ambiguity (1-5 scale)
    # LS confidence values are '5 - Certain', '4 - Highly Likely', '3 - Unsure'
    def parse_ambiguity(conf_str):
        if pd.isna(conf_str):
            return 1.0
        try:
            num = int(str(conf_str).split(' ')[0])
            return float(6 - num) # Certain (5) -> 1, Unsure (3) -> 3
        except:
            return 1.0
            
    merged['human_ambiguity'] = merged['confidence'].apply(parse_ambiguity)
    
    # Calculate Unified AI Uncertainty (U)
    agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    sigma_agents = merged[agent_cols].std(axis=1).fillna(0)
    c_bar = merged[agent_cols].mean(axis=1).fillna(0)
    merged['ai_uncertainty'] = (0.6 * sigma_agents) + (0.4 * (1 - c_bar))
    
    # Calculate correlations
    pearson_r, pearson_p = stats.pearsonr(merged['human_ambiguity'], merged['ai_uncertainty'])
    spearman_rho, spearman_p = stats.spearmanr(merged['human_ambiguity'], merged['ai_uncertainty'])
    
    print("\n" + "="*50)
    print("🧠 HUMAN AMBIGUITY VS AI UNCERTAINTY CORRELATION")
    print("="*50)
    print(f"  Pearson Correlation (r)   : {pearson_r:.4f} (p = {pearson_p:.2e})")
    print(f"  Spearman Correlation (rho): {spearman_rho:.4f} (p = {spearman_p:.2e})")
    print("="*50)
    
    # Save Report
    report = {
        "multi_class_kappa": float(multi_kappa),
        "binary_kappas": binary_kappas,
        "mean_binary_kappa": mean_bin_kappa,
        "pearson_correlation": {
            "r": float(pearson_r),
            "p_value": float(pearson_p)
        },
        "spearman_correlation": {
            "rho": float(spearman_rho),
            "p_value": float(spearman_p)
        }
    }
    
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"\n✅ Agreement audit complete. Results saved to: {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
