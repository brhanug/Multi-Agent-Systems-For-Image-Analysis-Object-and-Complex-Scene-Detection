#!/usr/bin/env python3
import pandas as pd
import numpy as np

expert = pd.read_csv("human_baseline_gold_kit/gold_labels_human.csv")
expert['cvat_id'] = expert.index
expert_reviewed = expert[expert['cvat_id'] <= 800].copy()
expert_reviewed['gold_has_scene'] = (expert_reviewed['n_scene_labels'] > 0).astype(int)

def normalize_id(name):
    p = str(name).replace("images/", "").replace("\\", "/")
    p = p.split("/")[-1].rsplit(".", 1)[0]
    parts = p.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 and parts[0].startswith("PPN") else p

expert_reviewed['image_id_norm'] = expert_reviewed['image_id'].apply(normalize_id)
expert_reviewed = expert_reviewed.drop_duplicates(subset=['image_id_norm'])

sci_df = pd.read_csv("results/multi_agent/scene_complexity_index.csv")
sci_df['image_id_norm'] = sci_df['image_id'].apply(normalize_id)
sci_df = sci_df.drop_duplicates(subset=['image_id_norm'])

joined = expert_reviewed.merge(sci_df, on='image_id_norm', how='inner')
print("Matched rows:", len(joined))

joined["mono_error"] = (joined["monolithic_pipeline_agent"] >= 0.5).astype(int) != joined["gold_has_scene"]

# Calculate agent metrics for SAA Unified Uncertainty
agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
joined["sigma_agents"] = joined[agent_cols].std(axis=1).fillna(0)
joined["c_bar"] = joined[agent_cols].mean(axis=1).fillna(0)
joined["unified_U"] = 0.6 * joined["sigma_agents"] + 0.4 * (1.0 - joined["c_bar"])

for name, group in joined.groupby('sci_bin'):
    total_errors = group['mono_error'].sum()
    n_group = len(group)
    if total_errors == 0:
        print(f"Bin: {name:6s} | N: {n_group:3d} | No errors")
        continue
    k = max(1, int(n_group * 0.20))
    
    # SAA Sort
    df_saa = group.sort_values(by="unified_U", ascending=False).reset_index(drop=True)
    saa_found = df_saa.loc[:k-1, "mono_error"].sum()
    saa_recall = saa_found / total_errors
    
    # Random Sort (mean of 100 runs)
    rand_recalls = []
    for run in range(100):
        df_rand = group.sample(frac=1.0, random_state=run).reset_index(drop=True)
        rand_found = df_rand.loc[:k-1, "mono_error"].sum()
        rand_recalls.append(rand_found / total_errors)
    rand_recall = np.mean(rand_recalls)
    
    lift = saa_recall / rand_recall if rand_recall > 0 else 0.0
    print(f"Bin: {name:6s} | N: {n_group:3d} | Errors: {total_errors:3d} | Budget: 20% (k={k:2d}) | SAA Recall: {saa_recall:.4f} | Random Recall: {rand_recall:.4f} | Lift: {lift:.3f}x")
