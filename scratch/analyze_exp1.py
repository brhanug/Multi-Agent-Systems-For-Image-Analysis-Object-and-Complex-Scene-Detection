import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")
GOLD_CSV = BASE / "human_baseline_gold_kit" / "gold_labels_human.csv"
SCORES_CSV = BASE / "results" / "multi_agent" / "agent_comparison_scores.csv"

def normalize_id(name):
    p = str(name).replace("images/", "").replace("\\", "/")
    p = p.split("/")[-1].rsplit(".", 1)[0]
    parts = p.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 and parts[0].startswith("PPN") else p

gold = pd.read_csv(GOLD_CSV)
gold['cvat_id'] = gold.index
gold_reviewed = gold[gold['cvat_id'] <= 800].copy()
gold_reviewed["gold_has_scene"] = (gold_reviewed["n_scene_labels"] > 0).astype(int)
gold_reviewed['image_id'] = gold_reviewed['image_id'].apply(normalize_id)

agents = pd.read_csv(SCORES_CSV)
agents['image_id'] = agents['image_id'].apply(normalize_id)
agents = agents.drop_duplicates(subset=['image_id'])

joined = gold_reviewed.merge(agents, on='image_id', how='inner')
print(f"Number of joined rows: {len(joined)}")

y_true = joined["gold_has_scene"].values
s_obj = joined["existing_pipeline_agent"].values
s_scene = joined["scene_agent"].values
s_vlm = joined["vlm_agent"].values

print("\n--- Basic Statistics ---")
print(f"y_true distribution (gold_has_scene): {np.bincount(y_true)}")
print(f"s_obj: min={s_obj.min():.4f}, max={s_obj.max():.4f}, mean={s_obj.mean():.4f}")
print(f"s_scene: min={s_scene.min():.4f}, max={s_scene.max():.4f}, mean={s_scene.mean():.4f}")
print(f"s_vlm: min={s_vlm.min():.4f}, max={s_vlm.max():.4f}, mean={s_vlm.mean():.4f}")

print("\n--- Individual ROC-AUCs (against y_true) ---")
print(f"s_obj ROC-AUC: {roc_auc_score(y_true, s_obj):.4f}")
print(f"s_scene ROC-AUC: {roc_auc_score(y_true, s_scene):.4f}")
print(f"s_vlm ROC-AUC: {roc_auc_score(y_true, s_vlm):.4f}")

print("\n--- Correlations ---")
print(f"Correlation s_obj vs y_true: {np.corrcoef(s_obj, y_true)[0,1]:.4f}")
print(f"Correlation s_scene vs y_true: {np.corrcoef(s_scene, y_true)[0,1]:.4f}")
print(f"Correlation s_vlm vs y_true: {np.corrcoef(s_vlm, y_true)[0,1]:.4f}")
