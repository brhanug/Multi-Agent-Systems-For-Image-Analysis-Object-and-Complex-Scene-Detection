import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")
GOLD_CSV = BASE / "human_baseline_gold_kit" / "gold_labels_human.csv"
SCORES_CSV = BASE / "results" / "multi_agent" / "agent_comparison_scores.csv"

def normalize_id(name):
    p = str(name).replace("images/", "").replace("\\", "/")
    p = p.split("/")[-1].rsplit(".", 1)[0]
    parts = p.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 and parts[0].startswith("PPN") else p

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if i == n_bins - 1:
            in_bin = in_bin | (y_prob == bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    return ece

gold = pd.read_csv(GOLD_CSV)
gold['cvat_id'] = gold.index
gold_reviewed = gold[gold['cvat_id'] <= 800].copy()
gold_reviewed["gold_has_scene"] = (gold_reviewed["n_scene_labels"] > 0).astype(int)
gold_reviewed['image_id'] = gold_reviewed['image_id'].apply(normalize_id)

agents = pd.read_csv(SCORES_CSV)
agents['image_id'] = agents['image_id'].apply(normalize_id)
agents = agents.drop_duplicates(subset=['image_id'])

joined = gold_reviewed.merge(agents, on='image_id', how='inner')

y_scene = joined["gold_has_scene"].values
s_obj = joined["existing_pipeline_agent"].values
s_scene = joined["scene_agent"].values
s_vlm = joined["vlm_agent"].values
s_fused = joined["comparison_fusion_score"].values

joined["is_error"] = (s_fused >= 0.5).astype(int) != y_scene
y_err = joined["is_error"].astype(int).values

# SAA Disagreement (Ours): std of the 4 validation agents
agent_cols_4 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
u_saa = joined[agent_cols_4].std(axis=1).values

# Temp-Scaled Entropy
p_sce = s_scene
entropy = - (p_sce * np.log2(p_sce + 1e-12) + (1-p_sce) * np.log2(1-p_sce + 1e-12))
temp_entropy = entropy / 1.5

# MC Dropout (Single-Model SD): Raw YOLO Confidence Inverse
mc_dropout = 1.0 - s_obj

# Deep Ensemble Uncertainty: std of all 6 agents
agent_cols_6 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent", "restoration_agent", "document_agent"]
deep_ensemble = joined[agent_cols_6].std(axis=1).values

uncertainty_methods = {
    "Multi-Agent SAA (Ours)": u_saa,
    "Temperature-Scaled Entropy": temp_entropy,
    "Monte Carlo Dropout": mc_dropout,
    "Deep Ensemble Uncertainty": deep_ensemble
}

exp6_results = []
for name, u in uncertainty_methods.items():
    # Target should be y_err (error target)
    auc = roc_auc_score(y_err, u)
    
    # Normalize u to [0,1] for ECE
    u_norm = u / np.max(u) if np.max(u) > 0 else u
    ece = expected_calibration_error(y_err, u_norm)
    
    # Error recall at 20% audit budget
    threshold_idx = int(len(u) * 0.20)
    sorted_indices = np.argsort(u)[::-1]
    top_20_err = y_err[sorted_indices[:threshold_idx]]
    err_recall = np.sum(top_20_err) / np.sum(y_err) if np.sum(y_err) > 0 else 0.0
    
    if name != "Multi-Agent SAA (Ours)":
        stat, p_val = wilcoxon(u_saa, u)
    else:
        p_val = 1.0
        
    exp6_results.append({
        "Uncertainty Estimator": name,
        "ROC-AUC": round(auc, 4),
        "ECE": round(ece, 4),
        "Error Recall @ 20% Budget": round(err_recall, 4),
        "Wilcoxon p-value": f"{p_val:.6f}" if name != "Multi-Agent SAA (Ours)" else "Baseline"
    })

print(pd.DataFrame(exp6_results).to_string(index=False))
