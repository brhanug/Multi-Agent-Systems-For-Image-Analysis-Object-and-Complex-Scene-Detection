import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
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

y_err = ((s_fused >= 0.5).astype(int) != y_scene).astype(int)
y_correct = 1 - y_err  # 1 = correct, 0 = error

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 5-Fold grid search to find optimal weights for predicting correctness
best_weights_folds = []
for fold, (train_idx, val_idx) in enumerate(kf.split(joined)):
    y_train, y_val = y_correct[train_idx], y_correct[val_idx]
    s_obj_tr, s_obj_va = s_obj[train_idx], s_obj[val_idx]
    s_scene_tr, s_scene_va = s_scene[train_idx], s_scene[val_idx]
    s_vlm_tr, s_vlm_va = s_vlm[train_idx], s_vlm[val_idx]
    
    best_auc_fold = 0.0
    best_w_fold = [0.60, 0.20, 0.20]
    
    for wo in np.linspace(0, 1, 21):
        for ws in np.linspace(0, 1 - wo, 21):
            wv = 1.0 - wo - ws
            if wv < -1e-9:
                continue
            s_saa_tr = wo * s_obj_tr + ws * s_scene_tr + wv * s_vlm_tr
            auc = roc_auc_score(y_train, s_saa_tr) if len(np.unique(y_train)) > 1 else 0.5
            if auc > best_auc_fold:
                best_auc_fold = auc
                best_w_fold = [wo, ws, wv]
    best_weights_folds.append(best_w_fold)

mean_w = np.mean(best_weights_folds, axis=0)
print(f"Optimized weights over 5-Fold CV (correctness target): wo={mean_w[0]:.3f}, ws={mean_w[1]:.3f}, wv={mean_w[2]:.3f}")

clf_saa = LogisticRegression(C=1.0, random_state=42)
clf_saa.fit(np.column_stack((s_obj, s_scene, s_vlm)), y_correct)
coef = clf_saa.coef_[0]
coef_norm = coef / np.sum(coef) if np.sum(coef) > 0 else np.array([0.333, 0.333, 0.333])
print(f"Logistic coefficients: {coef}, norm: {coef_norm}")

saa_strategies = {
    "Hand-defined [0.60, 0.20, 0.20]": [0.60, 0.20, 0.20],
    "Equal weights [0.333, 0.333, 0.333]": [0.333, 0.333, 0.333],
    f"Logistic-learned [{coef_norm[0]:.3f}, {coef_norm[1]:.3f}, {coef_norm[2]:.3f}]": coef_norm.tolist(),
    f"5-Fold CV Optimized [{mean_w[0]:.3f}, {mean_w[1]:.3f}, {mean_w[2]:.3f}]": mean_w.tolist()
}

exp1_results = []
for name, w in saa_strategies.items():
    s_saa = w[0] * s_obj + w[1] * s_scene + w[2] * s_vlm
    auc = roc_auc_score(y_correct, s_saa)
    ece = expected_calibration_error(y_correct, s_saa)
    brier = np.mean((s_saa - y_correct) ** 2)
    
    threshold_idx = int(len(s_saa) * 0.10)
    sorted_indices = np.argsort(s_saa)[::-1]
    top_10_correct = y_correct[sorted_indices[:threshold_idx]]
    prec_10 = np.mean(top_10_correct) if len(top_10_correct) > 0 else 0.0
    
    exp1_results.append({
        "Method": name,
        "ROC-AUC": round(auc, 4),
        "ECE": round(ece, 4),
        "Brier": round(brier, 4),
        "Precision@10%": round(prec_10, 4)
    })

print(pd.DataFrame(exp1_results).to_string(index=False))
