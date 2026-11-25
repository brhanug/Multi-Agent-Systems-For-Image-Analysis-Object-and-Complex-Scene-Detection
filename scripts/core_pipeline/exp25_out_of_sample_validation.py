#!/usr/bin/env python3
"""
exp25_out_of_sample_validation.py
----------------------------------
Performs a fully held-out out-of-sample calibration audit.
Reserves Batch 2 (n=300) strictly as a test set.
Trains the AWLF classifier and optimizes scene-adaptive weights ONLY on Batch 1 (n=801).
Evaluates clean generalization performance on Batch 2.

Outputs:
  results/multi_agent/exp25_out_of_sample_results.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE         = Path("/data/brhanu/thesis_project")
BATCH1_CSV   = BASE / "human_spatial_audit/user_annotations_800.csv"
BATCH2_CSV   = BASE / "human_spatial_audit/user_annotations_300.csv"
MANIFEST_CSV = BASE / "final_dataset/metadata/manifest.csv"
AGENTS_CSV   = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT_DIR      = BASE / "results/multi_agent"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def to_ppn_id(raw_id):
    name = str(raw_id).replace("images/", "").rsplit(".", 1)[0]
    parts = name.split("_", 1)
    if len(parts) > 1:
        return f"{parts[0]}/{parts[1]}"
    return name

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
    print("EXP 25: OUT-OF-SAMPLE CALIBRATION AUDIT (HELD-OUT VALIDATION)")
    print("=" * 70)

    # ── Load datasets ──
    b1_raw = pd.read_csv(BATCH1_CSV)
    b2_raw = pd.read_csv(BATCH2_CSV)
    
    # Train strictly on reviewed Batch 1 subset
    b1_raw = b1_raw[b1_raw.index <= 800].copy()
    
    b1_raw["has_scene"] = (b1_raw["n_scene_labels"] > 0).astype(int)
    b2_raw["has_scene"] = (b2_raw["scene"].notna() & (b2_raw["scene"] != "") & (b2_raw["scene"] != "none")).astype(int)
    
    b1_raw["ppn_id"] = b1_raw["raw_name"].apply(to_ppn_id)
    b2_raw["ppn_id"] = b2_raw["image_id"].apply(to_ppn_id)
    
    agents = pd.read_csv(AGENTS_CSV).drop_duplicates("image_id")
    manifest = pd.read_csv(MANIFEST_CSV)
    
    # Merge scores
    b1 = b1_raw.merge(agents, left_on="ppn_id", right_on="image_id", how="inner")
    b2 = b2_raw.merge(agents, left_on="ppn_id", right_on="image_id", how="inner")
    
    # Merge primary scene
    b1 = b1.merge(manifest[["image_id", "vqa_primary_scene"]], left_on="ppn_id", right_on="image_id", suffixes=("", "_manifest"))
    b2 = b2.merge(manifest[["image_id", "vqa_primary_scene"]], left_on="ppn_id", right_on="image_id", suffixes=("", "_manifest"))
    
    print(f"  Training set (Batch 1 Reviewed): {len(b1)} samples")
    print(f"  Test set (Batch 2 Spatial):      {len(b2)} samples")

    # Define features
    features = [
        "existing_pipeline_agent", 
        "agreement_agent", 
        "scene_agent", 
        "vlm_agent",
        "restoration_agent",
        "document_agent"
    ]
    
    # SAA predictions and error labels
    alpha, beta, gamma = 0.6, 0.2, 0.2
    
    b1["saa"] = alpha * b1["existing_pipeline_agent"] + beta * b1["scene_agent"] + gamma * b1["vlm_agent"]
    b2["saa"] = alpha * b2["existing_pipeline_agent"] + beta * b2["scene_agent"] + gamma * b2["vlm_agent"]
    
    b1["saa_pred"] = (b1["saa"] >= 0.5).astype(int)
    b2["saa_pred"] = (b2["saa"] >= 0.5).astype(int)
    
    b1["error_label"] = (b1["saa_pred"] != b1["has_scene"]).astype(int)
    b2["error_label"] = (b2["saa_pred"] != b2["has_scene"]).astype(int)

    # ── Part 1: AWLF Generalization ──
    print("\n  ── Part 1: AWLF Error Prediction Generalization ──")
    X_train = b1[features].fillna(0.5)
    y_train = b1["error_label"]
    X_test  = b2[features].fillna(0.5)
    y_test  = b2["error_label"]
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_sc, y_train)
    
    train_probs = lr.predict_proba(X_train_sc)[:, 1]
    test_probs  = lr.predict_proba(X_test_sc)[:, 1]
    
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc  = roc_auc_score(y_test, test_probs)
    
    train_ece = ece(y_train, train_probs)
    test_ece  = ece(y_test, test_probs)
    
    print(f"    Train (B1) AUC: {train_auc:.4f} | ECE: {train_ece:.4f}")
    print(f"    Test  (B2) AUC: {test_auc:.4f} | ECE: {test_ece:.4f}")

    # ── Part 2: Scene-Adaptive Weight Generalization ──
    print("\n  ── Part 2: Scene-Adaptive Weight Optimization Generalization ──")
    categories = ["drawing", "landscape", "playing", "family", "teaching"]
    
    # 1. Learn optimal weights per category on Batch 1 (Train)
    optimal_weights_by_cat = {}
    for cat in categories:
        df_cat = b1[b1["vqa_primary_scene"] == cat].copy()
        if len(df_cat) == 0:
            optimal_weights_by_cat[cat] = (alpha, beta, gamma)
            continue
            
        x_yolo  = df_cat["existing_pipeline_agent"].values
        x_scene = df_cat["scene_agent"].values
        x_vlm   = df_cat["vlm_agent"].values
        y_gold  = df_cat["has_scene"].values
        
        best_err = 1.0
        best_w   = (alpha, beta, gamma)
        best_e   = ece(y_gold, alpha*x_yolo + beta*x_scene + gamma*x_vlm)
        
        for a in np.linspace(0.0, 1.0, 51):
            for b in np.linspace(0.0, 1.0 - a, 51):
                c = 1.0 - a - b
                fusion = a * x_yolo + b * x_scene + c * x_vlm
                preds  = (fusion >= 0.5).astype(int)
                err    = (preds != y_gold).mean()
                
                if err < best_err:
                    best_err = err
                    best_w   = (a, b, c)
                    best_e   = ece(y_gold, fusion)
                elif err == best_err:
                    curr_e = ece(y_gold, fusion)
                    if curr_e < best_e:
                        best_w   = (a, b, c)
                        best_e   = curr_e
        
        optimal_weights_by_cat[cat] = best_w
        print(f"    Optimized weights for '{cat}' on B1: YOLO={best_w[0]:.2f}, Scene={best_w[1]:.2f}, VLM={best_w[2]:.2f}")

    # 2. Apply these weights zero-shot to Batch 2 (Test)
    b2_base_errs = []
    b2_opt_errs  = []
    b2_base_eces = []
    b2_opt_eces  = []
    
    for cat in categories:
        b2_cat = b2[b2["vqa_primary_scene"] == cat].copy()
        n_test = len(b2_cat)
        if n_test == 0:
            continue
            
        x_yolo  = b2_cat["existing_pipeline_agent"].values
        x_scene = b2_cat["scene_agent"].values
        x_vlm   = b2_cat["vlm_agent"].values
        y_gold  = b2_cat["has_scene"].values
        
        # Heuristic baseline
        base_fusion = alpha*x_yolo + beta*x_scene + gamma*x_vlm
        base_preds  = (base_fusion >= 0.5).astype(int)
        base_err    = (base_preds != y_gold).mean()
        base_ece    = ece(y_gold, base_fusion)
        
        # Optimized weights (from Train set)
        w_opt = optimal_weights_by_cat[cat]
        opt_fusion = w_opt[0]*x_yolo + w_opt[1]*x_scene + w_opt[2]*x_vlm
        opt_preds  = (opt_fusion >= 0.5).astype(int)
        opt_err    = (opt_preds != y_gold).mean()
        opt_ece    = ece(y_gold, opt_fusion)
        
        b2_base_errs.append(base_err * n_test)
        b2_opt_errs.append(opt_err * n_test)
        b2_base_eces.append(base_ece * n_test)
        b2_opt_eces.append(opt_ece * n_test)
        
        print(f"    '{cat}' (Test n={n_test}): Baseline err={base_err:.4f}, ECE={base_ece:.4f} | Generalised err={opt_err:.4f}, ECE={opt_ece:.4f}")

    # Aggregated metrics on Test Set
    total_test = len(b2)
    agg_base_err = sum(b2_base_errs) / total_test
    agg_opt_err  = sum(b2_opt_errs) / total_test
    agg_base_ece = sum(b2_base_eces) / total_test
    agg_opt_ece  = sum(b2_opt_eces) / total_test
    
    print("\n  ── Final Out-of-Sample Test Set Summary (Batch 2) ──")
    print(f"    Aggregated Baseline Error Rate: {agg_base_err:.4f} -> Optimized: {agg_opt_err:.4f}")
    print(f"    Aggregated Baseline ECE:        {agg_base_ece:.4f} -> Optimized: {agg_opt_ece:.4f}")
    
    results = {
        "train_samples": len(b1),
        "test_samples": len(b2),
        "awlf": {
            "train_auc": round(float(train_auc), 4),
            "test_auc": round(float(test_auc), 4),
            "train_ece": round(float(train_ece), 4),
            "test_ece": round(float(test_ece), 4),
        },
        "scene_adaptive_weights": {
            "optimal_weights": {cat: [round(float(w), 2) for w in optimal_weights_by_cat[cat]] for cat in categories},
            "baseline_error_rate": round(float(agg_base_err), 4),
            "optimized_error_rate": round(float(agg_opt_err), 4),
            "baseline_ece": round(float(agg_base_ece), 4),
            "optimized_ece": round(float(agg_opt_ece), 4),
        }
    }
    
    out_path = OUT_DIR / "exp25_out_of_sample_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✅ Results saved to: {out_path}")

if __name__ == "__main__":
    main()
