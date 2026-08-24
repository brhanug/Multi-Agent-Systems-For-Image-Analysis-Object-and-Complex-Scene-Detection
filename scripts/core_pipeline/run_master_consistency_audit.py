#!/usr/bin/env python3
"""
run_master_consistency_audit.py
--------------------------------
Reproducibility spot-checks against a handful of intermediate numbers computed
during development. NOTE (full-project reproducibility audit): this script's
original TEST 3 asserted "PASSED" against ROC-AUC=0.7868, evaluated exactly
the same way the thesis (Sec. 4.4, "AWLF v2 Error-Targeted Retraining") and
defense slides (A.9) explicitly disclose as flawed -- a 40% split of C1
evaluated against the discredited scene-presence target, not the real
error-detection target the rest of this thesis uses. Presenting that as a
verification "PASS" was itself misleading, independent of whether the number
reproduces. TEST 3 has been relabeled to make clear it reproduces a
superseded computation, not a validated thesis claim; the thesis's actual
generalization estimate (ROC-AUC=0.6669, group-level held-out) is computed
separately in the main experiment pipeline (Section 4.4 / Sec. clean_holdout),
not here. TEST 1's "yolo_error" target is also YOLO-alone's own prediction
error, a different task from the fusion-score error target that produces the
headline SAA/inverse-confidence figures (0.9724 / 0.9726) reported elsewhere
-- it is not a re-derivation of those two numbers and should not be read as
one.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")
EXPERT_CSV = BASE / "human_spatial_audit" / "user_annotations_800.csv"
SCORES_CSV = BASE / "results/multi_agent/agent_comparison_scores.csv"

def normalize_id(name):
    p = str(name).replace("images/", "").replace("\\", "/")
    p = p.split("/")[-1].rsplit(".", 1)[0]
    parts = p.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 and parts[0].startswith("PPN") else p

def main():
    print("=========================================================")
    print(" 🛡️ VISUAL HISTORIAN: MASTER CONSISTENCY AUDIT ")
    print("=========================================================\n")
    
    # 1. Load Data
    expert = pd.read_csv(EXPERT_CSV)
    expert['cvat_id'] = expert.index
    expert_reviewed = expert[expert['cvat_id'] <= 800].copy()
    expert_reviewed["gold_has_scene"] = (expert_reviewed["n_scene_labels"] > 0).astype(int)
    
    agents = pd.read_csv(SCORES_CSV)
    agents['image_id'] = agents['image_id'].apply(normalize_id)
    agents = agents.drop_duplicates(subset=['image_id'])
    
    joined = expert_reviewed.merge(agents, on='image_id', how='inner')
    print(f"✅ Data Integrity: Matched {len(joined)} expert images.\n")
    
    # ---------------------------------------------------------
    # TEST 1: Path 1 (Disagreement vs Confidence)
    # ---------------------------------------------------------
    print("--- [TEST 1: YOLO-Alone Prediction-Error Detection (NOT the headline SAA/inverse-confidence task)] ---")
    joined["yolo_error"] = (joined["existing_pipeline_agent"] >= 0.5).astype(int) != joined["gold_has_scene"]
    y_true = joined["yolo_error"].astype(int).values

    yolo_conf_inv = 1.0 - joined["existing_pipeline_agent"].values
    agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    joined["sigma_agents"] = joined[agent_cols].std(axis=1)
    disagreement_std = joined["sigma_agents"].values

    auc_conf = roc_auc_score(y_true, yolo_conf_inv)
    auc_disag = roc_auc_score(y_true, disagreement_std)

    print(f"  Single-Model Confidence (Inverse) AUROC: {auc_conf:.4f}")
    print(f"  Inter-Agent Disagreement (Std) AUROC   : {auc_disag:.4f}")
    print(f"  Status: {'REPRODUCED' if abs(auc_conf - 0.9920) < 0.001 else 'DID NOT REPRODUCE'} (development spot-check value ~0.992;")
    print(f"          target here is whether YOLO ALONE mispredicts, distinct from the fusion-score error")
    print(f"          target used for the thesis's headline 0.9724/0.9726 figures -- do not conflate the two)\n")
    
    # ---------------------------------------------------------
    # TEST 2: Path 2 (Drawing Deep Dive)
    # ---------------------------------------------------------
    print("--- [TEST 2: Path 2 'Drawing' Class Deep Dive] ---")
    drawing_subset = joined[joined['label_drawing'] == 1].copy()
    yolo_preds = (drawing_subset["existing_pipeline_agent"] >= 0.5).astype(int)
    vlm_preds = (drawing_subset["vlm_agent"] >= 0.5).astype(int)
    
    yolo_acc = accuracy_score(drawing_subset["gold_has_scene"], yolo_preds)
    vlm_acc = accuracy_score(drawing_subset["gold_has_scene"], vlm_preds)
    
    yolo_failed_vlm_caught = drawing_subset[(yolo_preds == 0) & (vlm_preds == 1)]
    
    print(f"  Total Drawing Images  : {len(drawing_subset)}")
    print(f"  YOLO Accuracy         : {yolo_acc:.4f} (Expected ~0.5817)")
    print(f"  VLM Accuracy          : {vlm_acc:.4f} (Expected ~0.8381)")
    print(f"  MAS Saves (VLM > YOLO): {len(yolo_failed_vlm_caught)}")
    print(f"  Status: {'PASSED' if len(yolo_failed_vlm_caught) == 160 else 'FAILED'}\n")
    
    # ---------------------------------------------------------
    # TEST 3: Path 3 -- HISTORICAL, SUPERSEDED METHODOLOGY, NOT A THESIS CLAIM
    # ---------------------------------------------------------
    print("--- [TEST 3: Historical Same-Cohort Split Reproduction -- SUPERSEDED, NOT A VALIDATION] ---")
    print("  WARNING: this reproduces the exact flawed methodology the thesis (Sec. 4.4) and")
    print("  defense slides (A.9) explicitly disclose as discredited: a random 40% split of C1")
    print("  evaluated against the scene-presence target, not the real classification-error target.")
    print("  Its F1=0.9969 is a trivial always-positive-classifier artifact on this split. This test")
    print("  exists only to confirm that legacy number is still mechanically reproducible from the")
    print("  same (flawed) recipe -- it is NOT a check of any number this thesis stands behind.")
    print("  The thesis's actual, disclosed generalization estimate is ROC-AUC=0.6669, computed on a")
    print("  genuinely disjoint group-level (publication-level) held-out partition against the correct")
    print("  classification-error target -- see the main experiment pipeline (Sec. clean_holdout), not")
    print("  this script.")
    X = joined[agent_cols].copy()
    y = joined["gold_has_scene"].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
    clf = LogisticRegression(C=1.0, random_state=42)
    clf.fit(X_train, y_train)

    test_probs = clf.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    awlf_f1 = f1_score(y_test, test_preds)
    awlf_auc = roc_auc_score(y_test, test_probs)

    print(f"  Legacy split F1 Score  : {awlf_f1:.4f} (matches superseded figure ~0.9969; trivial artifact)")
    print(f"  Legacy split ROC-AUC   : {awlf_auc:.4f} (matches superseded figure ~0.7868; discredited target)")
    print(f"  Status: {'REPRODUCED (superseded, not a validation)' if abs(awlf_auc - 0.7868) < 0.001 else 'DID NOT EVEN REPRODUCE THE SUPERSEDED FIGURE'}\n")
    
    # ---------------------------------------------------------
    # TEST 4: Triage Overlap Analysis (Top 10%)
    # ---------------------------------------------------------
    print("--- [TEST 4: Triage Overlap Analysis (Top 10%)] ---")
    agents_full = pd.read_csv(SCORES_CSV)
    agents_full["sigma_disagreement"] = agents_full[agent_cols].std(axis=1)
    agents_full["yolo_confidence"] = agents_full["existing_pipeline_agent"]
    
    k = int(len(agents_full) * 0.10)
    mas_triage = set(agents_full.nlargest(k, "sigma_disagreement")["image_id"])
    yolo_triage = set(agents_full.nsmallest(k, "yolo_confidence")["image_id"])
    
    intersection = mas_triage.intersection(yolo_triage)
    iou = len(intersection) / len(mas_triage.union(yolo_triage))
    mas_only = mas_triage - yolo_triage
    
    print(f"  Triage Budget (k)     : {k}")
    print(f"  Intersection Size     : {len(intersection)}")
    print(f"  Jaccard Similarity    : {iou:.4f} (Expected ~0.9739)")
    print(f"  Images in 'MAS Only'  : {len(mas_only)}")
    print(f"  Status: {'PASSED' if len(mas_only) == 16 else 'FAILED'}\n")
    
    print("=========================================================")
    print(" Spot-checks above reproduce specific intermediate numbers.")
    print(" This script does NOT re-run or verify every headline claim")
    print(" in the thesis (e.g. the SAA 0.9724 / CRC / clean-holdout")
    print(" figures are computed by the main experiment pipeline, not")
    print(" here) -- see Table 7.1 (Evaluation Provenance) for the")
    print(" authoritative status of every reported result.")
    print("=========================================================")

if __name__ == "__main__":
    main()
