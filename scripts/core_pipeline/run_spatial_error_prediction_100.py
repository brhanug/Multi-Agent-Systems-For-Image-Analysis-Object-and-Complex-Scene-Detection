#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import scipy.stats as stats

# Paths
BASE = Path("/data/brhanu/thesis_project")
AUDIT_DIR = BASE / "human_spatial_audit"
AUDIT_IMAGES = AUDIT_DIR / "images"
AUDIT_LABELS = AUDIT_DIR / "labels"
MODEL_WEIGHTS = BASE / "runs/detect/yolo11_v2_augmented_refresh/weights/best.pt"
SCORES_CSV = BASE / "results/multi_agent/agent_comparison_scores.csv"
USER_CSV = BASE / "human_spatial_audit/user_annotations_300.csv"

def get_basename(path_str):
    return str(path_str).split('/')[-1].rsplit('.', 1)[0]

def compute_box_iou(box1, box2):
    # box format: [x_center, y_center, width, height] (normalized)
    x1_min = box1[0] - box1[2] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-9)

def main():
    print("🧪 Running SPATIAL ERROR PREDICTION AUDIT...")
    
    if not MODEL_WEIGHTS.exists():
        print(f"❌ Weights not found at {MODEL_WEIGHTS}")
        return
        
    # Load YOLO Model
    model = YOLO(MODEL_WEIGHTS)
    
    # Load User Annotations to get 100 images
    user_df = pd.read_csv(USER_CSV)
    user_df['basename'] = user_df['filename'].apply(get_basename)
    
    # Load agent comparison scores to get AI Uncertainty
    agent_df = pd.read_csv(SCORES_CSV)
    agent_df['basename'] = agent_df['image_id'].apply(lambda x: str(x).replace('/', '_'))
    
    spatial_errors = []
    basenames = []
    
    print("  Processing 100 images to detect YOLOv11 spatial failures...")
    for idx, row in user_df.iterrows():
        basename = row['basename']
        img_path = AUDIT_IMAGES / f"{basename}.jpg"
        label_path = AUDIT_LABELS / f"{basename}.txt"
        
        if not img_path.exists() or not label_path.exists():
            continue
            
        # 1. Read ground truth boxes
        gt_boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    gt_boxes.append((cls_id, coords))
                    
        # 2. Run YOLO prediction
        results = model(img_path, verbose=False)
        pred_boxes = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            # Convert xywh to normalized format
            orig_h, orig_w = results[0].orig_shape
            xywh = box.xywh[0].cpu().numpy()
            norm_coords = [
                xywh[0] / orig_w,
                xywh[1] / orig_h,
                xywh[2] / orig_w,
                xywh[3] / orig_h
            ]
            pred_boxes.append((cls_id, norm_coords))
            
        # 3. Determine if model made an error
        # An error occurs if:
        # - Any GT box does not match any Pred box of same class with IoU >= 0.30 (False Negative)
        # - Any Pred box does not match any GT box of same class with IoU >= 0.30 (False Positive)
        has_error = 0
        
        # Check for False Negatives
        for gt_cls, gt_coords in gt_boxes:
            matched = False
            for pred_cls, pred_coords in pred_boxes:
                if gt_cls == pred_cls:
                    iou = compute_box_iou(gt_coords, pred_coords)
                    if iou >= 0.30:
                        matched = True
                        break
            if not matched:
                has_error = 1
                break
                      
        # Check for False Positives if no error found yet
        if not has_error:
            for pred_cls, pred_coords in pred_boxes:
                matched = False
                for gt_cls, gt_coords in gt_boxes:
                    if gt_cls == pred_cls:
                        iou = compute_box_iou(gt_coords, pred_coords)
                        if iou >= 0.30:
                            matched = True
                            break
                if not matched:
                    has_error = 1
                    break
                    
        spatial_errors.append(has_error)
        basenames.append(basename)
        
    error_df = pd.DataFrame({
        'basename': basenames,
        'yolo_spatial_error': spatial_errors
    })
    
    # Merge with Agent comparison scores
    merged = error_df.merge(agent_df, on='basename')
    
    # Also merge with user_df to get the ground truth scene label
    merged = merged.merge(user_df[['basename', 'scene']], on='basename')
    merged['has_scene'] = (merged['scene'].notna() & (merged['scene'] != "") & (merged['scene'] != "none")).astype(int)
    
    # Calculate Unified AI Uncertainty (U)
    agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    sigma_agents = merged[agent_cols].std(axis=1).fillna(0)
    c_bar = merged[agent_cols].mean(axis=1).fillna(0)
    merged['ai_uncertainty'] = (0.6 * sigma_agents) + (0.4 * (1 - c_bar))
    merged['sigma_agents'] = sigma_agents
    
    # --- Function to report metrics for a cohort ---
    def report_metrics(df_sub, name):
        total_errors = df_sub['yolo_spatial_error'].sum()
        n_total = len(df_sub)
        print(f"\n==================================================")
        print(f"📊 {name} (N = {n_total})")
        print(f"==================================================")
        print(f"  YOLOv11 Spatial Errors: {total_errors} / {n_total} ({total_errors/n_total*100:.1f}%)")
        
        y_true = df_sub['yolo_spatial_error'].values
        y_scores = df_sub['ai_uncertainty'].values
        y_sigma = df_sub['sigma_agents'].values
        
        # Check if we have both classes for ROC-AUC
        if len(np.unique(y_true)) > 1:
            auc_roc = roc_auc_score(y_true, y_scores)
            auc_pr = average_precision_score(y_true, y_scores)
            point_biserial_r, p_val = stats.pointbiserialr(y_true, y_scores)
            
            auc_roc_sigma = roc_auc_score(y_true, y_sigma)
            
            print(f"  [AI Uncertainty Formula]:")
            print(f"    AUROC (Area Under ROC)     : {auc_roc:.4f}")
            print(f"    AUPRC (Area Under PR)      : {auc_pr:.4f}")
            print(f"    Point Biserial Correlation : {point_biserial_r:.4f} (p = {p_val:.2e})")
            print(f"  [Pure Agent Disagreement (sigma)]:")
            print(f"    AUROC (Area Under ROC)     : {auc_roc_sigma:.4f}")
        else:
            print("  ⚠️ Only one class present in ground truth errors - skipping AUC/Correlation.")
            
        mean_u_error = df_sub[df_sub['yolo_spatial_error'] == 1]['ai_uncertainty'].mean()
        mean_u_success = df_sub[df_sub['yolo_spatial_error'] == 0]['ai_uncertainty'].mean()
        print(f"  Mean Uncertainty on Success  : {mean_u_success:.4f}")
        print(f"  Mean Uncertainty on Error    : {mean_u_error:.4f}")
    
    # 1. Overall Cohort
    report_metrics(merged, "OVERALL COHORT")
    
    # 2. Positive-only Cohort (images containing a scene)
    report_metrics(merged[merged['has_scene'] == 1], "POSITIVE SCENE COHORT (Images containing a scene)")
    
    # 3. Negative-only Cohort (images containing no scene / empty / text pages)
    report_metrics(merged[merged['has_scene'] == 0], "NEGATIVE SCENE COHORT (Images with no scene)")
    print("==================================================")
    
    # 4. Train a Meta-Classifier (Logistic Regression) to predict YOLO spatial errors
    print("\n==================================================")
    print("🤖 TRAINING SPATIAL ERROR PREDICTOR (METACLASS)")
    print("==================================================")
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    
    X = merged[agent_cols].fillna(0.5)
    y = merged['yolo_spatial_error']
    
    lr = LogisticRegression(random_state=42)
    scores_auc = cross_val_score(lr, X, y, cv=5, scoring='roc_auc')
    scores_pr = cross_val_score(lr, X, y, cv=5, scoring='average_precision')
    
    print(f"  5-Fold CV AUROC : {np.mean(scores_auc):.4f}")
    print(f"  5-Fold CV AUPRC : {np.mean(scores_pr):.4f}")
    print("==================================================")

if __name__ == '__main__':
    main()
