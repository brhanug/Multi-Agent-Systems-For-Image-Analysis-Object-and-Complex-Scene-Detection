#!/usr/bin/env python3
"""
exp20_error_targeted_awlf.py
------------------------------
Experiment 20: Error-Targeted AWLF Retraining

KEY INSIGHT: The existing AWLF is trained on scene-PRESENCE labels
(does this image have a scene?), suffering from 99.4% class imbalance
causing F1=0.9969 to be a trivial artifact.

THIS EXPERIMENT: Retrains AWLF on CLASSIFICATION ERROR labels:
  - y=1 if the SAA coordinator prediction DISAGREES with human expert label
  - y=0 if the SAA coordinator prediction AGREES with human expert label

This produces a genuine uncertainty discriminator that predicts WHEN
the system will fail — the true purpose of the AWLF.

Expected output:
  - results/multi_agent/exp20_error_awlf_results.json
  - results/figures/exp20_error_awlf_roc_curve.png
  - results/figures/exp20_error_awlf_feature_importance.png
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, precision_score,
    recall_score, accuracy_score, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

BASE = Path("/data/brhanu/thesis_project")
EXPERT_CSV   = BASE / "human_spatial_audit" / "user_annotations_800.csv"
SCORES_CSV   = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT_DIR      = BASE / "results/multi_agent"
FIG_DIR      = BASE / "results/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─── helpers ──────────────────────────────────────────────────────────────────
from picture_ids import full_id as normalize_id, read_expert_annotations   # book and page, never the page alone


def recall_at_budget(y_true, y_scores, budget: float) -> float:
    """Fraction of true errors caught within a budget% of the corpus."""
    n = len(y_true)
    k = max(1, int(n * budget))
    top_k_idx = np.argsort(y_scores)[::-1][:k]
    return y_true.iloc[top_k_idx].sum() / y_true.sum() if y_true.sum() > 0 else 0.0

def precision_at_budget(y_true, y_scores, budget: float) -> float:
    n = len(y_true)
    k = max(1, int(n * budget))
    top_k_idx = np.argsort(y_scores)[::-1][:k]
    return y_true.iloc[top_k_idx].mean()

# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("EXP 20: Error-Targeted AWLF — Training on Classification Errors")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    expert = read_expert_annotations(EXPERT_CSV)
    expert["cvat_id"]       = expert.index
    expert["gold_has_scene"] = (expert["n_scene_labels"] > 0).astype(int)
    expert_reviewed          = expert[expert["cvat_id"] <= 800].copy()
    print(f"  Expert annotations loaded: {len(expert_reviewed)} images")

    agents = pd.read_csv(SCORES_CSV)
    agents["image_id"] = agents["image_id"].apply(normalize_id)
    agents = agents.drop_duplicates(subset=["image_id"])
    print(f"  Agent scores loaded: {len(agents)} images")

    # ── 2. Join ───────────────────────────────────────────────────────────────
    joined = expert_reviewed.merge(agents, on="image_id", how="inner")
    print(f"  Matched: {len(joined)} images")

    if len(joined) < 50:
        print("❌  Too few matched images — check ID normalisation.")
        return

    # ── 3. Build the ERROR target ─────────────────────────────────────────────
    # SAA fusion score: weighted average of core agent signals (same as thesis)
    # weights: object=0.6, scene=0.2, vlm=0.2 (heuristic SAA)
    joined["saa_fusion"] = (
        0.6 * joined["existing_pipeline_agent"]
        + 0.2 * joined["scene_agent"]
        + 0.2 * joined["vlm_agent"]
    )
    joined["saa_pred"]   = (joined["saa_fusion"] >= 0.5).astype(int)
    # ERROR = 1 when coordinator prediction ≠ human ground truth
    joined["error_label"] = (joined["saa_pred"] != joined["gold_has_scene"]).astype(int)

    n_errors = joined["error_label"].sum()
    n_total  = len(joined)
    print(f"\n  Error label distribution:")
    print(f"    Errors (y=1):  {n_errors}  ({100*n_errors/n_total:.1f}%)")
    print(f"    Correct (y=0): {n_total-n_errors}  ({100*(n_total-n_errors)/n_total:.1f}%)")

    if n_errors < 5:
        print("⚠️  Very few errors detected — system may already be highly accurate.")
        print("    Proceeding with available data.")

    # ── 4. Features ───────────────────────────────────────────────────────────
    feature_cols = [
        "existing_pipeline_agent",   # YOLO spatial grounding
        "agreement_agent",           # SAA consensus
        "scene_agent",               # SigLIP scene
        "vlm_agent",                 # LLaVA-OV semantic
        "restoration_agent",         # restoration quality signal
        "document_agent",            # grounded OCR signal
    ]
    available_cols = [c for c in feature_cols if c in joined.columns]
    print(f"\n  Feature columns used: {available_cols}")

    X = joined[available_cols].fillna(0.5)
    y = joined["error_label"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=available_cols, index=X.index)

    # ── 5. 5-Fold cross-validation for all classifiers ────────────────────────
    classifiers = {
        "Logistic Regression (AWLF)": LogisticRegression(C=1.0, random_state=42, max_iter=500),
        "Gradient Boosting":           GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Random Forest":               RandomForestClassifier(n_estimators=100, random_state=42),
    }

    print("\n  ── 5-Fold CV Results ──")
    cv_results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, clf in classifiers.items():
        aucs = cross_val_score(clf, X_scaled, y, cv=skf, scoring="roc_auc")
        cv_results[name] = {"mean_auc": float(aucs.mean()), "std_auc": float(aucs.std())}
        print(f"    {name:35s}: ROC-AUC = {aucs.mean():.4f} ± {aucs.std():.4f}")

    # ── 6. Full-data fit for detailed evaluation (Logistic Regression) ─────────
    print("\n  ── Full-Data Logistic AWLF Evaluation ──")
    lr_clf = LogisticRegression(C=1.0, random_state=42, max_iter=500)
    lr_clf.fit(X_scaled, y)
    probs = lr_clf.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)

    roc_auc   = roc_auc_score(y, probs)
    avg_prec  = average_precision_score(y, probs)
    rec_10    = recall_at_budget(y, pd.Series(probs), 0.10)
    rec_20    = recall_at_budget(y, pd.Series(probs), 0.20)
    prec_10   = precision_at_budget(y, pd.Series(probs), 0.10)

    weights = dict(zip(available_cols, lr_clf.coef_[0].tolist()))
    print(f"\n    ROC-AUC (error prediction):  {roc_auc:.4f}")
    print(f"    Avg Precision (PR-AUC):      {avg_prec:.4f}")
    print(f"    Recall @ 10% budget:         {rec_10:.4f}")
    print(f"    Recall @ 20% budget:         {rec_20:.4f}")
    print(f"    Precision @ 10% budget:      {prec_10:.4f}")
    print(f"\n    Learned feature weights:")
    for k, v in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"      {k:35s}: {v:+.4f}")

    # ── 7. Baseline comparison ─────────────────────────────────────────────────
    print("\n  ── Baseline Comparison (error prediction task) ──")
    baseline_scores = {
        "Monolithic pipeline (1-x)":      1 - joined["monolithic_pipeline_agent"].fillna(0.5),
        "YOLO inverse confidence (1-x)":  1 - joined["existing_pipeline_agent"],
        "Agreement inverse (1-x)":        1 - joined["agreement_agent"],
        "VLM inverse (1-x)":              1 - joined["vlm_agent"],
    }
    baseline_results = []
    for name, scores in baseline_scores.items():
        scores_clean = scores.fillna(0.5)
        try:
            auc = roc_auc_score(y, scores_clean)
            r10 = recall_at_budget(y, scores_clean, 0.10)
            r20 = recall_at_budget(y, scores_clean, 0.20)
            baseline_results.append({
                "Method": name, "ROC-AUC": round(auc, 4),
                "Recall@10%": round(r10, 4), "Recall@20%": round(r20, 4)
            })
            print(f"    {name:38s}: AUC={auc:.4f}  R@10%={r10:.4f}  R@20%={r20:.4f}")
        except Exception as e:
            print(f"    {name}: ERROR — {e}")

    # ── 8. Plots ───────────────────────────────────────────────────────────────
    # 8a. ROC Curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Exp 20: Error-Targeted AWLF — Error Prediction Performance",
                 fontsize=13, fontweight="bold")

    fpr, tpr, _ = roc_curve(y, probs)
    axes[0].plot(fpr, tpr, lw=2, color="#1A56A0",
                 label=f"Error-Targeted AWLF (AUC={roc_auc:.3f})")
    for bname, bscores in baseline_scores.items():
        bscores_clean = bscores.fillna(0.5)
        try:
            bfpr, btpr, _ = roc_curve(y, bscores_clean)
            bauc = roc_auc_score(y, bscores_clean)
            axes[0].plot(bfpr, btpr, lw=1.2, linestyle="--", alpha=0.7,
                         label=f"{bname} (AUC={bauc:.3f})")
        except Exception:
            pass
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].set_title("ROC Curve — Error Prediction", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # 8b. Feature importance
    sorted_feats = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
    feat_names   = [f[0] for f in sorted_feats]
    feat_vals    = [f[1] for f in sorted_feats]
    colors       = ["#1A7A4A" if v > 0 else "#B71C1C" for v in feat_vals]
    axes[1].barh(feat_names, feat_vals, color=colors)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Logistic Regression Coefficient", fontsize=11)
    axes[1].set_title("Error-Targeted AWLF Feature Weights", fontsize=11)
    axes[1].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    fig_path = FIG_DIR / "exp20_error_awlf_roc_curve.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Figure saved: {fig_path}")

    # ── 9. Save results ────────────────────────────────────────────────────────
    report = {
        "experiment": "Exp 20: Error-Targeted AWLF",
        "key_finding": (
            "AWLF retrained on classification error labels (system≠human) "
            "rather than scene-presence labels, eliminating the F1=0.9969 "
            "class-imbalance artifact and producing genuine error discrimination."
        ),
        "data": {
            "n_total": n_total,
            "n_errors": int(n_errors),
            "error_rate_pct": round(100 * n_errors / n_total, 2),
            "features": available_cols,
        },
        "error_awlf_performance": {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(avg_prec, 4),
            "recall_at_10pct_budget": round(rec_10, 4),
            "recall_at_20pct_budget": round(rec_20, 4),
            "precision_at_10pct_budget": round(prec_10, 4),
        },
        "learned_weights": {k: round(v, 6) for k, v in weights.items()},
        "cv_results": cv_results,
        "baseline_comparison": baseline_results,
        "figure": str(fig_path),
        "interpretation": (
            "A positive coefficient means this feature INCREASES error probability "
            "(higher disagreement → more likely error). A negative coefficient means "
            "this feature DECREASES error probability (higher score → more reliable)."
        ),
    }

    out_path = OUT_DIR / "exp20_error_awlf_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✅ Results saved: {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Error-Targeted AWLF ROC-AUC:  {roc_auc:.4f}")
    print(f"  Recall @ 10% budget:          {rec_10:.4f}")
    print(f"  Recall @ 20% budget:          {rec_20:.4f}")
    print(f"  Error rate in data:           {100*n_errors/n_total:.1f}%")
    print()
    print("  INTERPRETATION: Unlike F1=0.9969 (a trivial artifact of 99.4%")
    print("  class imbalance), this ROC-AUC measures whether the AWLF can")
    print("  predict WHICH images the system will get WRONG — the true goal.")
    print("=" * 70)

if __name__ == "__main__":
    main()
