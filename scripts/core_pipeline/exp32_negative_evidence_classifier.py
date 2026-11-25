#!/usr/bin/env python3
"""
exp32_negative_evidence_classifier.py
----------------------------------------
Attempts the fix proposed for the negative-control blind spot (Section
"Limitations", thesis; Slide 4.3, defense deck): U_SAA is built from
agent DISAGREEMENT, so it is structurally blind to unanimous false
positives -- cases where all four agents confidently agree a scene is
present when it is not. The proposed fix is a dedicated classifier whose
job is exactly that judgment, trained directly on the agents' raw
confidence scores (not their disagreement), on a realistic, group-level,
leak-free split.

This does NOT require new annotation: the frozen clean_holdout_split.csv
(results/multi_agent/build_clean_holdout_split.py) already carries real,
human-verified has_scene labels for all 2,000 images, roughly balanced
(~40% positive) across train/calibration/test, specifically because it
was built to fix the earlier positive-heavy-cohort leakage problem.

Trains a Gradient-Boosted classifier on the 4 raw Tier 1 agent scores
(object, scene, vlm, agreement) using the TRAIN split (n=1,200), then
reports its real performance on the held-out TEST split (n=401),
decomposed by true class exactly as Section "Group-Level Held-Out
Validation" already does for U_SAA -- for a direct, apples-to-apples
comparison of whether this actually closes the gap.

Outputs:
  results/multi_agent/exp32_negative_evidence_classifier.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

BASE = Path("/data/brhanu/thesis_project")
SPLIT_CSV = BASE / "results" / "multi_agent" / "clean_holdout_split.csv"
SCORES_CSV = BASE / "results" / "multi_agent" / "agent_comparison_scores.csv"
OUT_JSON = BASE / "results" / "multi_agent" / "exp32_negative_evidence_classifier.json"

FEATURES = ["existing_pipeline_agent", "scene_agent", "vlm_agent", "agreement_agent"]


def main() -> None:
    split = pd.read_csv(SPLIT_CSV)
    agents = pd.read_csv(SCORES_CSV)
    joined = split.merge(agents, left_on="ppn_id", right_on="image_id", how="inner")
    print(f"Matched {len(joined)}/{len(split)} images to agent scores.")

    train = joined[joined["split"] == "train"]
    test = joined[joined["split"] == "test"]
    print(f"Train: {len(train)} (positive rate {train['has_scene'].mean():.3f})")
    print(f"Test:  {len(test)} (positive rate {test['has_scene'].mean():.3f})")

    X_train, y_train = train[FEATURES].values, train["has_scene"].values
    X_test, y_test = test[FEATURES].values, test["has_scene"].values

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)
    p_test = clf.predict_proba(X_test)[:, 1]

    overall_auroc = roc_auc_score(y_test, p_test)

    pos_mask = y_test == 1
    neg_mask = y_test == 0

    # For the negative class, a classifier's P(scene) should be LOW; frame
    # its "error-detection" AUROC on negatives as U_SAA's was framed, using
    # 1 - p as the error-likelihood score for that subgroup, and correctness
    # (matches human label) as the target -- for a direct comparison.
    error_label = (p_test.round() != y_test).astype(int)  # not used directly; kept for inspection

    pos_auroc = roc_auc_score(y_test[pos_mask], p_test[pos_mask]) if pos_mask.sum() > 1 and len(set(y_test[pos_mask])) > 1 else None
    neg_correct_rate = float((p_test[neg_mask] < 0.5).mean())  # fraction of true negatives correctly predicted negative
    pos_correct_rate = float((p_test[pos_mask] >= 0.5).mean())  # fraction of true positives correctly predicted positive

    result = {
        "n_train": len(train),
        "n_test": len(test),
        "features": FEATURES,
        "overall_test_auroc_has_scene": round(float(overall_auroc), 4),
        "negative_controls_correctly_flagged_as_no_scene": round(neg_correct_rate, 4),
        "n_negative_controls_test": int(neg_mask.sum()),
        "positive_images_correctly_flagged_as_scene": round(pos_correct_rate, 4),
        "n_positive_images_test": int(pos_mask.sum()),
        "comparison": {
            "U_SAA_disagreement_negative_control_error_detection_auroc": 0.0201,
            "note": "U_SAA's 0.0201 measures whether disagreement-magnitude tracks "
                    "error on negative controls (it is anti-correlated -- disagreement "
                    "is LOW exactly when the ensemble is wrong). This classifier is a "
                    "different, complementary question: can a model trained directly "
                    "on raw confidence scores (not their spread) correctly call "
                    "has_scene=False on those same images. The metric "
                    "'negative_controls_correctly_flagged_as_no_scene' above is the "
                    "real, comparable answer to that question.",
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))

    print(f"\nOverall test AUROC (has_scene): {result['overall_test_auroc_has_scene']}")
    print(f"Negative controls correctly flagged 'no scene': "
          f"{result['negative_controls_correctly_flagged_as_no_scene']:.1%} "
          f"(n={result['n_negative_controls_test']})")
    print(f"Positive images correctly flagged 'has scene':  "
          f"{result['positive_images_correctly_flagged_as_scene']:.1%} "
          f"(n={result['n_positive_images_test']})")
    print(f"\n✅ Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
