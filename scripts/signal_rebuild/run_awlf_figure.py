#!/usr/bin/env python3
"""Figure F.1 redrawn for print: the AWLF error meta-classifier's ROC curve and coefficients.

Recomputes exactly what scripts/core_pipeline/exp20_error_targeted_awlf.py computes (same data,
same features, same full-data logistic fit) and stops unless the ROC-AUC and the six coefficients
equal the stored results/multi_agent/exp20_error_awlf_results.json. Only the drawing differs:
no internal experiment title, signal names instead of column names, a legend readable in print.
Writes latex_assets/awlf_roc_print.png; the result file is not touched.

    python run_awlf_figure.py
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

BASE = Path("/data/brhanu/thesis_project")
sys.path.insert(0, str(BASE / "scripts/core_pipeline"))
from picture_ids import full_id as normalize_id, read_expert_annotations  # noqa: E402

STORED = json.loads((BASE / "results/multi_agent/exp20_error_awlf_results.json").read_text())
OUT = BASE / "latex_assets/awlf_roc_print.png"
COLS = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent", "restoration_agent", "document_agent"]
NAMES = {"existing_pipeline_agent": "Object signal", "agreement_agent": "Agreement (derived)", "scene_agent": "Scene signal",
         "vlm_agent": "VLM signal", "restoration_agent": "Restoration quality", "document_agent": "OCR signal"}


def main():
    expert = read_expert_annotations(BASE / "human_spatial_audit/user_annotations_800.csv")
    expert["cvat_id"] = expert.index
    expert["gold_has_scene"] = (expert["n_scene_labels"] > 0).astype(int)
    expert = expert[expert["cvat_id"] <= 800].copy()
    agents = pd.read_csv(BASE / "results/multi_agent/agent_comparison_scores.csv")
    agents["image_id"] = agents["image_id"].apply(normalize_id)
    j = expert.merge(agents.drop_duplicates(subset=["image_id"]), on="image_id", how="inner")
    saa = 0.6 * j["existing_pipeline_agent"] + 0.2 * j["scene_agent"] + 0.2 * j["vlm_agent"]
    y = ((saa >= 0.5).astype(int) != j["gold_has_scene"]).astype(int)
    X = StandardScaler().fit_transform(j[COLS].fillna(0.5))
    lr = LogisticRegression(C=1.0, random_state=42, max_iter=500).fit(X, y)
    p = lr.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, p)
    w = dict(zip(COLS, lr.coef_[0]))
    assert round(auc, 4) == STORED["error_awlf_performance"]["roc_auc"], auc
    for c in COLS:
        assert abs(w[c] - STORED["learned_weights"][c]) < 1e-5, (c, w[c])
    base = {"Single-pipeline baseline": 1 - j["monolithic_pipeline_agent"].fillna(0.5),
            "1 − object signal": 1 - j["existing_pipeline_agent"],
            "1 − agreement": 1 - j["agreement_agent"], "1 − VLM signal": 1 - j["vlm_agent"]}
    ink, accent, grey = "#1E2733", "#2D5F8B", "#8A929A"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    f, t, _ = roc_curve(y, p)
    ax[0].plot(f, t, lw=2.2, color=accent, label=f"AWLF meta-classifier ({auc:.3f})")
    for (name, s), ls in zip(base.items(), ["-", "--", "-.", ":"]):
        bf, bt, _ = roc_curve(y, s.fillna(0.5))
        ax[0].plot(bf, bt, lw=1.3, ls=ls, color=grey, label=f"{name} ({roc_auc_score(y, s.fillna(0.5)):.3f})")
    ax[0].plot([0, 1], [0, 1], color="#C9CED3", lw=0.8)
    ax[0].set_xlabel("False positive rate", fontsize=11); ax[0].set_ylabel("True positive rate", fontsize=11)
    ax[0].set_title("Ranking the pipeline's errors (ROC-AUC in brackets)", fontsize=11, color=ink, loc="left")
    ax[0].legend(fontsize=9.5, frameon=False, loc="lower right")
    order = sorted(COLS, key=lambda c: abs(w[c]))
    ax[1].barh([NAMES[c] for c in order], [w[c] for c in order],
               color=[accent if w[c] > 0 else "#A2461B" for c in order], height=0.6)
    ax[1].axvline(0, color=ink, lw=0.8)
    ax[1].set_xlabel("Logistic coefficient (standardised signals)", fontsize=11)
    ax[1].set_title("What the meta-classifier weighs", fontsize=11, color=ink, loc="left")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=10)
    fig.tight_layout(); fig.savefig(OUT, dpi=220)
    print(f"reproduces stored ROC-AUC {auc:.4f} and coefficients; wrote {OUT}")


if __name__ == "__main__":
    main()
