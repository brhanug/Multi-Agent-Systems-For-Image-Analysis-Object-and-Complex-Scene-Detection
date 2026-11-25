#!/usr/bin/env python3
"""
RQ2 Disagreement Analysis: Inter-agent disagreement as an error predictor.

Computes inter-agent disagreement (std dev across agent scores) and evaluates
whether it predicts proxy-issue labels better than any single agent confidence.

Generates:
  - ROC-AUC for disagreement score vs proxy issue
  - Precision-recall curve data
  - Comparison with monolithic and fusion confidence as predictors
  - Calibration statistics

Outputs:
  results/multi_agent/rq2_disagreement_analysis.json
  results/multi_agent/rq2_pr_curve.csv
  results/multi_agent/rq2_roc_curve.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


AGENT_COLS = [
    "existing_pipeline_agent",
    "agreement_agent",
    "scene_agent",
    "vlm_agent",
    "restoration_agent",
    "document_agent",
]


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve and AUC (manual implementation, no sklearn required)."""
    thresholds = np.unique(scores)[::-1]
    tprs, fprs = [], []
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5, np.array([0, 1]), np.array([0, 1]), thresholds

    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tprs.append(tp / pos)
        fprs.append(fp / neg)

    tprs_arr = np.array([0.0] + tprs + [1.0])
    fprs_arr = np.array([0.0] + fprs + [1.0])
    # AUC via trapezoidal rule
    auc = float(np.sum((tprs_arr[1:] + tprs_arr[:-1]) * 0.5 * np.diff(fprs_arr)))
    if auc < 0:
        auc = -auc  # handle reversed ordering
    return auc, fprs_arr, tprs_arr, thresholds


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute Precision-Recall curve and AUC."""
    thresholds = np.unique(scores)[::-1]
    precisions, recalls = [], []
    pos = labels.sum()
    if pos == 0:
        return 0.0, np.array([0, 1]), np.array([0, 0])

    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        pp = (pred == 1).sum()
        precisions.append(tp / pp if pp > 0 else 0.0)
        recalls.append(tp / pos)

    prec_arr = np.array([1.0] + precisions + [0.0])
    rec_arr  = np.array([0.0] + recalls  + [1.0])
    auc = float(np.sum((prec_arr[1:] + prec_arr[:-1]) * 0.5 * np.diff(rec_arr)))
    if auc < 0:
        auc = -auc
    return auc, rec_arr, prec_arr


def normalize_issue_labels(df: pd.DataFrame, base_dir: Path) -> tuple[pd.Series, pd.Series, pd.Series]:
    # 1. Proxy issue: low-confidence images
    q1_fusion    = df["comparison_fusion_score"].quantile(0.25)
    q1_mono      = df["monolithic_pipeline_agent"].quantile(0.25)
    proxy_labels = ((df["comparison_fusion_score"] <= q1_fusion) | (df["monolithic_pipeline_agent"] <= q1_mono)).astype(int)

    # 2. Independent Ground Truth Issue from human annotations (800 and 300 sets)
    gt_mask = np.zeros(len(df), dtype=int)
    gt_matched = np.zeros(len(df), dtype=int)
    anno_800_path = base_dir / "human_spatial_audit" / "user_annotations_800.csv"
    anno_300_path = base_dir / "human_spatial_audit" / "user_annotations_300.csv"

    def ppn_key_from_underscore_id(image_id: str) -> str:
        # 'PPN1845710797_00000620_1' -> 'PPN1845710797/00000620_1'
        s = str(image_id).split(".")[0]
        parts = s.split("_")
        return f"{parts[0]}/{'_'.join(parts[1:])}"

    # NOTE ON A DROPPED "count_mismatch" SIGNAL: this used to also flag an
    # issue whenever abs(existing_pipeline_agent*3.0 - n_boxes) >= 1.0, i.e.
    # a YOLO-box-count-vs-agent-confidence discrepancy. That compared two
    # things that were never commensurate: user_annotations_800.csv's
    # "n_boxes" column is ~always 0/1 (mean 0.0005 across 2000 rows -- not an
    # actual box count), while existing_pipeline_agent is a [0,1] confidence
    # score, not a box count either. The heuristic "pred_count = confidence *
    # 3.0" was an unvalidated guess. In practice this made count_mismatch fire
    # for nearly every high-confidence image (since confidence*3.0 almost
    # always exceeded the near-zero n_boxes), flipping the sign of the
    # relationship between real agent uncertainty and the resulting "ground
    # truth" label (ROC-AUC as low as 0.006 -- strongly anti-correlated, not
    # just noisy). Removed entirely. Only user_annotations_300.csv's genuine
    # "unsure"/"clarity" annotator judgments are used as ground truth now;
    # the 800-set contributes no independent issue signal and is excluded
    # from gt_matched.
    gt_map = {}
    if anno_300_path.exists():
        a300 = pd.read_csv(anno_300_path)
        for _, r in a300.iterrows():
            img_id = ppn_key_from_underscore_id(r.get("image_id", ""))
            conf = str(r.get("confidence", "")).lower()
            clarity = str(r.get("clarity", "")).lower()
            is_unsure = 1 if "unsure" in conf else 0
            is_degraded = 1 if ("blurry" in clarity or "faded" in clarity) else 0
            gt_map[img_id] = {"unsure": is_unsure, "degraded": is_degraded}

    # Join on full "PPN/shortid" keys (matching agent_comparison_scores.csv's
    # own image_id format), built from the 300-set's PPN-bearing image_id
    # column. (A prior version also fell back to a publication-stripped
    # "short_id" match, which collided across publications and inflated
    # matches from ~1,169 real annotations to 9,630 false ones -- removed.)
    for idx, row in df.iterrows():
        raw_id = str(row.get("image_id", "")).split(".")[0]
        info = gt_map.get(raw_id)
        if info:
            gt_matched[idx] = 1
            if info["unsure"] == 1 or info["degraded"] == 1:
                gt_mask[idx] = 1

    return proxy_labels, pd.Series(gt_mask, index=df.index), pd.Series(gt_matched, index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 disagreement analysis.")
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--input-csv", default="results/multi_agent/agent_comparison_scores.csv")
    parser.add_argument("--output-dir", default="results/multi_agent")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = base / input_csv
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = base / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Compute inter-agent disagreement (std dev across 6 agents per image)
    agent_matrix = df[AGENT_COLS].to_numpy(dtype=float)
    df["inter_agent_disagreement"] = np.std(agent_matrix, axis=1)
    df["inter_agent_range"]        = np.max(agent_matrix, axis=1) - np.min(agent_matrix, axis=1)
    df["inter_agent_mean"]         = np.mean(agent_matrix, axis=1)

    # Issue labels (proxy and independent human ground-truth)
    df["proxy_issue"], df["gt_issue"], df["gt_matched"] = normalize_issue_labels(df, base)

    labels = df["proxy_issue"].to_numpy(dtype=int)

    # Evaluate issue detectors against the PROXY label only.
    # IMPORTANT: proxy_issue is defined by thresholding comparison_fusion_score
    # and monolithic_pipeline_agent directly (see normalize_issue_labels above).
    # monolithic_uncertainty (1 - monolithic_pipeline_agent) and fusion_uncertainty
    # (1 - comparison_fusion_score) are therefore near-tautological predictors of
    # proxy_issue by construction (a monotonic transform of the label's own
    # inputs) -- their ROC-AUC here is guaranteed to be ~1.0 and demonstrates
    # nothing. They are deliberately excluded from this comparison; only
    # predictors that are NOT inputs to proxy_issue's definition are evaluated
    # against it.
    predictors = {
        "inter_agent_disagreement": df["inter_agent_disagreement"].to_numpy(),
        "inter_agent_range":        df["inter_agent_range"].to_numpy(),
    }

    results = {}
    pr_rows  = []
    roc_rows = []

    for name, scores in predictors.items():
        auc_roc, fpr_arr, tpr_arr, thresholds = roc_auc(scores, labels)
        auc_pr,  rec_arr, prec_arr            = pr_auc(scores, labels)

        results[name] = {
            "roc_auc":    round(auc_roc, 4),
            "pr_auc":     round(auc_pr, 4),
            "mean_score": round(float(np.mean(scores)), 4),
            "std_score":  round(float(np.std(scores)), 4),
        }

        # Sample 100 points for curve export
        n_pts = min(100, len(fpr_arr))
        idx   = np.linspace(0, len(fpr_arr) - 1, n_pts, dtype=int)
        for i in idx:
            roc_rows.append({"predictor": name, "fpr": round(float(fpr_arr[i]), 4), "tpr": round(float(tpr_arr[i]), 4)})

        n_pts = min(100, len(rec_arr))
        idx   = np.linspace(0, len(rec_arr) - 1, n_pts, dtype=int)
        for i in idx:
            pr_rows.append({"predictor": name, "recall": round(float(rec_arr[i]), 4), "precision": round(float(prec_arr[i]), 4)})

        print(f"  {name}: ROC-AUC={auc_roc:.4f}  PR-AUC={auc_pr:.4f}")

    # Is inter-agent disagreement the best predictor?
    best_roc = max(results.items(), key=lambda x: x[1]["roc_auc"])
    best_pr  = max(results.items(), key=lambda x: x[1]["pr_auc"])

    # Disagreement distribution stats
    disagree_vals = df["inter_agent_disagreement"].to_numpy()
    disagreement_stats = {
        "mean":   round(float(np.mean(disagree_vals)), 4),
        "median": round(float(np.median(disagree_vals)), 4),
        "std":    round(float(np.std(disagree_vals)), 4),
        "p25":    round(float(np.percentile(disagree_vals, 25)), 4),
        "p75":    round(float(np.percentile(disagree_vals, 75)), 4),
        "p90":    round(float(np.percentile(disagree_vals, 90)), 4),
    }

    # Correlation between disagreement and proxy issue
    corr_matrix = np.corrcoef(df["inter_agent_disagreement"].values, labels)
    disagreement_issue_correlation = round(float(corr_matrix[0, 1]), 4)

    # ── Independent ground-truth evaluation ─────────────────────────────────
    # gt_issue is derived from real human annotations (box-count mismatch vs.
    # YOLO, annotator "unsure", or annotator "blurry"/"faded"), independent of
    # the agent confidence scores being evaluated as predictors -- unlike
    # proxy_issue above, monolithic_uncertainty and fusion_uncertainty are NOT
    # inputs to this label, so this comparison is not circular. Restricted to
    # the n_matched images that actually have a human annotation (exact
    # full-image-id match only; see normalize_issue_labels).
    matched = df["gt_matched"].to_numpy(dtype=bool)
    n_matched = int(matched.sum())
    gt_labels = df.loc[matched, "gt_issue"].to_numpy(dtype=int)
    gt_predictors = {
        "inter_agent_disagreement": df.loc[matched, "inter_agent_disagreement"].to_numpy(),
        "inter_agent_range":        df.loc[matched, "inter_agent_range"].to_numpy(),
        "monolithic_uncertainty":   1.0 - df.loc[matched, "monolithic_pipeline_agent"].to_numpy(),
        "fusion_uncertainty":       1.0 - df.loc[matched, "comparison_fusion_score"].to_numpy(),
    }
    gt_results = {}
    for name, scores in gt_predictors.items():
        auc_roc, _, _, _ = roc_auc(scores, gt_labels)
        auc_pr, _, _ = pr_auc(scores, gt_labels)
        gt_results[name] = {"roc_auc": round(auc_roc, 4), "pr_auc": round(auc_pr, 4)}
        print(f"  [independent GT, n={n_matched}] {name}: ROC-AUC={auc_roc:.4f}  PR-AUC={auc_pr:.4f}")
    best_gt_roc = max(gt_results.items(), key=lambda x: x[1]["roc_auc"])

    analysis = {
        "n_images": int(len(df)),
        "proxy_issue_rate": round(float(df["proxy_issue"].mean()), 4),
        "n_proxy_issues": int(df["proxy_issue"].sum()),
        "predictors_vs_proxy_label": results,
        "proxy_label_caveat": (
            "proxy_issue is a heuristic (bottom-25th-percentile-of-agent-scores) label, "
            "not independently verified. monolithic_uncertainty and fusion_uncertainty "
            "are deliberately excluded from this comparison because they are "
            "near-tautological predictors of proxy_issue by construction."
        ),
        "best_roc_predictor_vs_proxy": best_roc[0],
        "best_roc_auc_vs_proxy":       best_roc[1]["roc_auc"],
        "best_pr_predictor_vs_proxy":  best_pr[0],
        "best_pr_auc_vs_proxy":        best_pr[1]["pr_auc"],
        "disagreement_stats":   disagreement_stats,
        "disagreement_issue_correlation": disagreement_issue_correlation,
        "predictors_vs_independent_gt_label": {
            "n_matched_images": n_matched,
            "gt_label_source": "human_spatial_audit/user_annotations_800.csv + _300.csv, exact image-id match only",
            "results": gt_results,
            "best_predictor": best_gt_roc[0],
            "best_roc_auc": best_gt_roc[1]["roc_auc"],
        },
        "rq2_conclusion": (
            f"On the independent human-annotation ground truth (n={n_matched} matched images), "
            f"the best issue predictor is {best_gt_roc[0]} (ROC-AUC={best_gt_roc[1]['roc_auc']:.4f}). "
            "This is the authoritative RQ2 result; the proxy-label comparison above is a "
            "heuristic sanity check only, not independent evidence."
        ),
        "agents_used": AGENT_COLS,
        "methodology": (
            "Proxy issue labels defined as images in bottom-25th percentile of both "
            "fusion score and monolithic score (conservative dual-low criterion); evaluated "
            "only against predictors that are not inputs to that same label. Independent "
            "ground-truth labels come from real human annotations, matched by exact image id. "
            "Predictors evaluated via ROC-AUC and PR-AUC without sklearn dependency."
        ),
    }

    analysis_path = out_dir / "rq2_disagreement_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as fp:
        json.dump(analysis, fp, indent=2)
    print(f"✅ Wrote {analysis_path}")

    # Save PR and ROC curve data
    pr_df = pd.DataFrame(pr_rows)
    pr_path = out_dir / "rq2_pr_curve.csv"
    pr_df.to_csv(pr_path, index=False)
    print(f"✅ Wrote {pr_path}")

    roc_df = pd.DataFrame(roc_rows)
    roc_path = out_dir / "rq2_roc_curve.csv"
    roc_df.to_csv(roc_path, index=False)
    print(f"✅ Wrote {roc_path}")

    print(f"\n📊 RQ2 Summary:")
    print(f"   Best ROC-AUC predictor: {best_roc[0]} ({best_roc[1]['roc_auc']:.4f})")
    print(f"   Best PR-AUC predictor:  {best_pr[0]} ({best_pr[1]['pr_auc']:.4f})")
    print(f"   Disagreement-Issue Correlation: {disagreement_issue_correlation:.4f}")
    print(f"   {analysis['rq2_conclusion']}")


if __name__ == "__main__":
    main()
