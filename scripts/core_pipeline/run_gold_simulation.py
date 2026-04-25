#!/usr/bin/env python3
"""
Gold Simulation Subset Generator (RQ1/RQ2 support).

Simulates a 'gold-standard' evaluation subset using consensus-of-consensus:
images where all primary agents agree strongly (all > threshold) are treated
as confident positives; images where all agents agree low are confident negatives.
This produces a measured-quality subset without manual annotation.

Outputs:
  results/multi_agent/gold_simulation_subset.csv
  results/multi_agent/gold_simulation_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_AGENTS = [
    "existing_pipeline_agent",
    "agreement_agent",
    "vlm_agent",
    "scene_agent",
]

HIGH_THRESH = 0.65   # all agents above this -> confident positive
LOW_THRESH  = 0.30   # all agents below this -> confident negative
MIN_SUBSET  = 200    # target minimum subset size


def assign_gold_label(row: pd.Series, high: float, low: float) -> int | None:
    """Return 1 (positive), 0 (negative), or None (uncertain)."""
    vals = [float(row[a]) for a in PRIMARY_AGENTS if a in row.index]
    if not vals:
        return None
    if all(v >= high for v in vals):
        return 1
    if all(v <= low for v in vals):
        return 0
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate gold simulation subset.")
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--input-csv", default="results/multi_agent/agent_comparison_scores.csv")
    parser.add_argument("--output-dir", default="results/multi_agent")
    parser.add_argument("--high-thresh", type=float, default=HIGH_THRESH)
    parser.add_argument("--low-thresh", type=float, default=LOW_THRESH)
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

    # Assign gold simulation labels
    df["gold_label"] = df.apply(lambda r: assign_gold_label(r, args.high_thresh, args.low_thresh), axis=1)

    gold_df = df[df["gold_label"].notna()].copy()
    gold_df["gold_label"] = gold_df["gold_label"].astype(int)

    # If subset is too small, relax thresholds iteratively
    if len(gold_df) < MIN_SUBSET:
        for relaxation in [0.05, 0.10, 0.15, 0.20]:
            h = args.high_thresh - relaxation
            l = args.low_thresh + relaxation
            df["gold_label"] = df.apply(lambda r, _h=h, _l=l: assign_gold_label(r, _h, _l), axis=1)
            gold_df = df[df["gold_label"].notna()].copy()
            gold_df["gold_label"] = gold_df["gold_label"].astype(int)
            if len(gold_df) >= MIN_SUBSET:
                break

    # Compute evaluation metrics on gold subset
    gold_df["monolithic_pred"] = (gold_df["monolithic_pipeline_agent"] > 0.5).astype(int)
    gold_df["fusion_pred"]     = (gold_df["comparison_fusion_score"]   > 0.5).astype(int)

    def accuracy(pred: pd.Series, label: pd.Series) -> float:
        return float((pred == label).mean())

    def precision(pred: pd.Series, label: pd.Series) -> float:
        tp = ((pred == 1) & (label == 1)).sum()
        pp = (pred == 1).sum()
        return float(tp / pp) if pp > 0 else 0.0

    def recall(pred: pd.Series, label: pd.Series) -> float:
        tp = ((pred == 1) & (label == 1)).sum()
        ap = (label == 1).sum()
        return float(tp / ap) if ap > 0 else 0.0

    def f1(pred: pd.Series, label: pd.Series) -> float:
        p = precision(pred, label)
        r = recall(pred, label)
        return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    mono_acc  = accuracy(gold_df["monolithic_pred"], gold_df["gold_label"])
    fus_acc   = accuracy(gold_df["fusion_pred"],     gold_df["gold_label"])
    mono_prec = precision(gold_df["monolithic_pred"], gold_df["gold_label"])
    fus_prec  = precision(gold_df["fusion_pred"],     gold_df["gold_label"])
    mono_rec  = recall(gold_df["monolithic_pred"], gold_df["gold_label"])
    fus_rec   = recall(gold_df["fusion_pred"],     gold_df["gold_label"])
    mono_f1   = f1(gold_df["monolithic_pred"], gold_df["gold_label"])
    fus_f1    = f1(gold_df["fusion_pred"],     gold_df["gold_label"])

    pos_rate = float(gold_df["gold_label"].mean())
    n_pos    = int(gold_df["gold_label"].sum())
    n_neg    = int((gold_df["gold_label"] == 0).sum())

    print(f"Gold subset size: {len(gold_df)} / {len(df)} ({100 * len(gold_df) / len(df):.1f}%)")
    print(f"  Positives: {n_pos}  |  Negatives: {n_neg}  |  Rate: {pos_rate:.3f}")
    print(f"  Monolithic -> Acc: {mono_acc:.3f}  Prec: {mono_prec:.3f}  Rec: {mono_rec:.3f}  F1: {mono_f1:.3f}")
    print(f"  Fusion     -> Acc: {fus_acc:.3f}  Prec: {fus_prec:.3f}  Rec: {fus_rec:.3f}  F1: {fus_f1:.3f}")

    # Save subset CSV
    subset_path = out_dir / "gold_simulation_subset.csv"
    gold_df.to_csv(subset_path, index=False)
    print(f"✅ Wrote {subset_path}")

    # Save report
    report = {
        "total_images": int(len(df)),
        "gold_subset_size": int(len(gold_df)),
        "gold_coverage_pct": round(100 * len(gold_df) / len(df), 2),
        "positive_rate": round(pos_rate, 4),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "high_threshold_used": args.high_thresh,
        "low_threshold_used": args.low_thresh,
        "primary_agents": PRIMARY_AGENTS,
        "monolithic_pipeline_agent": {
            "accuracy": round(mono_acc, 4),
            "precision": round(mono_prec, 4),
            "recall": round(mono_rec, 4),
            "f1": round(mono_f1, 4),
        },
        "comparison_fusion_score": {
            "accuracy": round(fus_acc, 4),
            "precision": round(fus_prec, 4),
            "recall": round(fus_rec, 4),
            "f1": round(fus_f1, 4),
        },
        "delta_accuracy_fusion_vs_mono": round(fus_acc - mono_acc, 4),
        "delta_f1_fusion_vs_mono": round(fus_f1 - mono_f1, 4),
        "methodology": (
            "Gold simulation labels derived from strong consensus across all primary agents "
            "(all >= high_threshold -> positive; all <= low_threshold -> negative). "
            "This approximates a measured gold set without manual annotation."
        ),
    }

    report_path = out_dir / "gold_simulation_report.json"
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"✅ Wrote {report_path}")


if __name__ == "__main__":
    main()
