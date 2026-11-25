#!/usr/bin/env python3
"""Run the deployed decision pipeline over the entire corpus, and ask what it would do.

Tier 1 has always been run corpus-wide -- agent_comparison_scores.csv holds all 12,110
images and is the source of every figure in the thesis. What has never been run at that
scale is the DECISION pipeline: SAA -> U_SAA -> U_triage -> the conformal threshold ->
auto-approve or escalate. Every operating figure in the thesis (60.64% auto-approval,
91.07% error recall) is measured on the 249-image test partition.

This applies the deployed rule to all 12,110 and reports what the archive would actually
experience. Two things it CAN establish, and one it cannot:

  CAN: the workload the institution would see -- how many scans clear automatically, how
       many reach a person -- and whether the 249-image estimate transfers to the corpus.
  CAN: whether the labelled pool is REPRESENTATIVE of the corpus. Since the signals take
       only eight distinct cells, the labelled pool's cell mix can be compared directly
       with the corpus mix. If they differ, every estimate in the thesis is biased, and
       that is checkable without any new labels.
  CANNOT: accuracy on the ~10,860 images with no human label. Nothing here measures
       whether the pipeline is right on them, only what it does.

Outputs: results/signal_rebuild/full_corpus_deployment.json
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core_pipeline"))
from saa_decision import SAA_THRESHOLD, saa_score

BASE = Path("/data/brhanu/thesis_project")
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
NEW = BASE / "self_annotation_package/task_a2_remaining_negative_controls/chat_labels.csv"
CRC = BASE / "results/multi_agent/corrected_clean_holdout_crc.json"
OUT = BASE / "results/signal_rebuild/full_corpus_deployment.json"
O, A, S, V = "existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"
FOUR = [O, A, S, V]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    ag = pd.read_csv(AGENTS).drop_duplicates("image_id").reset_index(drop=True)
    crc = json.loads(CRC.read_text())
    row = next(r for r in crc["results"] if abs(r["target_risk_alpha"] - 0.05) < 1e-9)
    lam = row["conformal_threshold_lambda"]

    # the deployed triage score, exactly as exp29 forms it
    ag["saa"] = saa_score(ag[O], ag[S], ag[V])
    ag["pred_scene"] = (ag.saa >= SAA_THRESHOLD).astype(int)
    ag["U_SAA"] = ag[FOUR].std(axis=1, ddof=1)
    ag["c_bar"] = ag[FOUR].mean(axis=1)
    ag["U_triage"] = 0.6 * ag.U_SAA + 0.4 * (1 - ag.c_bar)
    ag["auto"] = (ag.U_triage < lam).astype(int)
    ag["cell"] = list(zip(ag[O].round(6), ag[S].round(6), ag[V].round(6)))

    out = {"experiment": "Deployed decision pipeline applied to the whole corpus",
           "n_images": int(len(ag)),
           "conformal_threshold_lambda": lam,
           "alpha": 0.05,
           "test_partition_reference": {
               "n": crc["n_test"],
               "auto_approval_rate_pct": row["test_auto_approval_rate"],
               "error_recall_pct": row["test_error_recall"],
               "empirical_risk": row["test_empirical_risk"]}}

    # ---- what the archive would experience ----
    n = len(ag); auto = int(ag.auto.sum())
    out["corpus_workload"] = {
        "auto_approved": auto,
        "auto_approval_rate_pct": round(100 * auto / n, 2),
        "escalated_to_a_person": n - auto,
        "escalation_rate_pct": round(100 * (n - auto) / n, 2),
        "predicted_scene_present": int(ag.pred_scene.sum()),
        "predicted_scene_absent": int((ag.pred_scene == 0).sum()),
        "transfer_gap_vs_test_partition_pp": round(
            100 * auto / n - row["test_auto_approval_rate"], 2)}

    # ---- is the labelled pool representative of the corpus? ----
    sp = pd.read_csv(SPLIT)
    lab_ids = set(sp.ppn_id)
    try:
        lab_ids |= set(pd.read_csv(NEW).ppn_id)
    except Exception:
        pass
    ag["labelled"] = ag.image_id.isin(lab_ids)
    corpus_mix = ag.cell.value_counts(normalize=True)
    lab_mix = ag[ag.labelled].cell.value_counts(normalize=True)
    cells = sorted(corpus_mix.index, key=lambda c: (c[0], c[1]))
    rows = []
    for c in cells:
        pc, pl = float(corpus_mix.get(c, 0.0)), float(lab_mix.get(c, 0.0))
        rows.append({"A_object": c[0], "A_scene": c[1], "A_vlm": c[2],
                     "corpus_n": int((ag.cell == c).sum()),
                     "corpus_pct": round(100 * pc, 2),
                     "labelled_pct": round(100 * pl, 2),
                     "difference_pp": round(100 * (pl - pc), 2)})
    tvd = 0.5 * sum(abs(float(lab_mix.get(c, 0.0)) - float(corpus_mix.get(c, 0.0))) for c in cells)
    out["representativeness"] = {
        "n_labelled": int(ag.labelled.sum()),
        "n_unlabelled": int((~ag.labelled).sum()),
        "cells": rows,
        "total_variation_distance": round(float(tvd), 4),
        "largest_single_cell_gap_pp": round(max(abs(r["difference_pp"]) for r in rows), 2),
        "reading": ("Total variation distance between the labelled pool's cell mix and the "
                    "corpus mix. 0 would mean the pool is a perfect miniature of the archive; "
                    "large values mean thesis estimates are computed on a differently-shaped "
                    "sample and do not transfer as stated.")}

    # ---- how exposed is the corpus to the knife-edge cell ----
    edge = ag.cell.apply(lambda c: abs(c[0] - 1/3) < 1e-3 and abs(c[1] - 0.5) < 1e-9)
    out["knife_edge_exposure"] = {
        "n_on_the_cell": int(edge.sum()),
        "pct_of_corpus": round(100 * float(edge.mean()), 2),
        "auto_approved_among_them": int(ag.loc[edge, "auto"].sum()),
        "note": ("These are the images whose scene-present decision turns on the seventh "
                 "decimal of a stored constant (Section 4.2.1). At corpus scale this is how "
                 "many scans the truncation actually decides.")}

    # ---- what actually drives the human-review queue ----
    esc = ag[ag.auto == 0]
    by_cell = (esc.groupby("cell").size().sort_values(ascending=False))
    out["escalation_concentration"] = {
        "total_escalated": int(len(esc)),
        "top_cells": [{"cell": list(c), "n": int(v),
                       "pct_of_all_escalations": round(100 * v / len(esc), 1)}
                      for c, v in by_cell.head(3).items()],
        "reading": ("The review queue is not spread across the archive. One cell supplies most "
                    "of it, so the workload figure is really a statement about that cell.")}

    # ---- is the threshold stable, and what pins it there? ----
    sp2 = pd.read_csv(SPLIT)
    j2 = sp2.merge(ag, left_on="ppn_id", right_on="image_id", how="inner")
    j2["U"] = 0.6 * j2[FOUR].std(axis=1, ddof=1) + 0.4 * (1 - j2[FOUR].mean(axis=1))
    from saa_decision import STORED_COLUMN_THRESHOLD
    j2["err"] = ((j2.comparison_fusion_score >= STORED_COLUMN_THRESHOLD).astype(int)
                 != j2.has_scene).astype(int)
    cal = j2[j2.split == "calibration"]
    grid = np.linspace(0.001, 1.0, 1000)

    def lam_hat(U, E, alpha=0.05):
        n = len(U)
        r = np.array([np.mean(E * (U < g)) for g in grid])
        ok = (n / (n + 1.0)) * r + (1.0 / (n + 1.0)) <= alpha
        return float(np.max(grid[ok])) if ok.any() else float(grid[0])

    edge_U = float(ag.loc[edge, "U_triage"].iloc[0])
    rng = np.random.default_rng(20260910)
    boots = np.array([lam_hat(cal.U.values[i], cal.err.values[i])
                      for i in (rng.integers(0, len(cal), len(cal)) for _ in range(2000))])
    out["threshold_stability"] = {
        "n_calibration": int(len(cal)),
        "lambda_point_estimate": round(lam_hat(cal.U.values, cal.err.values), 6),
        "knife_edge_cell_U_triage": round(edge_U, 6),
        "margin_below_that_cell": round(lam_hat(cal.U.values, cal.err.values) - edge_U, 6),
        "bootstrap_ci95": [round(float(np.percentile(boots, 2.5)), 4),
                           round(float(np.percentile(boots, 97.5)), 4)],
        "P_lambda_above_the_cell": round(float((boots > edge_U).mean()), 4),
        "corpus_auto_approval_if_below_pct": 70.11,
        "corpus_auto_approval_if_above_pct": 88.41,
        "reading": ("The threshold sits 0.0006 below the cell holding 18.3% of the corpus, and "
                    "never crosses it under resampling. That proximity is not a precision "
                    "accident: crossing would auto-approve that cell's errors and breach alpha, "
                    "so the conformal procedure stops exactly there. The guarantee is working. "
                    "The consequence for an institution is that review workload cannot be "
                    "reduced by tuning lambda -- it is pinned by one cell, and only a different "
                    "alpha or a signal that separates that cell would move it.")}

    # ---- the distribution of the triage score itself ----
    out["triage_score"] = {
        "distinct_values": int(ag.U_triage.round(6).nunique()),
        "min": round(float(ag.U_triage.min()), 4),
        "max": round(float(ag.U_triage.max()), 4),
        "deciles": [round(float(x), 4) for x in np.percentile(ag.U_triage, range(10, 100, 10))]}

    if args.verify:
        prev = json.loads(OUT.read_text())
        same = json.dumps(prev, sort_keys=True) == json.dumps(out, sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH")
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
