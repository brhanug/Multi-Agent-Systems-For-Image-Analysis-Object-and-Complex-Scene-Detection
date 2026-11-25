#!/usr/bin/env python3
"""What happens if A_agree is removed and the uncertainty values are learned?

Two moves are asked about together:
  (1) drop A_agree, the derived term that equals A_object on 88.4% of images and
      jumps to 1 on the rest;
  (2) stop using an unweighted standard deviation and learn the combination instead.

The eight-cell structure makes this answerable exactly rather than by trying models.
A_object takes 4 values, A_scene 2, and A_vlm = 1[A_object > 0], so the three measured
signals realise only 8 distinct triples. ANY function of them -- any weighting, any
nonlinearity, any classifier, learned however well -- is constant within a cell and can
therefore take at most 8 distinct values. The best possible such function ranks the
cells by their true error rate. That ranking's AUROC is a hard CEILING no learned
uncertainty score over these signals can exceed.

Reported:
  ceiling_in_sample  - cells ranked by error rate measured on the same data (optimistic)
  ceiling_honest     - cell error rates estimated on train, applied to test
  the deployed 4-term std, the 3-term std with A_agree dropped, a learned logistic
  combination, and the plain confidence baseline, all on the same partition.

Outputs: results/signal_rebuild/drop_agree_learn_values.json
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
BASE = Path("/data/brhanu/thesis_project")
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT = BASE / "results/signal_rebuild/drop_agree_learn_values.json"

O, S, V, A = "existing_pipeline_agent", "scene_agent", "vlm_agent", "agreement_agent"
SEED = 20260908
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core_pipeline"))
from saa_decision import SAA_THRESHOLD, saa_score


def load():
    sp = pd.read_csv(SPLIT)
    ag = pd.read_csv(AGENTS).drop_duplicates("image_id")
    j = sp.merge(ag, left_on="ppn_id", right_on="image_id", how="inner")
    j["saa"] = saa_score(j[O], j[S], j[V])
    j["err"] = ((j.saa >= SAA_THRESHOLD).astype(int) != j.has_scene).astype(int)
    j["U4"] = j[[O, A, S, V]].std(axis=1, ddof=1)      # deployed
    j["U3"] = j[[O, S, V]].std(axis=1, ddof=1)          # A_agree dropped
    j["cell"] = list(zip(j[O].round(6), j[S].round(6), j[V].round(6)))
    return j


def auc(y, s):
    return round(float(roc_auc_score(y, s)), 4) if len(np.unique(y)) > 1 else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    j = load()
    tr, te = j[j.split == "train"], j[j.split == "test"]

    # per-cell error rates
    rate_tr = tr.groupby("cell").err.mean()
    tbl = (j.groupby("cell").agg(n=("err", "size"), errors=("err", "sum")).reset_index())
    tbl["err_rate"] = (tbl.errors / tbl.n).round(4)

    out = {"experiment": "Drop A_agree and learn the values -- what is actually reachable",
           "why_a_ceiling_exists": (
               "A_object has 4 values, A_scene 2, A_vlm = 1[A_object>0]. The three measured "
               "signals realise 8 distinct triples, so ANY function of them -- any weighting, "
               "any nonlinearity, any learned model -- is constant within a cell and takes at "
               "most 8 values. Ranking cells by true error rate is the best such function, and "
               "its AUROC bounds every learned uncertainty score over these signals."),
           "n_cells": int(len(tbl)),
           "cells": [{"A_object": c[0], "A_scene": c[1], "A_vlm": c[2],
                      "n": int(n), "errors": int(e), "error_rate": float(r)}
                     for c, n, e, r in zip(tbl.cell, tbl.n, tbl.errors, tbl.err_rate)],
           "n_train": int(len(tr)), "n_test": int(len(te)),
           "test_errors": int(te.err.sum())}

    y = te.err.values
    # ---- the ceiling ----
    ceil_in = te.cell.map(te.groupby("cell").err.mean()).values
    ceil_hon = te.cell.map(rate_tr).fillna(tr.err.mean()).values
    # ---- learned combination over the three measured signals ----
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    lr.fit(tr[[O, S, V]].values, tr.err.values)
    learned = lr.predict_proba(te[[O, S, V]].values)[:, 1]
    # ---- learned combination that still includes A_agree ----
    lr4 = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    lr4.fit(tr[[O, A, S, V]].values, tr.err.values)
    learned4 = lr4.predict_proba(te[[O, A, S, V]].values)[:, 1]

    # brute-force proof: no ordering of the 8 cells beats the error-rate ordering
    import itertools as _it
    cells_u = list(te.cell.unique())
    best_perm = -1.0
    for perm in _it.permutations(range(len(cells_u))):
        rank = dict(zip(cells_u, perm))
        v = auc(y, te.cell.map(rank).values)
        if v is not None and v > best_perm:
            best_perm = v
    out["ceiling_proof"] = {
        "method": "exhaustive over all orderings of the 8 cells",
        "n_orderings_checked": int(np.math.factorial(len(cells_u))) if hasattr(np, "math") else 40320,
        "best_auroc_any_ordering": round(best_perm, 4),
        "matches_error_rate_ordering": bool(abs(best_perm - auc(y, ceil_in)) < 1e-9)}

    out["auroc_on_test"] = {
        "CEILING_max_over_any_function_of_these_signals": auc(y, ceil_in),
        "train_estimated_cell_ranking": auc(y, ceil_hon),
        "learned_logistic_3_signals_no_Aagree": auc(y, learned),
        "learned_logistic_4_signals_with_Aagree": auc(y, learned4),
        "deployed_U_SAA_4term_std": auc(y, te.U4.values),
        "U_3term_std_Aagree_dropped": auc(y, te.U3.values),
        "confidence_baseline_1_minus_SAA": auc(y, 1 - te.saa.values),
    }
    out["learned_coefficients_3_signals"] = {
        k: round(float(v), 4) for k, v in zip(["A_object", "A_scene", "A_vlm"], lr.coef_[0])}
    a = out["auroc_on_test"]
    ceil = a["CEILING_max_over_any_function_of_these_signals"]
    conf = a["confidence_baseline_1_minus_SAA"]
    out["headroom"] = {
        "ceiling": ceil, "confidence_baseline": conf,
        "max_possible_gain_over_confidence": round(ceil - conf, 4),
        "best_learned_score": a["learned_logistic_3_signals_no_Aagree"],
        "best_learned_minus_confidence": round(a["learned_logistic_3_signals_no_Aagree"] - conf, 4),
        "note": ("The ceiling bounds every possible uncertainty score built from these three "
                 "signals. Confidence already sits within this margin of it, so no learned "
                 "disagreement score can beat confidence by more than that amount, however it "
                 "is constructed.")}

    # paired bootstrap: learned-without-Aagree vs confidence
    rng = np.random.default_rng(SEED); dif = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            dif.append(roc_auc_score(y[i], learned[i]) - roc_auc_score(y[i], (1 - te.saa.values)[i]))
    o = np.array(dif)
    out["learned_vs_confidence_paired_bootstrap"] = {
        "delta_auroc": round(float(o.mean()), 4),
        "ci95": [round(float(np.percentile(o, 2.5)), 4), round(float(np.percentile(o, 97.5)), 4)],
        "significant": bool(np.percentile(o, 2.5) > 0)}

    out["reading"] = (
        f"Every learned score over these signals is capped at {ceil}. "
        f"Dropping A_agree moves the "
        f"unweighted std from {a['deployed_U_SAA_4term_std']} to {a['U_3term_std_Aagree_dropped']}; "
        f"learning the combination gives {a['learned_logistic_3_signals_no_Aagree']}. The plain "
        f"confidence baseline scores {a['confidence_baseline_1_minus_SAA']} on the same partition. "
        f"The gap between the best learned score and the ceiling is what learning can still buy; "
        f"the gap between the ceiling and 1.0 is what these three signals cannot express at all.")

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
