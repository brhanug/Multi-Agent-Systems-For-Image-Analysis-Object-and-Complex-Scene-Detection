#!/usr/bin/env python3
"""Three gaps an external review identified, run on the publication-disjoint split.

P2.8  Predictive ablation of Object / Scene / VLM.
      The existing Tier 1 ablation (Section 4.6) reports how the mean per-image standard
      deviation moves when a model is dropped. That is descriptive, not predictive: it
      never asks whether error DETECTION degrades. This measures leave-one-out ROC-AUC
      against the error label instead, for both the unweighted spread and a learned
      combination.

P2.10 Calibration on genuinely held-out data.
      Section 4.5 reports ECE < 0.001 after Platt scaling, fitted and evaluated on a
      random 50% split of C1. A random within-cohort split lets pages of one volume fall
      on both sides. This refits Platt on the train partition of the corrected,
      PPN-disjoint split and reports ECE on its test partition, where no volume is shared.

P3.13 A consensus + disagreement risk model.
      SAA is consensus, U_SAA is spread. They are different quantities; whether using both
      beats either alone has not been tested. Fitted on train, evaluated on test.

Outputs: results/signal_rebuild/review_gaps_p2_p3.json
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core_pipeline"))
from saa_decision import SAA_THRESHOLD, saa_score

BASE = Path("/data/brhanu/thesis_project")
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT = BASE / "results/signal_rebuild/review_gaps_p2_p3.json"
O, S, V, A = "existing_pipeline_agent", "scene_agent", "vlm_agent", "agreement_agent"
SEED = 20260908


def ece(y, p, bins=10):
    """Standard binned expected calibration error."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1); tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        tot += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return round(float(tot), 6)


def auc(y, s):
    return round(float(roc_auc_score(y, s)), 4) if len(np.unique(y)) > 1 else None


def boot_delta(y, sa, sb, n=2000, seed=SEED):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            d.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
    o = np.array(d)
    return {"delta": round(float(o.mean()), 4),
            "ci95": [round(float(np.percentile(o, 2.5)), 4),
                     round(float(np.percentile(o, 97.5)), 4)],
            "significant": bool(np.percentile(o, 2.5) > 0)}


def fit(cols, tr, te):
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced",
                                         random_state=SEED))
    m.fit(tr[cols].values, tr.err.values)
    return m.predict_proba(te[cols].values)[:, 1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    sp = pd.read_csv(SPLIT)
    ag = pd.read_csv(AGENTS).drop_duplicates("image_id")
    j = sp.merge(ag, left_on="ppn_id", right_on="image_id", how="inner")
    j["saa"] = saa_score(j[O], j[S], j[V])
    j["err"] = ((j.saa >= SAA_THRESHOLD).astype(int) != j.has_scene).astype(int)
    j["U_SAA"] = j[[O, A, S, V]].std(axis=1, ddof=1)
    tr, te = j[j.split == "train"], j[j.split == "test"]
    y = te.err.values

    out = {"experiment": "External-review gaps P2.8, P2.10, P3.13",
           "split": "corrected PPN-disjoint (publication-level) train/test",
           "n_train": int(len(tr)), "n_test": int(len(te)),
           "test_errors": int(y.sum()),
           "train_volumes": int(tr.ppn.nunique()), "test_volumes": int(te.ppn.nunique()),
           "volumes_shared_across_split": int(len(set(tr.ppn) & set(te.ppn)))}

    # ---------- P2.8 predictive leave-one-out ----------
    full3 = [O, S, V]
    abl = {"full (Object, Scene, VLM)": full3,
           "drop Object": [S, V], "drop Scene": [O, V], "drop VLM": [O, S]}
    p28 = {}
    for name, cols in abl.items():
        spread = te[cols].std(axis=1, ddof=1).values if len(cols) > 1 else np.zeros(len(te))
        row = {"signals": cols,
               "auroc_unweighted_spread": auc(y, spread) if len(cols) > 1 else None,
               "auroc_learned": auc(y, fit(cols, tr, te))}
        p28[name] = row
    base = p28["full (Object, Scene, VLM)"]["auroc_learned"]
    for name, row in p28.items():
        row["delta_learned_vs_full"] = (None if row["auroc_learned"] is None or base is None
                                        else round(row["auroc_learned"] - base, 4))
    out["P2_8_predictive_ablation"] = p28
    out["P2_8_reading"] = (
        "Leave-one-out on error-detection ROC-AUC, not on mean spread. A signal whose removal "
        "leaves AUROC unchanged contributes nothing predictive, whatever it does to the spread.")

    # ---------- P2.10 calibration on genuinely held-out data ----------
    pl = LogisticRegression(max_iter=2000, random_state=SEED)
    pl.fit(tr[["saa"]].values, tr.err.values)
    p_te = pl.predict_proba(te[["saa"]].values)[:, 1]
    p_tr = pl.predict_proba(tr[["saa"]].values)[:, 1]
    raw_te = 1 - te.saa.values
    out["P2_10_calibration_held_out"] = {
        "platt_fitted_on": "train partition of the PPN-disjoint split",
        "ece_uncalibrated_test": ece(y, raw_te),
        "ece_platt_train_in_sample": ece(tr.err.values, p_tr),
        "ece_platt_TEST_publication_disjoint": ece(y, p_te),
        "auroc_platt_test": auc(y, p_te),
        "note": ("Section 4.5's ECE < 0.001 is fitted and scored on a random 50% split of C1, "
                 "where pages of one volume can fall on both sides. This row is the same "
                 "procedure evaluated where no volume is shared.")}

    # ---------- P3.13 consensus + disagreement ----------
    models = {"consensus only (SAA)": ["saa"],
              "disagreement only (U_SAA)": ["U_SAA"],
              "consensus + disagreement": ["saa", "U_SAA"]}
    p313 = {k: {"auroc": auc(y, fit(c, tr, te))} for k, c in models.items()}
    both = fit(["saa", "U_SAA"], tr, te); cons = fit(["saa"], tr, te)
    p313["combined_vs_consensus_alone"] = boot_delta(y, both, cons)
    out["P3_13_consensus_plus_disagreement"] = p313

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
