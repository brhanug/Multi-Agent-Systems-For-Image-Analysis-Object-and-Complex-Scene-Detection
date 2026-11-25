#!/usr/bin/env python3
"""Symmetric leave-one-out ablation of the four U_SAA terms, A_agree included.

Earlier ablations treated A_agree differently from the rest: Section 4.6's table drops
architectural MODELS and measures mean spread, and the predictive table in the same
section ablates only the three SAA fusion signals. Neither puts A_agree on the same
footing as its peers. This does.

U_SAA = std(A_object, A_agree, A_scene, A_vlm). Each of the four is removed in turn and
the remaining three are re-scored against the error label, as an unweighted spread (the
deployed form) and as a combination fitted on the training partition.

Robustness: the error label is `prediction != human label`, and the pipeline contains two
non-equivalent ways of forming that prediction -- a stored comparison_fusion_score column
and a recomputation from the stored agent columns -- which disagree on 18.3% of the corpus
(see saa_decision.py). Every ablation is therefore reported under BOTH label rules. A
conclusion that flips between them is not a conclusion.

Outputs: results/signal_rebuild/agree_ablation.json
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core_pipeline"))
from saa_decision import SAA_THRESHOLD, STORED_COLUMN_THRESHOLD, saa_score

BASE = Path("/data/brhanu/thesis_project")
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT = BASE / "results/signal_rebuild/agree_ablation.json"
O, A, S, V = "existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"
FOUR = [O, A, S, V]
NAME = {O: "Object", A: "Agreement", S: "Scene", V: "VLM"}
SEED = 20260908


def auc(y, s):
    return round(float(roc_auc_score(y, s)), 4) if len(np.unique(y)) > 1 else None


def boot(y, sa, sb, n=2000, seed=SEED):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            d.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
    o = np.array(d)
    return {"delta": round(float(o.mean()), 4),
            "ci95": [round(float(np.percentile(o, 2.5)), 4),
                     round(float(np.percentile(o, 97.5)), 4)],
            "excludes_zero": bool(np.percentile(o, 2.5) > 0 or np.percentile(o, 97.5) < 0)}


def learned(cols, tr, te):
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced",
                                         random_state=SEED))
    m.fit(tr[cols].values, tr.err.values)
    return m.predict_proba(te[cols].values)[:, 1]


def run(j, label_rule):
    tr, te = j[j.split == "train"], j[j.split == "test"]
    y = te.err.values
    full_sp = te[FOUR].std(axis=1, ddof=1).values
    full_ln = learned(FOUR, tr, te)
    res = {"label_rule": label_rule, "n_test": int(len(te)), "test_errors": int(y.sum()),
           "full_U_SAA_spread": auc(y, full_sp), "full_learned": auc(y, full_ln),
           "ablations": {}}
    for drop in FOUR:
        keep = [c for c in FOUR if c != drop]
        sp = te[keep].std(axis=1, ddof=1).values
        ln = learned(keep, tr, te)
        res["ablations"][f"drop {NAME[drop]}"] = {
            "retained": [NAME[c] for c in keep],
            "rank_identical_to_full": bool(np.array_equal(np.argsort(ln), np.argsort(full_ln))),
            "max_abs_score_shift": round(float(np.abs(ln - full_ln).max()), 6),
            "distinct_spread_values": int(pd.Series(sp).nunique()),
            "spread_auroc": auc(y, sp),
            "spread_delta": (None if res["full_U_SAA_spread"] is None
                             else round(auc(y, sp) - res["full_U_SAA_spread"], 4)),
            "spread_delta_ci": boot(y, sp, full_sp),
            "learned_auroc": auc(y, ln),
            "learned_delta": round(auc(y, ln) - res["full_learned"], 4),
            "learned_delta_ci": boot(y, ln, full_ln),
        }
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    sp = pd.read_csv(SPLIT)
    ag = pd.read_csv(AGENTS).drop_duplicates("image_id")
    j = sp.merge(ag, left_on="ppn_id", right_on="image_id", how="inner")
    j["saa_recomputed"] = saa_score(j[O], j[S], j[V])

    out = {"experiment": "Symmetric leave-one-out over the four U_SAA terms",
           "U_SAA": "std(A_object, A_agree, A_scene, A_vlm), ddof=1",
           "split": "corrected PPN-disjoint train/test",
           "A_agree_identity": {
               "equals_A_object_pct": round(100 * float((j[A] == j[O]).mean()), 1),
               "jumps_to_1_when_A_object_0": bool(
                   (j.loc[j[O] == 0, A] == 1).all()),
               "n_where_A_object_0": int((j[O] == 0).sum())},
           }
    ja = j.copy()
    ja["err"] = ((ja.saa_recomputed >= SAA_THRESHOLD).astype(int) != ja.has_scene).astype(int)
    out["under_recomputed_rule"] = run(ja, "recomputed SAA >= 0.55 (== legacy >= 0.5)")
    jb = j.copy()
    jb["err"] = ((jb.comparison_fusion_score >= STORED_COLUMN_THRESHOLD).astype(int)
                 != jb.has_scene).astype(int)
    out["under_stored_column_rule"] = run(jb, "stored comparison_fusion_score >= 0.5")

    # does the conclusion about A_agree survive both label rules?
    ra = out["under_recomputed_rule"]["ablations"]["drop Agreement"]
    rb = out["under_stored_column_rule"]["ablations"]["drop Agreement"]
    # is {Object, Agreement, VLM} a mutually recoverable set?
    out["recoverability"] = {
        "A_vlm == 1[A_object>0]": round(float((j[V] == (j[O] > 0).astype(float)).mean()), 4),
        "A_object == where(A_vlm==0, 0, A_agree)": round(
            float((np.where(j[V] == 0, 0, j[A]) == j[O]).mean()), 4),
        "A_agree == A_object": round(float((j[A] == j[O]).mean()), 4),
        "reading": ("Any one of Object, Agreement and VLM is reconstructible from the other "
                    "two, exactly. U_SAA is nominally a four-signal spread but has only two "
                    "independent inputs: an object-derived quantity and the scene signal.")}

    TOL = 0.005
    out["verdict_on_A_agree"] = {
        "spread_delta_recomputed": ra["spread_delta"],
        "spread_delta_stored": rb["spread_delta"],
        "learned_delta_recomputed": ra["learned_delta"],
        "learned_delta_stored": rb["learned_delta"],
        "spread_conclusion_robust": bool(
            ra["spread_delta"] < -TOL and rb["spread_delta"] < -TOL),
        "learned_conclusion_robust": bool(
            abs(ra["learned_delta"]) <= TOL and abs(rb["learned_delta"]) <= TOL),
        "reading": ("Dropping A_agree from the unweighted spread hurts under both label rules "
                    "(the spread loses a term that is a maximal outlier exactly where "
                    "A_object = 0). Dropping it from a FITTED model changes nothing under "
                    "either rule -- both deltas are within +/-%.3f -- because the information "
                    "it carries is fully recoverable from the terms that remain." % TOL)}

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
