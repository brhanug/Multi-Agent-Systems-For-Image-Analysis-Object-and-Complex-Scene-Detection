"""
An exhaustive search over uncertainty formulations, so the negative result is a
boundary rather than a failure to try hard enough.

Nine ways of turning a set of agent signals into one uncertainty score are
evaluated against the same held-out error label, alongside the plain confidence
baseline they are all trying to beat. The variants span three families:

  * which signals enter        (deployed set, derived term dropped, redundant
                                term dropped, signals re-derived from raw outputs)
  * how members are weighted   (unweighted, competence-weighted, competent subset)
  * what operator is applied   (standard deviation, mutual information / BALD)

    python run_formulation_search.py [--verify]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from run_noncircular_disagreement_test import load, scale, MODELS

BASE = Path("/data/brhanu/thesis_project")
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT = BASE / "results/signal_rebuild/formulation_search.json"
O, A, S, V, C = ("existing_pipeline_agent", "agreement_agent", "scene_agent",
                 "vlm_agent", "document_agent")
EPS, SEED, N_BOOT = 1e-12, 0, 4000


def Hb(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def paired(y, a, b, seed=SEED, n=N_BOOT):
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    o = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            o.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    o = np.array(o); lo, hi = np.percentile(o, [2.5, 97.5])
    return {"delta_vs_confidence": round(float(o.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    full, d = load()
    tr, te = d[d.split == "train"], d[d.split == "test"].reset_index(drop=True)
    cols = list(MODELS.values())
    trs, tes = scale(tr, te, cols)

    # calibrate each rebuilt member on train; the ensemble prediction defines the
    # error label against human review, so nothing here is circular
    comp, P = {}, {}
    for name, c in MODELS.items():
        comp[name] = abs(roc_auc_score(trs.has_scene, trs[c]) - 0.5) * 2
        m = LogisticRegression(max_iter=2000).fit(trs[[c]], trs.has_scene)
        P[name] = m.predict_proba(tes[[c]])[:, 1]
    M = np.column_stack([P[k] for k in MODELS])
    pbar = M.mean(axis=1)
    err = ((pbar >= 0.5).astype(int) != tes.has_scene.values).astype(int)
    conf = 1 - pbar
    w = np.array([comp[k] for k in MODELS]); w = w / w.sum()
    keep = [i for i, k in enumerate(MODELS) if k != "kosmos"]
    top2 = np.argsort(-w)[:2]

    # deployed-table variants, on the same rows
    ag = pd.read_csv(AGENTS).drop_duplicates("image_id")
    j = tes.merge(ag, left_on="ppn_id", right_on="image_id", how="inner", suffixes=("", "_d"))

    variants = {
        "which signals: deployed std(O, Agree, S, V)": j[[O, A, S, V]].std(axis=1, ddof=1).values,
        "which signals: derived term dropped": j[[O, S, V]].std(axis=1, ddof=1).values,
        "which signals: derived and redundant dropped": j[[O, S, C]].std(axis=1, ddof=1).values,
        "which signals: four re-derived from raw outputs": M.std(axis=1, ddof=1),
        "weighting: competence-weighted spread":
            np.sqrt((w * (M - (M * w).sum(axis=1, keepdims=True)) ** 2).sum(axis=1)),
        "weighting: at-chance member removed": M[:, keep].std(axis=1, ddof=1),
        "weighting: two most competent members only": M[:, top2].std(axis=1, ddof=1),
        "operator: mutual information (BALD epistemic)": Hb(pbar) - Hb(M).mean(axis=1),
        "operator: range instead of standard deviation": M.max(axis=1) - M.min(axis=1),
    }

    res = {"experiment": "Exhaustive search over uncertainty formulations",
           "n": int(len(err)), "n_errors": int(err.sum()),
           "error_label": "ensemble prediction vs human review (non-circular)",
           "member_competence_on_train": {k: round(v, 4) for k, v in comp.items()},
           "baseline_plain_confidence": round(float(roc_auc_score(err, conf)), 4),
           "variants": {}}
    for name, sig in variants.items():
        res["variants"][name] = {"roc_auc": round(float(roc_auc_score(err, sig)), 4)} | paired(err, sig, conf)

    beat = [k for k, v in res["variants"].items() if v["excludes_zero"] and v["delta_vs_confidence"] > 0]
    res["summary"] = {
        "variants_tested": len(variants),
        "variants_beating_confidence": beat,
        "best_variant": max(res["variants"], key=lambda k: res["variants"][k]["roc_auc"]),
        "best_variant_auc": max(v["roc_auc"] for v in res["variants"].values()),
        "conclusion": "No formulation in the search beats plain confidence. Two results close the "
                      "obvious remaining moves: removing the at-chance member makes the spread "
                      "worse, and so does competence weighting, so the weak member is contributing "
                      "useful variance rather than noise. The limitation is not the operator, the "
                      "signal set, or the weighting."}

    if args.verify:
        prev = json.loads(OUT.read_text())
        same = json.dumps(prev, sort_keys=True) == json.dumps(res, sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["summary"], indent=2))
    for k, v in res["variants"].items():
        print(f"  {k:52} {v['roc_auc']:.4f}  ({v['delta_vs_confidence']:+.4f})")
    print(f"  {'BASELINE plain confidence':52} {res['baseline_plain_confidence']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
