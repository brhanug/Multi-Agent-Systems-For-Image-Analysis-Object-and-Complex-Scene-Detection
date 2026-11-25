"""
Does disagreement beat confidence when disagreement is measured properly?

The deployed proxy U_SAA correlates r=0.91 with plain confidence -- it shares 82%
of its variance with the thing it is supposed to improve on, which is why it
cannot outperform it. This replaces the standard deviation with the mutual
information / BALD decomposition, which separates the two by construction:

    Total      = H[ mean_m p_m ]          predictive entropy
    Aleatoric  = mean_m H[ p_m ]          uncertainty inside each member
    Epistemic  = Total - Aleatoric        disagreement, with confidence removed

Each member is first calibrated on the training partition so that p_m is a
probability of scene presence rather than an arbitrary score; entropies of
uncalibrated scores are not comparable. The error label is human review, so the
evaluation is not circular.

    python run_bald_decomposition.py [--verify]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from run_noncircular_disagreement_test import load, scale, MODELS

BASE = Path("/data/brhanu/thesis_project")
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT = BASE / "results/signal_rebuild/bald_decomposition.json"
A4 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
EPS, SEED, N_BOOT = 1e-12, 0, 4000


def Hb(p):
    """Binary entropy, nats."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def auc(y, s):
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def paired(y, a, b, seed=SEED, n=N_BOOT):
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    o = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            o.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    o = np.array(o); lo, hi = np.percentile(o, [2.5, 97.5])
    return {"delta": round(float(o.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    full, d = load()
    tr, te = d[d.split == "train"], d[d.split == "test"].reset_index(drop=True)
    cols = list(MODELS.values())
    trs, tes = scale(tr, te, cols)

    # --- calibrate each member separately on train: raw signal -> P(scene) ---
    P_tr, P_te = {}, {}
    for name, c in MODELS.items():
        m = LogisticRegression(max_iter=2000).fit(trs[[c]], trs.has_scene)
        P_tr[name] = m.predict_proba(trs[[c]])[:, 1]
        P_te[name] = m.predict_proba(tes[[c]])[:, 1]

    Pte = np.column_stack([P_te[k] for k in MODELS])
    pbar = Pte.mean(axis=1)
    total = Hb(pbar)
    aleatoric = Hb(Pte).mean(axis=1)
    epistemic = total - aleatoric                       # mutual information

    # --- non-circular error label: ensemble prediction vs human review ---
    err = ((pbar >= 0.5).astype(int) != tes.has_scene.values).astype(int)

    # --- the deployed proxy on the same rows, for comparison ---
    ag = pd.read_csv(AGENTS).drop_duplicates("image_id")
    j = tes.merge(ag, left_on="ppn_id", right_on="image_id", how="inner", suffixes=("", "_d"))
    u_saa = j[A4].std(axis=1, ddof=1).values
    conf_dep = (1 - j[A4].mean(axis=1)).values

    out = {"experiment": "Mutual-information (BALD) disagreement vs plain confidence",
           "n_test": int(len(tes)), "n_errors": int(err.sum()),
           "members": list(MODELS),
           "why": "U_SAA shares 82.4% of its variance with confidence (r=0.9076); the "
                  "epistemic term removes confidence by construction.",
           "confounding_check": {
               "corr_deployed_U_SAA_with_confidence": round(float(np.corrcoef(u_saa, conf_dep)[0, 1]), 4),
               "corr_epistemic_with_aleatoric": round(float(np.corrcoef(epistemic, aleatoric)[0, 1]), 4),
               "corr_epistemic_with_total": round(float(np.corrcoef(epistemic, total)[0, 1]), 4)},
           "error_detection_auc": {
               "epistemic (disagreement, confidence removed)": round(auc(err, epistemic), 4),
               "aleatoric (per-member uncertainty)": round(auc(err, aleatoric), 4),
               "total predictive entropy": round(auc(err, total), 4),
               "1 - mean member probability [plain confidence]": round(auc(err, 1 - pbar), 4)},
           "epistemic_minus_confidence": paired(err, epistemic, 1 - pbar),
           "epistemic_minus_aleatoric": paired(err, epistemic, aleatoric),
           "why_three_signals_share_an_auc": {
               "pbar_range": [round(float(pbar.min()), 4), round(float(pbar.max()), 4)],
               "fraction_above_half": round(float((pbar > 0.5).mean()), 4),
               "spearman_total_vs_confidence": round(float(spearmanr(total, 1 - pbar).statistic), 6),
               "spearman_aleatoric_vs_confidence": round(float(spearmanr(aleatoric, 1 - pbar).statistic), 6),
               "note": "Every calibrated ensemble probability lies above 0.5, because the corpus is "
                       "84.5% scene-positive and no member ever predicts absence. Binary entropy is "
                       "monotone over that range, so total entropy ranks identically to 1 - pbar "
                       "(Spearman exactly 1.0) and aleatoric almost so. Their equal AUROC is a "
                       "consequence of the class balance, not a coding error. Epistemic is the only "
                       "one of the four that is genuinely a different ordering."}}

    e = out["error_detection_auc"]["epistemic (disagreement, confidence removed)"]
    c = out["error_detection_auc"]["1 - mean member probability [plain confidence]"]
    out["verdict"] = ("Epistemic beats confidence" if out["epistemic_minus_confidence"]["excludes_zero"]
                      and e > c else
                      "Epistemic does NOT beat confidence, even decoupled from it")

    if args.verify:
        prev = json.loads(OUT.read_text())
        same = json.dumps(prev, sort_keys=True) == json.dumps(out, sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
