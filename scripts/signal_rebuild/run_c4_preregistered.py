#!/usr/bin/env python3
"""C4, run exactly as pre-registered (roadmap/preregistration_c4.md, 19 September 2026).

Question: does uncertainty built at the image-and-class level detect the pipeline's errors better
than plain confidence?

Error. The pipeline's scene decision is the thesis's Level-2 fusion,
0.6 object + 0.2 scene + 0.2 VLM >= 0.5 (as in the held-out evaluation); an error is a picture
where that decision disagrees with the human label.
  Set A: the 300 independently annotated pictures (authoritative); the 164 the detector never
         trained on are reported separately.
  Set B: the held-out test partition of the corrected scene pool (249 pictures), supporting only.

Signals (higher = more likely an error):
  1 per-class disagreement: mean over the 8 classes of the spread (population std) of the three
    systems' calibrated per-class probabilities            <- the signal tested
  2 per-class entropy: mean over classes of the binary entropy of the mean calibrated probability
  3 global disagreement: spread of each system's maximum calibrated probability over classes
  4 the deployed scalar signal, unchanged (std of the four stored agent scores)
  5 plain confidence: 1 - |fused scene probability - 0.5|  (Hendrycks and Gimpel's baseline)

Calibration (pre-registration amendment): Platt scaling -- a logistic fit on the logit of each raw
score, per system and class. The pre-registration fixed where calibration is fitted, not the method;
Platt scaling is the thesis's own calibration method. Set A is cross-fitted (5 folds, fitted on four,
applied to the fifth); set B uses calibrators fitted on the 300 minus the 37 that sit in its test
partition.

Decision rule: if on set A the DeLong 95% interval for AUROC(1) - AUROC(5) lies entirely above zero,
RQ1 becomes a qualified yes; otherwise it stays no. Set B cannot overturn set A.

    python run_c4_preregistered.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

BASE = Path("/data/brhanu/thesis_project")
CS = BASE / "results/signal_rebuild/class_scores"
OUT = BASE / "results/signal_rebuild/c4/c4_preregistered.json"
CLASSES = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]
SYSTEMS = ["detector", "siglip", "llava"]
AGENTS4 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
SEED = 20260919
NAMES = {"s1": "per-class disagreement", "s2": "per-class entropy", "s3": "global disagreement",
         "s4": "deployed scalar signal", "s5": "plain confidence"}


def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6); return np.log(x / (1 - x))


def fit_calibrators(Rsub, tsub):
    """Platt scaling per system and class, on the logit of the raw score."""
    cal = {}
    for si in range(len(SYSTEMS)):
        for ci, c in enumerate(CLASSES):
            y = tsub[c].values; x = logit(Rsub[:, si, ci]).reshape(-1, 1)
            cal[(si, ci)] = LogisticRegression(C=100.0).fit(x, y) if 0 < y.sum() < len(y) else float(y.mean())
    return cal


def apply_calibrators(cal, R):
    P = np.zeros_like(R)
    for (si, ci), m in cal.items():
        P[:, si, ci] = m if isinstance(m, float) else m.predict_proba(logit(R[:, si, ci]).reshape(-1, 1))[:, 1]
    return P


def signals(P, fused, u_saa):
    mean_p = P.mean(axis=1).clip(1e-9, 1 - 1e-9)
    ent = -(mean_p * np.log2(mean_p) + (1 - mean_p) * np.log2(1 - mean_p))
    return {"s1": P.std(axis=1).mean(axis=1), "s2": ent.mean(axis=1), "s3": P.max(axis=2).std(axis=1),
            "s4": u_saa, "s5": 1 - np.abs(fused - 0.5)}


# DeLong's test for two correlated ROC curves (fast algorithm of Sun and Xu, 2014)
def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]: k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1; i = k
    out = np.empty(n); out[j] = t; return out


def delong(y, a, b):
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum(); preds = np.vstack([a, b])
    tx = np.array([_midrank(p[pos]) for p in preds]); ty = np.array([_midrank(p[neg]) for p in preds])
    tz = np.array([_midrank(np.concatenate([p[pos], p[neg]])) for p in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    d = aucs[0] - aucs[1]; se = np.sqrt(max(S[0, 0] + S[1, 1] - 2 * S[0, 1], 1e-12))
    return float(d), [float(d - 1.96 * se), float(d + 1.96 * se)], float(2 * norm.sf(abs(d) / se))


def evaluate(sig, err, label):
    res = {"pictures": int(len(err)), "errors": int(err.sum()),
           "auroc": {NAMES[k]: round(float(roc_auc_score(err, v)), 4) for k, v in sig.items()}}
    d, ci, p = delong(err, sig["s1"], sig["s5"])
    res["tested_difference"] = {"per-class disagreement minus plain confidence": round(d, 4),
                                "delong_ci95": [round(ci[0], 4), round(ci[1], 4)], "p_value": round(p, 4),
                                "interval_above_zero": bool(ci[0] > 0)}
    d4, ci4, p4 = delong(err, sig["s4"], sig["s5"])
    res["reported_difference_deployed_minus_confidence"] = {"difference": round(d4, 4), "delong_ci95": [round(ci4[0], 4), round(ci4[1], 4)]}
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    m = np.load(CS / "class_score_matrix.npz", allow_pickle=True)
    ids = [str(x) for x in m["ids"]]; pos = {i: k for k, i in enumerate(ids)}
    mc, ms = [str(x) for x in m["classes"]], [str(x) for x in m["systems"]]
    R = m["raw"][:, [ms.index(s) for s in SYSTEMS]][:, :, [mc.index(c) for c in CLASSES]].astype(np.float64)
    truth = pd.read_csv(CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    ag = pd.read_csv(BASE / "results/multi_agent/agent_comparison_scores.csv").drop_duplicates("image_id").set_index("image_id")
    fused_all = (0.6 * ag["existing_pipeline_agent"] + 0.2 * ag["scene_agent"] + 0.2 * ag["vlm_agent"])
    u_all = ag[AGENTS4].std(axis=1)

    # set A: the 300, cross-fitted calibration
    rowsA = np.array([pos[i] for i in truth.index]); nA = len(rowsA)
    PA = np.zeros((nA, len(SYSTEMS), len(CLASSES)))
    tt = truth.reset_index(drop=True)
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(np.arange(nA)):
        PA[te] = apply_calibrators(fit_calibrators(R[rowsA[tr]], tt.iloc[tr]), R[rowsA[te]])
    fA = fused_all.loc[truth.index].values; uA = u_all.loc[truth.index].values
    errA = ((fA >= 0.5).astype(int) != truth["has_scene"].values).astype(int)
    sigA = signals(PA, fA, uA)
    unseen = truth["in_detector_train"].values == 0

    # set B: the test partition, calibrators fitted on the 300 minus those in it
    split = pd.read_csv(BASE / "results/multi_agent/corrected_clean_holdout_split.csv")
    test = split[split.split == "test"]
    in_test = truth.index.isin(test.ppn_id)
    keep = np.where(~in_test)[0]
    calB = fit_calibrators(R[rowsA[keep]], tt.iloc[keep])
    rowsB = np.array([pos[i] for i in test.ppn_id])
    PB = apply_calibrators(calB, R[rowsB])
    fB = fused_all.loc[test.ppn_id].values; uB = u_all.loc[test.ppn_id].values
    errB = ((fB >= 0.5).astype(int) != test["has_scene"].values).astype(int)
    sigB = signals(PB, fB, uB)

    A = evaluate(sigA, errA, "A")
    A_unseen = evaluate({k: v[unseen] for k, v in sigA.items()}, errA[unseen], "A unseen")
    B = evaluate(sigB, errB, "B")
    verdict = ("RQ1 becomes a qualified yes" if A["tested_difference"]["interval_above_zero"]
               else "RQ1 stays no, now at two levels of granularity")
    out = {"experiment": "C4 as pre-registered on 19 September 2026", "calibration": "Platt scaling on the logit of each raw score, per system and class",
           "set_A_300": A, "set_A_164_not_in_detector_training": A_unseen, "set_B_test_partition": B,
           "calibration_pictures_for_set_B": int(len(keep)), "verdict_by_the_registered_rule": verdict}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out, indent=2))
    for name, r in [("Set A, 300", A), ("Set A, 164 unseen", A_unseen), ("Set B, test partition", B)]:
        print(f"\n{name}: {r['pictures']} pictures, {r['errors']} errors")
        for k, v in r["auroc"].items(): print(f"  {k:26s} {v:.4f}")
        t = r["tested_difference"]
        print(f"  tested: disagreement - confidence {t['per-class disagreement minus plain confidence']:+.4f} "
              f"CI {t['delong_ci95']}  p={t['p_value']}")
    print("\nVERDICT:", verdict)


if __name__ == "__main__":
    raise SystemExit(main())
