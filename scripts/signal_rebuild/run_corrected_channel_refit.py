#!/usr/bin/env python3
"""Apply the corrected scene channel end to end: refit the fusion, refit the triage.

Section 2.4 established that the deployed scene signal was read from SigLIP's softmax
across five scene prompts rather than from its native sigmoid, and therefore scored 0.4172
against scene presence -- below chance. This script applies the correction properly rather
than only substituting the column:

  1. The fusion weights were hand-set to 0.6/0.2/0.2 around a signal that was
     anti-predictive. They are refitted here, on the training partition only, and the
     hand-set and fitted weights are compared on test.
  2. The decision threshold is refitted for every weighting, since changing the weights
     changes the scale of SAA.
  3. The conformal triage is recalibrated on the resulting score, so the system is
     evaluated as a system rather than as a column swapped into someone else's settings --
     the error that made the rebuilt pipeline look broken in Section 2.4.

Every configuration is reported under both channels so the correction's effect is visible,
and each carries the degeneracy guard: a rule predicting one class almost everywhere, or
scoring balanced accuracy at or below 0.55, is reported as degenerate and its downstream
figures are withheld rather than printed.

Outputs: results/signal_rebuild/corrected_channel_refit.json
"""
import argparse, glob, json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
OUT = SR / "corrected_channel_refit.json"
TAX = ["person", "child", "horse", "building", "weapon", "vehicle", "tree",
       "clothing", "text", "animal"]
GRID = np.arange(0.02, 0.98, 0.002)
CRC_GRID = np.linspace(0.001, 1.0, 1000)
ALPHAS = [0.01, 0.02, 0.05, 0.10, 0.15]
RNG = np.random.default_rng(20260912)
B = 2000


def rj(paths):
    r = []
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    r.append(json.loads(line))
                except Exception:
                    pass
    return pd.DataFrame(r).drop_duplicates("image_id")


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def boot(y, a, b):
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    n, o = len(y), []
    for _ in range(B):
        i = RNG.integers(0, n, n)
        if 0 < y[i].sum() < n:
            o.append(auc(y[i], a[i]) - auc(y[i], b[i]))
    lo, hi = np.percentile(o, [2.5, 97.5])
    return {"delta": round(float(np.mean(o)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def bal_acc(pred, h):
    tp = int(((pred == 1) & (h == 1)).sum()); fn = int(((pred == 0) & (h == 1)).sum())
    tn = int(((pred == 0) & (h == 0)).sum()); fp = int(((pred == 1) & (h == 0)).sum())
    return ((tp / (tp + fn) if tp + fn else 0.0) + (tn / (tn + fp) if tn + fp else 0.0)) / 2


def crc(U_cal, E_cal, U_te, E_te, alpha):
    n = len(U_cal)
    risks = np.array([np.mean(E_cal * (U_cal < g)) for g in CRC_GRID])
    ok = (n / (n + 1.0)) * risks + (1.0 / (n + 1.0)) <= alpha
    lam = float(np.max(CRC_GRID[ok])) if ok.any() else 0.0  # no cut-off keeps the promise: approve nothing
    auto = U_te < lam
    caught = int(E_te[~auto].sum()); tot = int(E_te.sum())
    risk = float(np.mean(E_te * auto))
    return {"alpha": alpha, "lambda": round(lam, 4), "test_empirical_risk": round(risk, 4),
            "bound_satisfied": bool(risk <= alpha),
            "auto_approval_pct": round(100 * float(auto.mean()), 2),
            "error_recall_pct": round(100 * caught / tot, 2) if tot else None}


def build():
    y = rj([SR / "yolo_full_corpus.jsonl"])
    v = rj(sorted(glob.glob(str(SR / "llava_full/shard_*.jsonl"))))
    s = rj([SR / "siglip_scene_full.jsonl"])
    s["image_id"] = s.image_id.map(lambda x: x if "/" in str(x) else str(x).replace("_", "/", 1))
    s["softmax_ch"] = pd.DataFrame(list(s.softmax)).fillna(0.0).max(axis=1).values
    s["sigmoid_ch"] = s.max_sigmoid.astype(float).values
    y["A_object"] = y.conf_max.clip(0, 1)
    txt = v.answers.apply(lambda a: " ".join(map(str, a)).lower() if isinstance(a, list) else "")
    v["A_vlm"] = txt.apply(lambda t: sum(bool(re.search(rf"\b{c}", t)) for c in TAX) / len(TAX))
    return (y[["image_id", "A_object"]]
            .merge(s[["image_id", "softmax_ch", "sigmoid_ch"]], on="image_id")
            .merge(v[["image_id", "A_vlm"]], on="image_id"))


def configure(j, chan, weights, tag):
    """Score one weighting, with the decision threshold refitted on calibration."""
    w = np.asarray(weights, float)
    X = j[["A_object", chan, "A_vlm"]].values
    j = j.copy(); j["SAA"] = X @ w
    cal, te = j[j.split == "calibration"], j[j.split == "test"]
    thr = float(GRID[int(np.argmax([bal_acc((cal.SAA >= t).astype(int).values,
                                            cal.has_scene.values) for t in GRID]))])
    pred = (te.SAA >= thr).astype(int).values
    ba = bal_acc(pred, te.has_scene.values); rate = float(pred.mean())
    deg = bool(rate > 0.95 or rate < 0.05 or ba <= 0.55)
    err = (pred != te.has_scene.values).astype(int)
    res = {"configuration": tag, "channel": chan,
           "weights": {"object": round(float(w[0]), 4), "scene": round(float(w[1]), 4),
                       "vlm": round(float(w[2]), 4)},
           "threshold": round(thr, 4), "balanced_accuracy": round(ba, 4),
           "predicted_positive_rate": round(rate, 4), "degenerate": deg,
           "test_error_rate": round(float(err.mean()), 4), "test_errors": int(err.sum()),
           "auroc_SAA_vs_scene_presence": round(auc(te.has_scene.values, te.SAA.values), 4)}
    if deg:
        res["note"] = "degenerate under the guard; downstream figures withheld"
        return res, j, err, thr
    res["auroc_confidence_vs_error"] = round(auc(err, (1 - te.SAA).values), 4)
    E_cal = ((cal.SAA >= thr).astype(int).values != cal.has_scene.values).astype(int)
    res["conformal_triage"] = [crc((1 - cal.SAA).values, E_cal, (1 - te.SAA).values, err, a)
                               for a in ALPHAS]
    return res, j, err, thr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    d = build()
    sp = pd.read_csv(SPLIT)
    j = sp.merge(d, left_on="ppn_id", right_on="image_id", how="inner")
    tr = j[j.split == "train"]
    out = {"experiment": "The corrected scene channel applied end to end",
           "n_corpus": int(len(d)), "labelled_overlap_n": int(len(j))}

    # ---- fit the weights on train, per channel -----------------------------------
    fitted = {}
    for chan in ["softmax_ch", "sigmoid_ch"]:
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            tr[["A_object", chan, "A_vlm"]].values, tr.has_scene.values)
        w = lr.coef_[0]
        fitted[chan] = {"raw": [round(float(x), 4) for x in w],
                        "normalised_to_unit_L1": [round(float(x), 4) for x in w / np.abs(w).sum()]}
    out["fitted_fusion_weights"] = fitted
    out["fitted_weights_reading"] = (
        "Given the deployed channel the fit assigns the scene agent a NEGATIVE coefficient -- "
        "it prefers to subtract it -- while the deployed formula adds it at +0.2. Given the "
        "corrected channel the scene agent receives the largest weight of the three. The "
        "hand-set 0.6/0.2/0.2 was chosen around a signal that was running backwards.")

    # ---- compare weightings under both channels ----------------------------------
    configs = []
    for chan, cname in [("softmax_ch", "deployed channel"), ("sigmoid_ch", "corrected channel")]:
        wn = np.abs(np.asarray(fitted[chan]["normalised_to_unit_L1"], float))
        wn = wn / wn.sum()
        for weights, wtag in [((0.6, 0.2, 0.2), "hand-set 0.6/0.2/0.2"),
                              (tuple(wn), "refitted on train")]:
            r, jj, err, thr = configure(j, chan, weights, f"{cname}, {wtag}")
            configs.append(r)
            if cname == "corrected channel" and wtag == "refitted on train":
                best = (jj, err)
            if cname == "deployed channel" and wtag == "hand-set 0.6/0.2/0.2":
                base = (jj, err)
    out["configurations"] = configs

    # ---- is the fully corrected system better than the deployed one? --------------
    jb, eb = base; jx, ex = best
    teb, tex = jb[jb.split == "test"], jx[jx.split == "test"]
    out["fully_corrected_vs_deployed"] = {
        "scene_presence_ranking": boot(tex.has_scene.values, tex.SAA.values, teb.SAA.values),
        "deployed_error_rate": round(float(eb.mean()), 4),
        "corrected_error_rate": round(float(ex.mean()), 4),
        "reading": ("The comparison that matters is ranking scene presence, since the error "
                    "labels differ between the two rules. A positive delta whose interval "
                    "excludes zero means the corrected system orders images better.")}

    if args.verify:
        prev = json.loads(OUT.read_text())
        same = json.dumps(prev, sort_keys=True) == json.dumps(out, sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH")
        return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
