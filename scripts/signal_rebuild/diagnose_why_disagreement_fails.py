#!/usr/bin/env python3
"""Why disagreement loses to confidence, measured rather than argued.

The redesigned signals (run_redesigned_signals.py) are rich -- 8,979 and 11,882 distinct
values -- and disagreement still loses to a plain confidence baseline, by more than it did
on the deployed signals. This asks why, with four measurements:

  1. Redundancy.     Correlation structure of the four signals, and the effective rank of
                     their covariance. A spread over k signals carries information only if
                     they vary along more than one direction.
  2. Error coupling. Whether the models are wrong about the SAME images. Disagreement can
                     only flag an error if at least one member gets it right.
  3. Level vs spread. Whether the spread carries anything once the mean is known --
                     AUROC of U_SAA computed WITHIN strata of SAA.
  4. Where errors sit. Whether errors concentrate at low confidence (which the mean sees)
                     or at high spread (which only the spread sees).

Outputs: results/signal_rebuild/why_disagreement_fails.json
"""
import argparse, glob, json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
OUT = SR / "why_disagreement_fails.json"
CHANNEL = "softmax"          # set from --channel; see main()
CATS = ["teaching", "family", "playing", "landscape", "drawing"]
TAX = ["person", "child", "horse", "building", "weapon", "vehicle", "tree",
       "clothing", "text", "animal"]


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


def build():
    y = rj([SR / "yolo_full_corpus.jsonl"])
    v = rj(sorted(glob.glob(str(SR / "llava_full/shard_*.jsonl"))))
    s = rj([SR / "siglip_scene_full.jsonl"])
    s["image_id"] = s.image_id.map(lambda x: x if "/" in str(x) else str(x).replace("_", "/", 1))
    P = pd.DataFrame(list(s.softmax)).fillna(0.0)
    s["A_scene"] = (P.max(axis=1).values if CHANNEL == "softmax"
                    else s.max_sigmoid.astype(float).values)
    txt = v.answers.apply(lambda a: " ".join(map(str, a)).lower() if isinstance(a, list) else "")
    v["A_vlm"] = txt.apply(lambda t: sum(bool(re.search(rf"\b{c}", t)) for c in TAX) / len(TAX))
    d = (y[["image_id", "conf_max"]].rename(columns={"conf_max": "A_object"})
         .merge(s[["image_id", "A_scene"]], on="image_id")
         .merge(v[["image_id", "A_vlm"]], on="image_id"))
    d["A_agree"] = 1 - (d.A_object - d.A_vlm).abs()
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--channel", choices=["softmax", "sigmoid"], default="softmax",
                        help="SigLIP scene channel. softmax reproduces the deployed record; "
                             "sigmoid is the corrected calibrated channel (Section 2.4).")
    args = ap.parse_args()
    global CHANNEL
    CHANNEL = args.channel
    globals()['OUT'] = (OUT if CHANNEL == 'softmax'
                        else OUT.with_name(OUT.stem + '_corrected.json'))
    d = build()
    S = ["A_object", "A_agree", "A_scene", "A_vlm"]
    out = {"experiment": "Why disagreement loses to confidence", "n_corpus": int(len(d))}

    # ---- 1. redundancy ----
    C = d[S].corr().round(3)
    cov = np.cov(d[S].values.T)
    ev = np.linalg.eigvalsh(cov)[::-1]
    p = ev / ev.sum()
    eff_rank = float(np.exp(-(p * np.log(np.clip(p, 1e-12, None))).sum()))
    out["redundancy"] = {
        "correlation_matrix": {a: {b: float(C.loc[a, b]) for b in S} for a in S},
        "eigenvalue_share_pct": [round(100 * float(x), 2) for x in p],
        "effective_rank": round(eff_rank, 3),
        "reading": ("Effective rank of 4 would mean four independent directions. Anything near 1 "
                    "means the four signals move together and their spread is close to a "
                    "deterministic function of their level.")}

    # ---- the labelled partition for everything error-related ----
    sp = pd.read_csv(SPLIT)
    j = sp.merge(d, left_on="ppn_id", right_on="image_id", how="inner")
    j["SAA"] = 0.6 * j.A_object + 0.2 * j.A_scene + 0.2 * j.A_vlm
    j["U"] = j[S].std(axis=1, ddof=1)
    cal = j[j.split == "calibration"]
    grid = np.arange(0.05, 0.95, 0.005)

    def bal(t, f):
        pr = (f.SAA >= t).astype(int); h = f.has_scene
        tp = ((pr == 1) & (h == 1)).sum(); fn = ((pr == 0) & (h == 1)).sum()
        tn = ((pr == 0) & (h == 0)).sum(); fp = ((pr == 1) & (h == 0)).sum()
        return ((tp / (tp + fn) if tp + fn else 0) + (tn / (tn + fp) if tn + fp else 0)) / 2

    thr = float(grid[int(np.argmax([bal(t, cal) for t in grid]))])
    j["err"] = ((j.SAA >= thr).astype(int) != j.has_scene).astype(int)
    te = j[j.split == "test"]

    # ---- 2. do the members err together? ----
    memb = {}
    for c in ["A_object", "A_scene", "A_vlm"]:
        t2 = float(grid[int(np.argmax([bal(t, cal) for t in grid]))])
        pred = (j[c] >= j[c].median()).astype(int)
        memb[c] = (pred != j.has_scene).astype(int)
    M = pd.DataFrame(memb)
    out["error_coupling"] = {
        "pairwise_error_correlation": {a: {b: round(float(M[a].corr(M[b])), 3) for b in M} for a in M},
        "all_three_wrong_together_pct": round(100 * float((M.sum(axis=1) == 3).mean()), 2),
        "at_least_one_right_pct": round(100 * float((M.sum(axis=1) < 3).mean()), 2),
        "reading": ("Disagreement can only flag an error when at least one member is right. "
                    "Where all members fail together, the spread is low and the ensemble is "
                    "confidently wrong.")}

    # ---- 3. does the spread add anything once the level is known? ----
    strata, qs = {}, np.quantile(te.SAA, [0, .25, .5, .75, 1.0])
    for i in range(4):
        m = (te.SAA >= qs[i]) & (te.SAA <= qs[i + 1])
        sub = te[m]
        if sub.err.nunique() > 1 and len(sub) > 20:
            strata[f"SAA quartile {i+1}"] = {
                "n": int(len(sub)), "errors": int(sub.err.sum()),
                "auroc_U_within_stratum": round(float(roc_auc_score(sub.err, sub.U)), 4)}
        else:
            strata[f"SAA quartile {i+1}"] = {"n": int(len(sub)), "errors": int(sub.err.sum()),
                                             "auroc_U_within_stratum": None}
    out["spread_given_level"] = {
        "overall_auroc_U": round(float(roc_auc_score(te.err, te.U)), 4),
        "overall_auroc_confidence": round(float(roc_auc_score(te.err, 1 - te.SAA)), 4),
        "within_strata": strata,
        "corr_U_with_SAA": round(float(j.U.corr(j.SAA)), 3),
        "reading": ("If the spread carried information the mean does not, it would still rank "
                    "errors inside a stratum where the mean is held roughly constant.")}

    # ---- 4. where do the errors actually sit? ----
    lo, hi = te.SAA < thr, te.SAA >= thr
    out["where_errors_sit"] = {
        "decision_threshold": round(thr, 3),
        "errors_below_threshold": int(te.loc[lo, "err"].sum()),
        "errors_above_threshold": int(te.loc[hi, "err"].sum()),
        "mean_U_on_errors": round(float(te.loc[te.err == 1, "U"].mean()), 4),
        "mean_U_on_correct": round(float(te.loc[te.err == 0, "U"].mean()), 4),
        "mean_SAA_on_errors": round(float(te.loc[te.err == 1, "SAA"].mean()), 4),
        "mean_SAA_on_correct": round(float(te.loc[te.err == 0, "SAA"].mean()), 4)}

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
