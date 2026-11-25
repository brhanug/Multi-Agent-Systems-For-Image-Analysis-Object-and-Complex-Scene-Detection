#!/usr/bin/env python3
"""The scene agent was read from the wrong channel. Fix it, and retest everything.

SigLIP is trained with a *sigmoid* objective: every image-prompt pair receives an
independent calibrated probability, and an image matching none of the prompts scores near
zero on all of them. The deployed pipeline discarded that channel and applied a softmax
across the five scene prompts instead.

The five prompts are "a historical illustration of a {teaching, family, playing, landscape,
drawing} scene" -- all five are scene categories. A softmax over them is a forced choice
among scene types, so it cannot express "no scene": its minimum over the corpus is 0.2956.
The sigmoid channel reaches 0.0 and sits below 0.01 on 48.6% of images.

Both channels were computed and stored for all 12,110 images. Only the softmax was used.

This script asks three questions, in order:

  1. Which channel measures scene presence?           (against human labels, not a proxy)
  2. What does using the right one do to the pipeline? (balanced accuracy, error rate, CRC)
  3. Does it rescue disagreement?                      (unanimity, then BALD epistemic)

Question 3 is the one that matters for the thesis's central claim. Disagreement can only
rank errors if the members sometimes reach different *conclusions*, not merely different
degrees of the same conclusion. The BALD run found every calibrated member probability
above 0.5 on every image -- the ensemble is unanimous corpus-wide. This tests whether the
scene fix, alone or together with class-balanced calibration, breaks that unanimity.

Outputs: results/signal_rebuild/scene_channel_fix.json
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
OUT = SR / "scene_channel_fix.json"
TAX = ["person", "child", "horse", "building", "weapon", "vehicle", "tree",
       "clothing", "text", "animal"]
GRID = np.arange(0.05, 0.95, 0.005)
CRC_GRID = np.linspace(0.001, 1.0, 1000)
ALPHAS = [0.01, 0.02, 0.05, 0.10, 0.15]
RNG = np.random.default_rng(20260912)
B = 2000
EPS = 1e-12


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
    """Paired bootstrap on AUROC(a) - AUROC(b)."""
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


def degenerate(pred, h):
    rate = float(pred.mean()); ba = bal_acc(pred, h)
    return bool(rate > 0.95 or rate < 0.05 or ba <= 0.55), round(rate, 4), round(ba, 4)


def Hb(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def crc(U_cal, E_cal, U_te, E_te, alpha):
    n = len(U_cal)
    risks = np.array([np.mean(E_cal * (U_cal < g)) for g in CRC_GRID])
    ok = (n / (n + 1.0)) * risks + (1.0 / (n + 1.0)) <= alpha
    lam = float(np.max(CRC_GRID[ok])) if ok.any() else 0.0  # no cut-off keeps the promise: approve nothing
    auto = U_te < lam
    caught = int(E_te[~auto].sum()); tot = int(E_te.sum())
    return {"alpha": alpha, "lambda": round(lam, 4),
            "test_empirical_risk": round(float(np.mean(E_te * auto)), 4),
            "bound_satisfied": bool(float(np.mean(E_te * auto)) <= alpha),
            "auto_approval_pct": round(100 * float(auto.mean()), 2),
            "error_recall_pct": round(100 * caught / tot, 2) if tot else None}


def build():
    y = rj([SR / "yolo_full_corpus.jsonl"])
    v = rj(sorted(glob.glob(str(SR / "llava_full/shard_*.jsonl"))))
    k = rj(sorted(glob.glob(str(SR / "kosmos_full/shard_*.jsonl"))))
    s = rj([SR / "siglip_scene_full.jsonl"])
    s["image_id"] = s.image_id.map(lambda x: x if "/" in str(x) else str(x).replace("_", "/", 1))

    # the two channels, both already computed for every image
    s["A_scene_softmax"] = pd.DataFrame(list(s.softmax)).fillna(0.0).max(axis=1).values
    s["A_scene_sigmoid"] = s.max_sigmoid.astype(float).values

    y["A_object"] = y.conf_max.clip(0, 1)
    txt = v.answers.apply(lambda a: " ".join(map(str, a)).lower() if isinstance(a, list) else "")
    v["A_vlm"] = txt.apply(lambda t: sum(bool(re.search(rf"\b{c}", t)) for c in TAX) / len(TAX))
    k["A_ocr"] = np.minimum(k.kosmos_output.fillna("").str.len() / 80.0, 1.0)

    d = (y[["image_id", "A_object"]]
         .merge(s[["image_id", "A_scene_softmax", "A_scene_sigmoid"]], on="image_id")
         .merge(v[["image_id", "A_vlm"]], on="image_id")
         .merge(k[["image_id", "A_ocr"]], on="image_id"))
    return d


def evaluate_rule(j, scene_col, tag):
    """Refit the decision threshold on calibration, score on test, with the degeneracy guard."""
    jj = j.copy()
    jj["SAA"] = 0.6 * jj.A_object + 0.2 * jj[scene_col] + 0.2 * jj.A_vlm
    cal, te = jj[jj.split == "calibration"], jj[jj.split == "test"]
    thr = float(GRID[int(np.argmax([bal_acc((cal.SAA >= t).astype(int).values,
                                            cal.has_scene.values) for t in GRID]))])
    pred = (te.SAA >= thr).astype(int).values
    deg, rate, ba = degenerate(pred, te.has_scene.values)
    err = (pred != te.has_scene.values).astype(int)
    res = {"variant": tag, "scene_column": scene_col, "threshold": round(thr, 3),
           "test_n": int(len(te)), "test_errors": int(err.sum()),
           "test_error_rate": round(float(err.mean()), 4),
           "balanced_accuracy": ba, "predicted_positive_rate": rate, "degenerate": deg,
           "auroc_SAA_vs_scene_presence": round(auc(te.has_scene.values, te.SAA.values), 4)}
    if not deg and 0 < err.sum() < len(err):
        res["auroc_confidence_vs_error"] = round(auc(err, (1 - te.SAA).values), 4)
    return res, jj, err, thr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    d = build()
    sp = pd.read_csv(SPLIT)
    j = sp.merge(d, left_on="ppn_id", right_on="image_id", how="inner")

    out = {"experiment": "The scene agent was read from the wrong SigLIP channel",
           "n_corpus": int(len(d)), "labelled_overlap_n": int(len(j))}

    # ---- 1. the defect ------------------------------------------------------------
    sm, sg = d.A_scene_softmax, d.A_scene_sigmoid
    no_match = sg < 0.01
    out["the_defect"] = {
        "softmax_min": round(float(sm.min()), 4),
        "softmax_median": round(float(sm.median()), 4),
        "sigmoid_min": round(float(sg.min()), 6),
        "sigmoid_median": round(float(sg.median()), 6),
        "pct_images_sigmoid_below_0.01": round(100 * float(no_match.mean()), 2),
        "corr_between_channels": round(float(sm.corr(sg)), 4),
        "on_images_matching_no_prompt": {
            "n": int(no_match.sum()),
            "mean_softmax_reported": round(float(sm[no_match].mean()), 4),
            "pct_of_them_softmax_above_0.5": round(100 * float((sm[no_match] > 0.5).mean()), 2)},
        "reading": ("All five prompts name a scene type, so a softmax over them is a forced "
                    "choice among scenes and cannot express absence -- its corpus minimum is "
                    "0.2956. The sigmoid channel is SigLIP's native calibrated output and "
                    "reaches 0. On the images SigLIP says match no scene prompt, the deployed "
                    "softmax signal reports high scene confidence.")}

    # ---- 2. which channel measures scene presence? --------------------------------
    yl = j.has_scene.values
    out["which_channel_measures_scene_presence"] = {
        "auroc_softmax": round(auc(yl, j.A_scene_softmax.values), 4),
        "auroc_sigmoid": round(auc(yl, j.A_scene_sigmoid.values), 4),
        "paired_delta_sigmoid_minus_softmax": boot(yl, j.A_scene_sigmoid.values,
                                                   j.A_scene_softmax.values),
        "by_split": {s: {"n": int(len(g)),
                         "softmax": round(auc(g.has_scene.values, g.A_scene_softmax.values), 4),
                         "sigmoid": round(auc(g.has_scene.values, g.A_scene_sigmoid.values), 4)}
                     for s, g in j.groupby("split")},
        "reading": ("The deployed channel is below chance against the quantity it names. "
                    "The unused channel, computed in the same pass, is a usable detector.")}

    # ---- 3. what it does to the pipeline ------------------------------------------
    r_sm, j_sm, err_sm, thr_sm = evaluate_rule(j, "A_scene_softmax", "deployed channel (softmax)")
    r_sg, j_sg, err_sg, thr_sg = evaluate_rule(j, "A_scene_sigmoid", "fixed channel (sigmoid)")
    out["what_it_does_to_the_pipeline"] = [r_sm, r_sg]
    te_sm, te_sg = j_sm[j_sm.split == "test"], j_sg[j_sg.split == "test"]
    out["scene_presence_ranking_delta"] = boot(te_sg.has_scene.values, te_sg.SAA.values,
                                               te_sm.SAA.values)

    # conformal triage under each channel, each calibrated on its own signal
    crc_out = {}
    for tag, jv, errv in [("deployed channel", j_sm, err_sm), ("fixed channel", j_sg, err_sg)]:
        cal, te = jv[jv.split == "calibration"], jv[jv.split == "test"]
        thr = float(GRID[int(np.argmax([bal_acc((cal.SAA >= t).astype(int).values,
                                                cal.has_scene.values) for t in GRID]))])
        E_cal = ((cal.SAA >= thr).astype(int).values != cal.has_scene.values).astype(int)
        U_cal, U_te = (1 - cal.SAA).values, (1 - te.SAA).values
        crc_out[tag] = [crc(U_cal, E_cal, U_te, errv, a) for a in ALPHAS]
    out["conformal_triage"] = crc_out

    # ---- 4. does it rescue disagreement? ------------------------------------------
    # Members are calibrated to a probability of scene presence on the training partition,
    # so their spread is a spread over one estimand. Both an unweighted and a class-balanced
    # fit are reported: the corpus is 84.5% positive, and an unweighted fit on that base rate
    # can leave every member above 0.5 regardless of what the image contains.
    MEM = {"detector": "A_object", "scene": None, "vlm": "A_vlm", "ocr": "A_ocr"}
    resc = {}
    for scene_col, tag in [("A_scene_softmax", "deployed channel"),
                           ("A_scene_sigmoid", "fixed channel")]:
        MEM["scene"] = scene_col
        tr = j[j.split == "train"]; te = j[j.split == "test"]
        for cw, cwtag in [(None, "unweighted calibration"), ("balanced", "class-balanced calibration")]:
            P = {}
            for m, col in MEM.items():
                lr = LogisticRegression(max_iter=2000, class_weight=cw).fit(
                    tr[[col]].values, tr.has_scene.values)
                P[m] = lr.predict_proba(te[[col]].values)[:, 1]
            M = np.column_stack([P[m] for m in MEM])
            pbar = M.mean(axis=1)
            total, aleat = Hb(pbar), Hb(M).mean(axis=1)
            epist = total - aleat
            errv = err_sm if scene_col == "A_scene_softmax" else err_sg
            unanimous = ((M > 0.5).all(axis=1) | (M <= 0.5).all(axis=1))
            key = f"{tag}, {cwtag}"
            resc[key] = {
                "member_prob_range": [round(float(M.min()), 4), round(float(M.max()), 4)],
                "ensemble_mean_range": [round(float(pbar.min()), 4), round(float(pbar.max()), 4)],
                "pct_images_all_members_agree_on_the_call": round(100 * float(unanimous.mean()), 2),
                "pct_images_some_member_says_absent": round(
                    100 * float((M <= 0.5).any(axis=1).mean()), 2),
                "auroc_epistemic_vs_error": round(auc(errv, epist), 4),
                "auroc_confidence_vs_error": round(auc(errv, 1 - pbar), 4),
                "epistemic_minus_confidence": boot(errv, epist, 1 - pbar)}
    out["does_it_rescue_disagreement"] = resc
    out["reading"] = (
        "Disagreement can rank errors only where members reach different conclusions. The "
        "unanimity row is the one to read: where it stays at 100%, the spread measures how "
        "emphatically the ensemble agrees, not whether it is right.")

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
