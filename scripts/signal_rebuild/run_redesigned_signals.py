#!/usr/bin/env python3
"""Redesign the Tier 1 signals to use what the models actually emit, and retest.

Section 2.4 shows the deployed signals discard most of what the models produce:
  A_object  min(k/3, 1) on an integer count -- 4 values, 55.6% of images pinned at 1.0,
            43% of the count's entropy discarded, and the detector's confidences,
            locations and classes never used at all.
  A_scene   read from a binary flag rather than the classifier -- 2 values, where the
            real SigLIP distribution over the five categories has 11,882.
  A_vlm     a paragraph reduced to one bit, from the same answer as A_object.
  A_ocr     character count: volume, not validity.

This asks whether that reduction, rather than disagreement itself, is what defeats the
central claim. Each signal is rebuilt from the model's own output:

  A_object  = detector max box confidence          (continuous, no saturation)
  A_scene   = max softmax over the 5 categories, or classifier peakedness 1 - H/log 5
  A_vlm     = fraction of the 10 taxonomy classes the VLM names (graded, not binary)
  A_agree   = 1 - |A_object - A_vlm|, now a comparison of two independent measurements

Every variant gets its own decision threshold, refitted on the calibration partition,
because the 0.5 cut belongs to the deployed scales. Comparing a redesign against an
inherited hyperparameter is how the first end-to-end attempt went wrong.

Outputs: results/signal_rebuild/redesigned_signals.json
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
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT = SR / "redesigned_signals.json"
CHANNEL = "softmax"          # set from --channel; see main()
CATS = ["teaching", "family", "playing", "landscape", "drawing"]
TAXONOMY = ["person", "child", "horse", "building", "weapon", "vehicle", "tree",
            "clothing", "text", "animal"]
SEED = 20260912


def read_jsonl(paths):
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
    y = read_jsonl([SR / "yolo_full_corpus.jsonl"])
    v = read_jsonl(sorted(glob.glob(str(SR / "llava_full/shard_*.jsonl"))))
    s = read_jsonl([SR / "siglip_scene_full.jsonl"])
    s["image_id"] = s.image_id.map(lambda x: x if "/" in str(x) else str(x).replace("_", "/", 1))

    d = y[["image_id", "n_box", "conf_max", "conf_mean"]].copy()
    d["A_object"] = d.conf_max.clip(0, 1)                       # detector's own confidence

    P = pd.DataFrame(list(s.softmax)).fillna(0.0)
    # softmax over five scene prompts is a forced choice and cannot express absence;
    # max_sigmoid is SigLIP's native calibrated output. See Section 2.4.
    s["max_softmax"] = (P.max(axis=1).values if CHANNEL == "softmax"
                        else s.max_sigmoid.astype(float).values)
    ent = -(P * np.log(P.clip(1e-12))).sum(axis=1) / np.log(len(CATS))
    s["peakedness"] = (1 - ent).values
    d = d.merge(s[["image_id", "max_softmax", "peakedness"]], on="image_id", how="inner")

    txt = v.answers.apply(lambda a: " ".join(map(str, a)).lower() if isinstance(a, list) else "")
    v["A_vlm_graded"] = txt.apply(
        lambda t: sum(bool(re.search(rf"\b{c}", t)) for c in TAXONOMY) / len(TAXONOMY))
    d = d.merge(v[["image_id", "A_vlm_graded"]], on="image_id", how="inner")
    return d


def evaluate(j, obj, scn, vlm, name, out):
    a = j[obj].values
    sc = j[scn].values
    vl = j[vlm].values
    ag = 1 - np.abs(a - vl)
    saa = 0.6 * a + 0.2 * sc + 0.2 * vl
    U = np.std(np.vstack([a, ag, sc, vl]), axis=0, ddof=1)

    cal = j.split.values == "calibration"
    te = j.split.values == "test"
    hs = j.has_scene.values
    # Choosing the threshold by raw error rate collapses to the all-positive rule in an
    # 84%-positive pool -- it scores the majority-class baseline exactly. Balanced
    # accuracy is the objective that cannot be won that way.
    grid = np.arange(0.05, 0.95, 0.005)

    def bal(t, m):
        pr = (saa[m] >= t).astype(int); h = hs[m]
        tp = int(((pr == 1) & (h == 1)).sum()); fn = int(((pr == 0) & (h == 1)).sum())
        tn = int(((pr == 0) & (h == 0)).sum()); fp = int(((pr == 1) & (h == 0)).sum())
        rec = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        return (rec + spec) / 2

    thr = float(grid[int(np.argmax([bal(t, cal) for t in grid]))])
    err = ((saa >= thr).astype(int) != hs).astype(int)

    y, u, c = err[te], U[te], (1 - saa[te])
    rate = float(y.mean())
    pos_rate = float((saa[te] >= thr).mean())
    majority_err = 1 - float(hs[te].mean())
    # degenerate if it predicts one class almost everywhere, or does no better than
    # always predicting the majority class
    # Degenerate means "predicts one class almost everywhere" or "no better than chance at
    # separating the two classes". Raw error rate is NOT the test: a balanced objective
    # deliberately trades raw accuracy for balance, and the majority-class rule scores a
    # balanced accuracy of exactly 0.5.
    bal_te = bal(thr, te)
    degenerate = bool(pos_rate > 0.97 or pos_rate < 0.03 or bal_te <= 0.55)
    rec = {"signals": f"object={obj}, scene={scn}, vlm={vlm}",
           "decision_threshold": round(thr, 3),
           "test_error_rate": round(rate, 4),
           "predicts_positive_pct": round(100 * pos_rate, 1),
           "majority_class_error_rate": round(majority_err, 4),
           "balanced_accuracy_test": round(bal_te, 4),
           "classifier_degenerate": degenerate,
           "U_distinct_values": int(pd.Series(U).round(6).nunique()),
           "auroc_disagreement": None, "auroc_confidence": None}
    if len(np.unique(y)) > 1 and not rec["classifier_degenerate"]:
        rec["auroc_disagreement"] = round(float(roc_auc_score(y, u)), 4)
        rec["auroc_confidence"] = round(float(roc_auc_score(y, c)), 4)
        rng = np.random.default_rng(SEED); dif = []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) > 1:
                dif.append(roc_auc_score(y[i], u[i]) - roc_auc_score(y[i], c[i]))
        o = np.array(dif)
        rec["disagreement_minus_confidence"] = {
            "delta": round(float(o.mean()), 4),
            "ci95": [round(float(np.percentile(o, 2.5)), 4),
                     round(float(np.percentile(o, 97.5)), 4)],
            "disagreement_wins": bool(np.percentile(o, 2.5) > 0)}
    elif rec["classifier_degenerate"]:
        rec["note"] = ("degenerate: predicts one class on %.1f%% of images, balanced accuracy "
                       "%.4f; the AUROC comparison is not usable" % (100 * pos_rate, bal_te))
    out[name] = rec


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
    sp = pd.read_csv(SPLIT)
    ag = pd.read_csv(AGENTS).drop_duplicates("image_id")
    j = sp.merge(ag, left_on="ppn_id", right_on="image_id", how="inner").merge(
        d, on="image_id", how="inner")

    out = {"experiment": "Signals redesigned to use what the models actually emit",
           "n_corpus": int(len(d)), "n_labelled": int(len(j)),
           "signal_richness": {
               "A_object deployed min(k/3,1)": 4,
               "A_object redesigned (detector conf_max)": int(d.A_object.round(6).nunique()),
               "A_scene deployed": 2,
               "A_scene redesigned (max softmax over 5 categories)": int(d.max_softmax.round(6).nunique()),
               "A_scene redesigned (classifier peakedness)": int(d.peakedness.round(6).nunique()),
               "A_vlm deployed": 2,
               "A_vlm redesigned (fraction of 10 classes named)": int(d.A_vlm_graded.round(6).nunique())}}

    res = {}
    # deployed baseline, for reference, on the same partition
    D = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    saa_d = 0.6 * j.existing_pipeline_agent + 0.2 * j.scene_agent + 0.2 * j.vlm_agent
    err_d = ((saa_d >= 0.5).astype(int) != j.has_scene).astype(int)
    te = j.split.values == "test"
    U_d = j[D].std(axis=1, ddof=1).values
    pr_d = (saa_d >= 0.5).astype(int).values[te]; h_d = j.has_scene.values[te]
    tp = int(((pr_d == 1) & (h_d == 1)).sum()); fn = int(((pr_d == 0) & (h_d == 1)).sum())
    tn = int(((pr_d == 0) & (h_d == 0)).sum()); fp = int(((pr_d == 1) & (h_d == 0)).sum())
    bal_d = ((tp / (tp + fn) if tp + fn else 0) + (tn / (tn + fp) if tn + fp else 0)) / 2
    res["deployed (reference)"] = {
        "signals": "as shipped",
        "test_error_rate": round(float(err_d[te].mean()), 4),
        "predicts_positive_pct": round(100 * float(pr_d.mean()), 1),
        "balanced_accuracy_test": round(bal_d, 4),
        "U_distinct_values": int(pd.Series(U_d).round(6).nunique()),
        "auroc_disagreement": round(float(roc_auc_score(err_d[te], U_d[te])), 4),
        "auroc_confidence": round(float(roc_auc_score(err_d[te], (1 - saa_d)[te])), 4)}

    evaluate(j, "A_object", "max_softmax", "A_vlm_graded", "redesigned (max softmax)", res)
    evaluate(j, "A_object", "peakedness", "A_vlm_graded", "redesigned (peakedness)", res)
    out["results"] = res

    best = max((v for v in res.values() if v.get("auroc_disagreement")),
               key=lambda v: v["auroc_disagreement"])
    out["reading"] = (
        "Every redesigned signal is far richer than the deployed one it replaces. Whether that "
        "richness buys anything is the question: the comparison to watch is disagreement against "
        "the plain confidence baseline within each row, not across rows, since each variant has "
        "its own decision threshold and therefore its own error label.")

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
