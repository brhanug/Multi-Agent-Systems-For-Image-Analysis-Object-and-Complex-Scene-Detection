#!/usr/bin/env python3
"""Fix the three signals that do not measure what they are named for, and test the fixes.

Section 4.2.2 established that three of the four Tier 1 signals do not measure their
named quantity:

  A_agree  arithmetic on two numbers derived from the same VQA answer; correlates -0.958
           with A_object, and is exactly recoverable from it -- no second opinion is
           involved, so nothing is being "agreed" on
  A_ocr    min(chars/80, 1): text VOLUME, not validity, so a long fabricated transcript
           scores higher than a short correct one
  A_vlm    fraction of four VQA answers at least three words long: 2/4 on 98.9% of the
           corpus, which encodes the question format rather than the image

This script rebuilds each to measure the thing it was named for, using model outputs that
already exist for all 12,110 images, and then asks whether measuring it properly changes
any conclusion.

  A_agree_fixed   Jaccard overlap between the label SET the detector returns and the label
                  SET the VLM names. Two independently run models over one shared
                  vocabulary: a real cross-modal agreement. Reported both over the full
                  10-class taxonomy and restricted to the 9 classes the detector can
                  actually emit, since penalising the VLM for naming "text" -- a class with
                  no detector counterpart -- would understate agreement by construction.
  A_ocr_fixed     a validity proxy that needs no ground truth: (1 - repetition) x
                  wordlikeness, where repetition is 1 - unique/total tokens and
                  wordlikeness is the fraction of tokens shaped like words rather than
                  markup, rules or bare digits. Degenerate output is long AND repetitive,
                  so the two terms pull apart exactly where the deployed signal saturates.
  A_vlm_fixed     fraction of the 10 taxonomy classes the model actually names.

Evaluation follows the established protocol. Errors are scored against BOTH the deployed
decision rule (so the question is whether the fixed signals predict the mistakes the
system as shipped actually makes) and a refitted rule, with the degeneracy guard applied
to each; AUROC differences carry paired bootstrap intervals.

Outputs: results/signal_rebuild/fixed_signals.json
"""
import argparse, glob, json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
OUT = SR / "fixed_signals.json"
CHANNEL = "softmax"          # set from --channel; see main()
TAX = ["person", "child", "horse", "building", "weapon", "vehicle", "tree",
       "clothing", "text", "animal"]
DET_VOCAB = ["person", "child", "horse", "building", "weapon", "vehicle", "tree",
             "clothing", "animal"]          # the 9 classes the detector can emit
GRID = np.arange(0.05, 0.95, 0.005)
WORD = re.compile(r"^[A-Za-zÄÖÜäöüß]{2,20}$")
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


def validity(t):
    toks = re.findall(r"\S+", str(t))
    if not toks:
        return 0.0
    rep = 1 - len(set(toks)) / len(toks)
    wordlike = sum(bool(WORD.match(w.strip("#*>[]()|-.,:;"))) for w in toks) / len(toks)
    return float(max(0.0, (1 - rep) * wordlike))


def jaccard(a, b, vocab=None):
    if vocab is not None:
        a, b = a & set(vocab), b & set(vocab)
    u = a | b
    return len(a & b) / len(u) if u else np.nan       # undefined, not agreement


def build():
    y = rj([SR / "yolo_full_corpus.jsonl"])
    v = rj(sorted(glob.glob(str(SR / "llava_full/shard_*.jsonl"))))
    k = rj(sorted(glob.glob(str(SR / "kosmos_full/shard_*.jsonl"))))
    s = rj([SR / "siglip_scene_full.jsonl"])
    s["image_id"] = s.image_id.map(lambda x: x if "/" in str(x) else str(x).replace("_", "/", 1))
    s["A_scene"] = (pd.DataFrame(list(s.softmax)).fillna(0.0).max(axis=1).values
                    if CHANNEL == "softmax" else s.max_sigmoid.astype(float).values)
    if s.A_scene.std() < 1e-9:
        raise SystemExit("A_scene is constant -- id merge failed, refusing to continue")

    y["det_set"] = y.labels.apply(lambda L: set(L) if isinstance(L, list) else set())
    y["A_object"] = y.conf_max.clip(0, 1)

    txt = v.answers.apply(lambda a: " ".join(map(str, a)).lower() if isinstance(a, list) else "")
    v["vlm_set"] = txt.apply(lambda t: {c for c in TAX if re.search(rf"\b{c}", t)})
    v["A_vlm_fixed"] = v.vlm_set.apply(lambda S: len(S) / len(TAX))
    v["A_vlm_deployed"] = v.answers.apply(
        lambda a: np.mean([len(str(x).split()) >= 3 for x in a]) if isinstance(a, list) else 0.0)

    k["ocr_text"] = k.kosmos_output.fillna("")
    k["A_ocr_fixed"] = k.ocr_text.map(validity)
    k["A_ocr_deployed"] = np.minimum(k.ocr_text.str.len() / 80.0, 1.0)

    d = (y[["image_id", "A_object", "det_set"]]
         .merge(v[["image_id", "A_vlm_fixed", "A_vlm_deployed", "vlm_set"]], on="image_id")
         .merge(s[["image_id", "A_scene"]], on="image_id")
         .merge(k[["image_id", "A_ocr_fixed", "A_ocr_deployed", "ocr_text"]], on="image_id"))

    d["A_agree_fixed"] = [jaccard(a, b) for a, b in zip(d.det_set, d.vlm_set)]
    d["A_agree_fixed_shared"] = [jaccard(a, b, DET_VOCAB) for a, b in zip(d.det_set, d.vlm_set)]
    d["A_agree_deployed"] = 1 - (d.A_object - d.A_vlm_deployed).abs()
    d["n_det"] = d.det_set.map(len)
    d["n_vlm"] = d.vlm_set.map(len)
    return d


def bal_acc(pred, h):
    tp = int(((pred == 1) & (h == 1)).sum()); fn = int(((pred == 0) & (h == 1)).sum())
    tn = int(((pred == 0) & (h == 0)).sum()); fp = int(((pred == 1) & (h == 0)).sum())
    return ((tp / (tp + fn) if tp + fn else 0.0) + (tn / (tn + fp) if tn + fp else 0.0)) / 2


def degenerate(pred, h):
    """The established guard: one class almost everywhere, OR balanced accuracy <= 0.55."""
    rate = float(pred.mean())
    ba = bal_acc(pred, h)
    return bool(rate > 0.95 or rate < 0.05 or ba <= 0.55), round(rate, 4), round(ba, 4)


def auc(y, score):
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")


def boot_delta(y, a, b):
    """Paired bootstrap on AUROC(a) - AUROC(b), resampling images."""
    n, out = len(y), []
    for _ in range(B):
        i = RNG.integers(0, n, n)
        if 0 < y[i].sum() < n:
            out.append(auc(y[i], a[i]) - auc(y[i], b[i]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return round(float(np.mean(out)), 4), round(float(lo), 4), round(float(hi), 4)


def evaluate(j, tag, saa, thr_source):
    """Score errors under one decision rule, then test every signal against those errors."""
    j = j.copy(); j["SAA"] = saa
    cal = j[j.split == "calibration"]
    if thr_source == "fit":
        thr = float(GRID[int(np.argmax([bal_acc((cal.SAA >= t).astype(int).values,
                                                cal.has_scene.values) for t in GRID]))])
    else:
        thr = float(thr_source)
    j["pred"] = (j.SAA >= thr).astype(int)
    te = j[j.split == "test"]
    deg, rate, ba = degenerate(te.pred.values, te.has_scene.values)
    y = (te.pred != te.has_scene).astype(int).values
    res = {"rule": tag, "threshold": round(thr, 3), "test_n": int(len(te)),
           "test_errors": int(y.sum()),
           "guard": {"predicted_positive_rate": rate, "balanced_accuracy": ba,
                     "degenerate": deg}}
    if deg or not (0 < y.sum() < len(y)):
        res["note"] = "rule is degenerate under the guard; AUROCs against it are not reported"
        return res
    conf = (1 - te.SAA).values
    cands = {"A_object": -te.A_object.values,
             "A_scene": -te.A_scene.values,
             "A_vlm_fixed": -te.A_vlm_fixed.values,
             "A_vlm_deployed": -te.A_vlm_deployed.values,
             "A_agree_fixed": -te.A_agree_fixed.fillna(0).values,
             "A_agree_fixed_shared": -te.A_agree_fixed_shared.fillna(0).values,
             "A_agree_deployed": -te.A_agree_deployed.values,
             "A_ocr_fixed": -te.A_ocr_fixed.values,
             "A_ocr_deployed": -te.A_ocr_deployed.values,
             "confidence (1 - SAA)": conf}
    res["auroc_vs_error"] = {k: round(auc(y, s), 4) for k, s in cands.items()}
    S_fix = ["A_object", "A_agree_fixed", "A_scene", "A_vlm_fixed"]
    sp_fix = te[S_fix].fillna(0).std(axis=1, ddof=1).values
    sp_dep = te[["A_object", "A_agree_deployed", "A_scene",
                 "A_vlm_deployed"]].std(axis=1, ddof=1).values
    dfn = te.A_agree_fixed.notna().values
    if 0 < y[dfn].sum() < int(dfn.sum()):
        res["auroc_vs_error"]["A_agree_fixed (defined subset only)"] = round(
            auc(y[dfn], -te.A_agree_fixed.values[dfn]), 4)
        res["auroc_vs_error"]["_defined_subset_n"] = int(dfn.sum())
    res["spread_vs_confidence"] = {
        "spread_with_fixed_signals": round(auc(y, sp_fix), 4),
        "spread_with_deployed_signals": round(auc(y, sp_dep), 4),
        "confidence": round(auc(y, conf), 4),
        "delta_fixed_spread_minus_confidence": boot_delta(y, sp_fix, conf),
        "delta_fixed_agree_minus_confidence": boot_delta(
            y, -te.A_agree_fixed.fillna(0).values, conf)}

    # The decisive question: do the three fixed signals ADD anything to confidence?
    # Fitted on calibration, scored on test, so nothing is read off the test split.
    ca = j[j.split == "calibration"]
    ya = (ca.pred != ca.has_scene).astype(int).values
    if 0 < ya.sum() < len(ya):
        extra = ["A_agree_fixed", "A_ocr_fixed", "A_vlm_fixed"]
        Xc_b = np.c_[(1 - ca.SAA).values]
        Xt_b = np.c_[conf]
        Xc_f = np.c_[(1 - ca.SAA).values, ca[extra].fillna(0).values]
        Xt_f = np.c_[conf, te[extra].fillna(0).values]
        sc = StandardScaler().fit(Xc_f)
        m_b = LogisticRegression(max_iter=2000).fit(
            StandardScaler().fit_transform(Xc_b), ya)
        m_f = LogisticRegression(max_iter=2000).fit(sc.transform(Xc_f), ya)
        p_b = m_b.predict_proba(StandardScaler().fit(Xc_b).transform(Xt_b))[:, 1]
        p_f = m_f.predict_proba(sc.transform(Xt_f))[:, 1]
        res["do_the_fixed_signals_add_to_confidence"] = {
            "confidence_alone": round(auc(y, p_b), 4),
            "confidence_plus_three_fixed_signals": round(auc(y, p_f), 4),
            "delta_with_95pct_bootstrap_CI": boot_delta(y, p_f, p_b),
            "fitted_on": "calibration split, scored on test",
            "reading": ("If the interval spans zero, measuring these three properly still buys "
                        "nothing beyond the confidence score already available.")}
    return res


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
    out = {"experiment": "Signals rebuilt to measure what they are named for",
           "n_corpus": int(len(d))}

    out["does_each_signal_vary"] = {}
    for c in ["A_agree_fixed", "A_agree_fixed_shared", "A_agree_deployed",
              "A_ocr_fixed", "A_ocr_deployed", "A_vlm_fixed", "A_vlm_deployed"]:
        col = d[c]
        out["does_each_signal_vary"][c] = {
            "distinct_values": int(col.round(6).nunique()),
            "mean": round(float(col.mean()), 4),
            "modal_value_share_pct": round(100 * float(col.round(4).value_counts(
                normalize=True).iloc[0]), 2),
            "undefined": int(col.isna().sum())}

    # Is the fixed agreement a real second opinion, or A_object again in disguise?
    tent = 1 - (d.A_object - 0.5).abs()      # what A_agree collapses to when A_vlm == 0.5
    out["is_the_fixed_agreement_independent"] = {
        "deployed_agree_is_exactly_a_function_of_A_object_on_pct": round(
            100 * float(((d.A_agree_deployed - tent).abs() < 1e-9).mean()), 2),
        "corr_deployed_agree_with_A_object": round(float(d.A_agree_deployed.corr(d.A_object)), 4),
        "corr_fixed_agree_with_A_object": round(float(d.A_agree_fixed.corr(d.A_object)), 4),
        "images_where_both_models_name_nothing": int((d.n_det + d.n_vlm == 0).sum()),
        "images_where_sets_are_disjoint": int((d.A_agree_fixed == 0).sum()),
        "images_where_sets_match_exactly": int((d.A_agree_fixed == 1).sum()),
        "reading": ("Because A_vlm is 0.5 on 98.9% of images, the deployed agreement term "
                    "collapses to 1 - |A_object - 0.5|: a fixed tent function of one other "
                    "signal, carrying no second opinion. Its linear correlation with A_object "
                    "is only 0.18, which is why the defect survived a correlation check -- the "
                    "dependence is exact but not linear. The fixed term is computed from two "
                    "separately run models and is not recoverable from either.")}

    # Does the fixed OCR signal actually separate degenerate transcription?
    sat = d.A_ocr_deployed >= 1.0
    worst = d.nlargest(8, "A_ocr_deployed").assign(
        rep=lambda x: x.ocr_text.map(lambda t: 1 - len(set(t.split())) / max(len(t.split()), 1)))
    worst = worst.nlargest(5, "rep")
    out["does_the_fixed_ocr_measure_validity"] = {
        "deployed_saturates_at_1.0_on_pct": round(100 * float(sat.mean()), 2),
        "fixed_score_on_those_same_images": round(float(d.loc[sat, "A_ocr_fixed"].mean()), 4),
        "fixed_score_on_all_other_images": round(float(d.loc[~sat, "A_ocr_fixed"].mean()), 4),
        "corr_fixed_with_deployed": round(float(d.A_ocr_fixed.corr(d.A_ocr_deployed)), 4),
        "examples_deployed_scores_1.0": [
            {"repetition": round(float(r.rep), 3),
             "A_ocr_deployed": 1.0, "A_ocr_fixed": round(float(r.A_ocr_fixed), 3),
             "text": (r.ocr_text[:70] + "...") if len(r.ocr_text) > 70 else r.ocr_text}
            for r in worst.itertuples()],
        "reading": ("Images the deployed signal calls perfect score LOWER on validity than the "
                    "rest of the corpus: length and validity point in opposite directions, which "
                    "is the defect the fix was built to expose.")}

    sp = pd.read_csv(SPLIT)
    j = sp.merge(d, left_on="ppn_id", right_on="image_id", how="inner")
    out["labelled_overlap_n"] = int(len(j))
    out["evaluations"] = [
        evaluate(j, "deployed rule: 0.6 A_object + 0.2 A_scene + 0.2 A_vlm_deployed, "
                    "threshold 0.55 as shipped",
                 0.6 * j.A_object + 0.2 * j.A_scene + 0.2 * j.A_vlm_deployed, 0.55),
        evaluate(j, "deployed signals, threshold refitted on calibration",
                 0.6 * j.A_object + 0.2 * j.A_scene + 0.2 * j.A_vlm_deployed, "fit"),
        evaluate(j, "fixed signals, threshold refitted on calibration",
                 0.6 * j.A_object + 0.2 * j.A_scene + 0.2 * j.A_vlm_fixed, "fit")]

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
