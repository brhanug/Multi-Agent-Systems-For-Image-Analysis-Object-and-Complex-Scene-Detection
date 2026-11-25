#!/usr/bin/env python3
"""Run a real detector on all 12,110 images and retest the central claim.

The standing objection to this thesis's negative result is that it rests on a defect:
A_object was never the detector's box count, because student_v3_count is zero on all
12,110 manifest rows and the fusion stage fell back to the VQA object count
(Section 2.4). Fix the provenance, the objection goes, and disagreement will work.

That is now testable rather than arguable. run_yolo_full_corpus.py executed the trained
YOLOv11m archival-taxonomy detector over every one of the 12,110 corpus images -- a real
detector, real inference, no fallback. This retests the claim on that signal.

Caveat stated up front: the distilled student_iter_3 checkpoint is no longer on disk, so
this is not that model. It is a genuine detector trained on the archival taxonomy, which
is what the question requires; it is not a reconstruction of the historical student, and
no claim is made that its classification quality matches one tuned for the deployed
pipeline.

Three object signals are compared: the deployed fallback, the real detector under the
deployed min(count/3, 1) transform, and the real detector left continuous -- the last
isolating whether the saturation or the provenance is what binds.

Outputs: results/signal_rebuild/real_detector_retest.json
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
BASE = Path("/data/brhanu/thesis_project")
YOLO_JSONL = BASE / "results/signal_rebuild/yolo_full_corpus.jsonl"
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
OUT = BASE / "results/signal_rebuild/real_detector_retest.json"
O, A, S, V = "existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"


def boot(y, sa, sb, n=2000, seed=20260910):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            d.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
    o = np.array(d)
    return {"delta": round(float(o.mean()), 4),
            "ci95": [round(float(np.percentile(o, 2.5)), 4),
                     round(float(np.percentile(o, 97.5)), 4)]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    y = pd.DataFrame([json.loads(l) for l in open(YOLO_JSONL)]).drop_duplicates("image_id")
    ag = pd.read_csv(AGENTS).drop_duplicates("image_id")
    corpus = ag.merge(y[["image_id", "n_box", "conf_mean"]], on="image_id", how="inner")

    out = {"experiment": "Real detector run corpus-wide, and the retest it enables",
           "detector": "YOLOv11m, runs/detect/yolo11_v2_augmented_refresh (10-class archival taxonomy)",
           "images_scored": int(len(y)),
           "caveat": ("The distilled student_iter_3 checkpoint is absent from disk, so this is a "
                      "genuine archival-taxonomy detector rather than a reconstruction of the "
                      "historical student.")}

    corpus["O_real"] = np.minimum(corpus.n_box / 3, 1.0)
    out["corpus_signal_comparison"] = {
        "n": int(len(corpus)),
        "deployed_A_object_distinct_values": int(corpus[O].nunique()),
        "real_A_object_distinct_values": int(corpus.O_real.nunique()),
        "deployed_zero_on": int((corpus[O] == 0).sum()),
        "real_zero_on": int((corpus.O_real == 0).sum()),
        "pearson_r": round(float(np.corrcoef(corpus.O_real, corpus[O])[0, 1]), 4),
        "exactly_equal_pct": round(100 * float((corpus.O_real == corpus[O]).mean()), 1),
        "presence_call_agreement_pct": round(
            100 * float(((corpus.O_real > 0) == (corpus[O] > 0)).mean()), 1),
        "dependency_A_vlm_eq_indicator": {
            "deployed_pct": round(100 * float((corpus[V] == (corpus[O] > 0).astype(float)).mean()), 1),
            "real_pct": round(100 * float((corpus[V] == (corpus.O_real > 0).astype(float)).mean()), 1)},
    }

    sp = pd.read_csv(SPLIT)
    j = sp.merge(corpus, left_on="ppn_id", right_on="image_id", how="inner")
    j["A_real"] = 1 - (j.O_real - j[V]).abs()
    # continuous variant: no saturation, scaled by corpus max
    j["O_cont"] = j.n_box / max(float(corpus.n_box.max()), 1.0)
    j["A_cont"] = 1 - (j.O_cont - j[V]).abs()

    variants = {
        "deployed_fallback": (j[O], j[A]),
        "real_detector_saturated": (j.O_real, j.A_real),
        "real_detector_continuous": (j.O_cont, j.A_cont),
    }
    te_mask = j.split == "test"
    res = {}
    for name, (obj, agr) in variants.items():
        saa = 0.6 * obj + 0.2 * j[S] + 0.2 * j[V]
        U = pd.concat([obj, agr, j[S], j[V]], axis=1).std(axis=1, ddof=1)
        err = ((saa >= 0.5).astype(int) != j.has_scene).astype(int)
        te = te_mask.values
        ye, Ue, ce = err[te].values, U[te].values, (1 - saa[te]).values
        if len(np.unique(ye)) < 2:
            res[name] = {"note": "single class"}; continue
        rate = float(ye.mean())
        degenerate = rate > 0.5   # worse than always predicting the majority class
        res[name] = {
            "test_n": int(te.sum()), "test_errors": int(ye.sum()),
            "test_error_rate": round(rate, 4),
            "classifier_degenerate": degenerate,
            "U_SAA_distinct_values": int(U.round(6).nunique()),
            "auroc_disagreement": round(float(roc_auc_score(ye, Ue)), 4),
            "auroc_confidence": round(float(roc_auc_score(ye, ce)), 4),
            "disagreement_minus_confidence": boot(ye, Ue, ce),
        }
        if degenerate:
            res[name]["interpretation"] = (
                "NOT USABLE for comparing uncertainty scores. Scaling the box count by the "
                "corpus maximum puts almost every image below the 0.5 threshold, so the "
                "classifier predicts scene-absent almost everywhere and errs on "
                f"{100*rate:.0f}% of the test partition. The confidence baseline scores below "
                "chance here for the same reason, which is what makes disagreement appear to "
                "win. The apparent win is an artifact of a broken classifier, not evidence.")
    out["retest_on_labelled_pool"] = res
    out["conclusion"] = (
        "Running a real detector over the whole corpus breaks the perfect dependency -- "
        "A_vlm = 1[A_object>0] falls from 100% to 75% of images -- and the two object signals "
        "agree exactly on only a quarter of the corpus, so the provenance defect was material. "
        "It does not rescue the central claim. On the real detector under the deployed transform, "
        "disagreement scores 0.8304 against confidence 0.8674: still behind, and the difference "
        "is no longer statistically distinguishable (CI spans zero), which is weaker evidence "
        "than the deployed comparison rather than stronger. The continuous variant is reported "
        "but must not be read as a disagreement win -- its classifier errs on 82% of the test "
        "partition, and confidence falls below chance for the same reason. The negative result "
        "therefore survives replacing the fallback with real corpus-wide inference; what it does "
        "not survive is being called precise, since the real-detector comparison is "
        "underpowered at n=249.")

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
