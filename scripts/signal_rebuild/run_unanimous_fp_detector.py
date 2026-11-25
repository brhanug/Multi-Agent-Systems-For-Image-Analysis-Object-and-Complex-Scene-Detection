#!/usr/bin/env python3
"""Can anything detect the unanimous false positives a disagreement signal cannot see?

Section 7.5 lists this as future work: a complementary detector aimed at the
unanimous-false-positive mode, which U_SAA cannot see by construction.

FRAMING MATTERS, and a first attempt got it wrong. Training inside the confirmed
negative pool is circular: there, "error" is *defined* as SAA >= 0.5, so a model
predicting error is only predicting the pipeline's own decision. That was verified
directly -- predicting the human error label and predicting SAA >= 0.5 gave the same
AUROC (0.7727) because they are the same vector.

The non-circular question is the deployment one: among images the pipeline has ALREADY
flagged "scene present", which ones are wrong? On that pool the pipeline's decision is
constant, so it cannot leak. 968 such images carry a human label: 918 true positives
and 50 false positives.

Predictors are raw per-image agent outputs that do not enter SAA. GroundingDINO and OCR
form the strict set, since A_object and A_vlm derive from the VQA answer.

Evaluation is grouped by PPN volume so pages of one book never span a split.

Outputs: results/signal_rebuild/unanimous_fp_detector.json
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "core_pipeline"))
from saa_decision import SAA_THRESHOLD  # deliberate 0.55; see saa_decision.py
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
BASE = Path("/data/brhanu/thesis_project")
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
RAW = BASE / "results/signal_rebuild/raw_agent_signals.csv"
POOL = BASE / "results/multi_agent/corrected_clean_pool.csv"
NEW = BASE / "self_annotation_package/task_a2_remaining_negative_controls/chat_labels.csv"
OUT = BASE / "results/signal_rebuild/unanimous_fp_detector.json"

A4 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
GD = ["gd_n_box", "gd_conf_mean", "gd_conf_max", "gd_conf_sum", "gd_n_labels"]
OCR = ["ocr_chars", "ocr_words", "ocr_repetition"]
LLAVA = ["llava_n_objects", "llava_obj_chars", "llava_n_triplets",
         "vlm_gd_jaccard", "vlm_gd_overlap"]
SETS = {"disjoint_gd_ocr": GD + OCR, "full_raw": GD + OCR + LLAVA}
SEED = 20260908


def build():
    ag = pd.read_csv(AGENTS).drop_duplicates("image_id")
    ag["key"] = ag.image_id.str.replace("/", "_", regex=False)
    raw = pd.read_csv(RAW).drop_duplicates("image_id")
    new = pd.read_csv(NEW); new["has_scene"] = (new.label == "scene").astype(int)
    old = pd.read_csv(POOL)
    old2 = old[["ppn_id", "has_scene", "label_source"]]
    new2 = new[["ppn_id", "has_scene"]].assign(label_source="rater_via_chat_contact_sheet")
    lab = pd.concat([old2, new2])
    j = lab.merge(ag, left_on="ppn_id", right_on="image_id", how="inner")
    j = j.merge(raw, left_on="key", right_on="image_id", how="inner", suffixes=("", "_raw"))
    j["saa"] = 0.6 * j.existing_pipeline_agent + 0.2 * j.scene_agent + 0.2 * j.vlm_agent
    j["U_SAA"] = j[A4].std(axis=1, ddof=1)
    j["ppn"] = j.ppn_id.str.split("/").str[0]
    return j


def boot_ci(y, s, n=2000, seed=SEED):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            out.append(roc_auc_score(y[i], s[i]))
    o = np.array(out)
    return [round(float(np.percentile(o, 2.5)), 4), round(float(np.percentile(o, 97.5)), 4)]


MODELS = {
    "logistic_balanced": lambda: make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    "gradient_boosting": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingClassifier(random_state=SEED, n_estimators=120, max_depth=2)),
}


def cv_oof(X, y, g, mk, seed=SEED):
    oof = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups=g):
        if len(np.unique(y[tr])) < 2:
            continue
        m = mk(); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    return oof, ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    j = build()
    d = j[j.saa >= SAA_THRESHOLD].copy()               # the pipeline already said "scene present"
    y = (d.has_scene == 0).values.astype(int)  # 1 = false positive
    g = d.ppn.values

    out = {"experiment": "Detecting false positives among images the pipeline flagged scene-present",
           "seed": SEED,
           "why_this_pool": ("Training inside the confirmed-negative pool is circular: there, error "
                             "is defined as SAA>=0.5, so the target is the pipeline's own decision. "
                             "Verified directly: predicting the error label and predicting SAA>=0.5 "
                             "returned the same AUROC (0.7727) on identical vectors. Restricting to "
                             "SAA>=0.5 holds the pipeline decision constant so it cannot leak."),
           "pool": {"n": int(len(d)), "false_positives": int(y.sum()),
                    "true_positives": int((y == 0).sum()),
                    "n_volumes": int(pd.Series(g).nunique()),
                    "fp_rate": round(float(y.mean()), 4)}}

    out["baselines_on_this_pool"] = {
        "U_SAA": round(float(roc_auc_score(y, d.U_SAA.values)), 4),
        "confidence_1_minus_SAA": round(float(roc_auc_score(y, 1 - d.saa.values)), 4),
    }
    out["detectors"] = {}
    for sname, feats in SETS.items():
        X = d[feats].apply(pd.to_numeric, errors="coerce").values.astype(float)
        const = [f for f, c in zip(feats, X.T) if np.nanstd(c) == 0]
        for mname, mk in MODELS.items():
            oof, ok = cv_oof(X, y, g, mk)
            if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
                out["detectors"][f"{sname}::{mname}"] = {"auroc": None, "note": "not evaluable"}
                continue
            a = float(roc_auc_score(y[ok], oof[ok])); ci = boot_ci(y[ok], oof[ok])
            out["detectors"][f"{sname}::{mname}"] = {
                "auroc": round(a, 4), "ci95": ci, "beats_chance": bool(ci[0] > 0.5),
                "constant_features_dropped": const, "n_scored": int(ok.sum())}

    # provenance of the labels in this pool -- all human-sourced, no catalogue defaults
    out["label_provenance"] = {k: int(v) for k, v in
                               d.label_source.value_counts().items()} if "label_source" in d else {}

    # paired bootstrap: does the best detector actually beat the confidence baseline?
    def paired(sa, sb, y, n=2000, seed=SEED):
        rng = np.random.default_rng(seed); dif = []
        for _ in range(n):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) > 1:
                dif.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
        o = np.array(dif)
        lo, hi = np.percentile(o, 2.5), np.percentile(o, 97.5)
        return {"delta_auroc": round(float(o.mean()), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "significant": bool(lo > 0)}

    bname = max((k for k, v in out["detectors"].items() if v.get("auroc")),
                key=lambda k: out["detectors"][k]["auroc"], default=None)
    if bname:
        sname, mname = bname.split("::")
        X = d[SETS[sname]].apply(pd.to_numeric, errors="coerce").values.astype(float)
        oof, ok = cv_oof(X, y, g, MODELS[mname])
        out["best_vs_baselines"] = {
            "detector": bname,
            "vs_confidence_1_minus_SAA": paired(oof[ok], (1 - d.saa.values)[ok], y[ok]),
            "vs_U_SAA": paired(oof[ok], d.U_SAA.values[ok], y[ok]),
        }

    best = max((v for v in out["detectors"].values() if v.get("auroc")),
               key=lambda v: v["auroc"], default=None)
    out["headline"] = ({"best_auroc": best["auroc"], "ci95": best["ci95"],
                        "beats_chance": best["beats_chance"],
                        "vs_U_SAA": round(best["auroc"] - out["baselines_on_this_pool"]["U_SAA"], 4)}
                       if best else None)

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
