#!/usr/bin/env python3
"""B1: each system, and a per-class fusion, on the 300 human-annotated pictures.

Systems: the detector (top box confidence per class), SigLIP (per-class prompt score),
LLaVA-OneVision (P(yes) to one question per class), Kosmos-2.5 (word-like tokens it reads;
text only), and a fusion (mean of the three systems' within-corpus percentiles per class).

Per class: AUROC, and precision / recall / F1 at a cut-off chosen by 5-fold
cross-validation -- the F1-best cut-off on four folds, applied to the fifth -- so no
picture is scored with a threshold fitted on itself. Where a system has a natural rule
(detector 0.25, LLaVA 0.5, Kosmos 3 words) that rule is reported too.
Macro averages over the classes a system can score, and over the 8 classes all three
image systems share. Split by the rater's clarity label (Clear vs Blurry/Faded), and for
the detector by whether the picture was in its training split.

    python run_b1_per_system_table.py [--verify]
"""
import argparse, glob, json, re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE = Path("/data/brhanu/thesis_project")
CS = BASE / "results/signal_rebuild/class_scores"
OUT = BASE / "results/signal_rebuild/b1/b1_per_system.json"
SHARED = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]
CLASSES = SHARED + ["furniture", "hat"]
WORD = re.compile(r"^[A-Za-zÄÖÜäöüß]{2,20}$")
RULES = {"detector": 0.25, "llava": 0.5, "kosmos": 3}
SEED = 20260917


def prf(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def best_cut(s, y):
    cand = np.unique(s); best, bf = cand[0], -1.0
    for c in cand:
        pr = s >= c; tp = (pr & (y == 1)).sum(); f = 2 * tp / (pr.sum() + y.sum()) if pr.sum() + y.sum() else 0
        if f > bf: best, bf = c, f
    return best


def cv_predict(s, y):
    pred = np.zeros(len(y), dtype=int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(s, y):
        pred[te] = (s[te] >= best_cut(s[tr], y[tr])).astype(int)
    return pred


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    m = np.load(CS / "class_score_matrix.npz", allow_pickle=True)
    ids = [str(x) for x in m["ids"]]; pos = {i: k for k, i in enumerate(ids)}
    mcls, msys = [str(x) for x in m["classes"]], [str(x) for x in m["systems"]]
    truth = pd.read_csv(CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    ann = pd.read_csv(BASE / "human_spatial_audit/user_annotations_300.csv")
    clarity = dict(zip(ann.image_id.str.replace("_", "/", n=1), ann.clarity))
    rows = np.array([pos[i] for i in truth.index])
    blurry = np.array([clarity[i] != "Clear" for i in truth.index]); unseen = truth.in_detector_train.values == 0
    kos = {}
    for f in sorted(glob.glob(str(BASE / "results/signal_rebuild/kosmos_full/shard_*.jsonl"))):
        for line in open(f):
            r = json.loads(line); t = r.get("kosmos_output") or ""
            kos[r["image_id"]] = sum(bool(WORD.match(w.strip("#*>[]()|-.,:;"))) for w in t.split())
    scores = {}
    for c in CLASSES:
        ci = mcls.index(c)
        for s in ("detector", "siglip", "llava"):
            v = m["raw"][rows, msys.index(s), ci].astype(np.float64)
            if not np.isnan(v).any(): scores[(s, c)] = v
        pc = m["percentile"][rows][:, :, ci].astype(np.float64)
        scores[("fusion", c)] = np.nanmean(pc, axis=1)
    scores[("kosmos", "text")] = np.array([kos.get(i, 0) for i in truth.index], dtype=np.float64)
    systems = ["detector", "siglip", "llava", "kosmos", "fusion"]
    table, cvpred = [], {}
    for (s, c), v in scores.items():
        y = truth[c].values
        pred = cv_predict(v, y); cvpred[(s, c)] = pred
        row = {"system": s, "class": c, "positives": int(y.sum()), "auroc": round(float(roc_auc_score(y, v)), 4), "cv": prf(pred, y),
               "always_yes_f1": round(2 * y.mean() / (1 + y.mean()), 4)}
        if pred.all(): row["note"] = "cut-off calls every picture positive: the F1 is the always-yes baseline, not detection"
        if s in RULES: row["fixed_rule"] = {"cutoff": RULES[s], **prf((v >= RULES[s]).astype(int), y)}
        table.append(row)
    macro = {}
    for s in systems:
        mine = [r for r in table if r["system"] == s]
        if not mine: continue
        e = {"classes_scored": len(mine), "auroc": round(float(np.mean([r["auroc"] for r in mine])), 4),
             "cv_f1": round(float(np.mean([r["cv"]["f1"] for r in mine])), 4)}
        sh = [r for r in mine if r["class"] in SHARED]
        if len(sh) == len(SHARED):
            e["shared8_auroc"] = round(float(np.mean([r["auroc"] for r in sh])), 4)
            e["shared8_cv_f1"] = round(float(np.mean([r["cv"]["f1"] for r in sh])), 4)
            for tag, mask in (("clear", ~blurry), ("blurry_or_faded", blurry)):
                fs = [prf(cvpred[(s, c)][mask], truth[c].values[mask])["f1"] for c in SHARED if truth[c].values[mask].sum()]
                e[f"shared_cv_f1_{tag}"] = round(float(np.mean(fs)), 4)
            if s == "detector":
                for tag, mask in (("in_training", ~unseen), ("not_in_training", unseen)):
                    fs = [prf(cvpred[(s, c)][mask], truth[c].values[mask])["f1"] for c in SHARED if truth[c].values[mask].sum()]
                    e[f"shared_cv_f1_{tag}"] = round(float(np.mean(fs)), 4)
        if "fixed_rule" in mine[0]:
            e["fixed_rule_f1"] = round(float(np.mean([r["fixed_rule"]["f1"] for r in mine])), 4)
        macro[s] = e
    base = {c: round(2 * truth[c].mean() / (1 + truth[c].mean()), 4) for c in CLASSES}
    macro["always_yes_baseline"] = {"cv_f1": round(float(np.mean(list(base.values()))), 4),
                                    "shared8_cv_f1": round(float(np.mean([base[c] for c in SHARED])), 4)}
    out = {"experiment": "B1: each system and a per-class fusion on the 300 human-annotated pictures", "pictures": len(rows),
           "clarity_groups": {"clear": int((~blurry).sum()), "blurry_or_faded": int(blurry.sum())},
           "cutoffs": "5-fold stratified cross-validation, F1-best cut-off on training folds", "macro": macro, "per_class": table}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out, indent=2))
    print("MACRO"); [print(f"  {s:9s} {e}") for s, e in macro.items()]
    print("\nF1 (cross-validated cut-off) per class")
    print("  " + "class".ljust(10) + "".join(s.rjust(10) for s in systems))
    for c in CLASSES:
        cells = {r["system"]: r["cv"]["f1"] for r in table if r["class"] == c}
        print(f"  {c:10s}" + "".join((f"{cells[s]:.2f}" if s in cells else "—").rjust(10) for s in systems))


if __name__ == "__main__":
    raise SystemExit(main())
