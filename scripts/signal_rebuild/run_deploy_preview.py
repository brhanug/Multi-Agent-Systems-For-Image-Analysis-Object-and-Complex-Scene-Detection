#!/usr/bin/env python3
"""What changes if the rebuilt detector replaces the deployed one.

The rebuilt student is better at boxes (mean AP50 0.210 against 0.123). Deploying it would also move
every result that reads the detector, so this measures that before anything is switched: the same
analyses as this session's, with only the detector's per-class column swapped.

  per class on the 300   AUROC and F1 at a cross-validated cut-off, detector and fusion
  agreement (C2, C3)     mean rank agreement and Fleiss' kappa per class
  the C4 signal          per-class disagreement against plain confidence, DeLong interval

The scene-level pipeline (its fusion, the triage score, conformal risk control) is NOT recomputed:
that needs the whole pipeline re-run, which is what deploying means. Nothing here is deployed.

    python run_deploy_preview.py [--verify]
"""
import argparse, json, sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve

BASE = Path("/data/brhanu/thesis_project")
sys.path.insert(0, str(BASE / "scripts/signal_rebuild"))
from run_b1_per_system_table import cv_predict, prf
from run_c4_preregistered import fit_calibrators, apply_calibrators, signals, delong
import run_c4_preregistered as c4

CS = BASE / "results/signal_rebuild/class_scores"
NEW = CS / "new_student_class_scores.jsonl"
OUT = BASE / "results/signal_rebuild/deploy_preview.json"
CLASSES = c4.CLASSES
SYSTEMS = c4.SYSTEMS


def matrices():
    m = np.load(CS / "class_score_matrix.npz", allow_pickle=True)
    ids = [str(x) for x in m["ids"]]; pos = {i: k for k, i in enumerate(ids)}
    mc, ms = [str(x) for x in m["classes"]], [str(x) for x in m["systems"]]
    R = m["raw"][:, [ms.index(s) for s in SYSTEMS]][:, :, [mc.index(c) for c in CLASSES]].astype(np.float64)
    Rn = R.copy()
    for line in open(NEW):
        r = json.loads(line); k = pos.get(r["image_id"])
        if k is None: continue
        for c, v in r["max_conf"].items():
            if c in CLASSES: Rn[k, 0, CLASSES.index(c)] = v
    return ids, pos, R, Rn


def percentile(col):
    p = np.empty_like(col)
    for k in range(col.shape[1]): p[:, k] = rankdata(col[:, k]) / len(col)
    return p


def per_class_and_fusion(R, rows, truth):
    out = {}
    P = np.stack([percentile(R[:, s]) for s in range(len(SYSTEMS))], axis=1)
    for ci, c in enumerate(CLASSES):
        y = truth[c].values
        d = R[rows, 0, ci]; f = P[rows][:, :, ci].mean(axis=1)
        out[c] = {"detector_auroc": round(float(roc_auc_score(y, d)), 4),
                  "detector_f1": prf(cv_predict(d, y), y)["f1"],
                  "fusion_auroc": round(float(roc_auc_score(y, f)), 4),
                  "fusion_f1": prf(cv_predict(f, y), y)["f1"]}
    return out, P


def agreement(P, rows, truth, R):
    pairs = list(combinations(range(len(SYSTEMS)), 2))
    A = np.stack([1 - np.abs(P[:, x] - P[:, y]) for x, y in pairs], axis=1)
    D = np.zeros(R.shape, dtype=int)
    for s in range(len(SYSTEMS)):
        for ci, c in enumerate(CLASSES):
            fpr, tpr, th = roc_curve(truth[c].values, R[rows, s, ci])
            D[:, s, ci] = (R[:, s, ci] >= float(th[np.argmax(tpr - fpr)])).astype(int)
    fk = {}
    for ci, c in enumerate(CLASSES):
        d = D[:, :, ci]; n = d.shape[1]; k1 = d.sum(1); k0 = n - k1
        P_i = (k1 * (k1 - 1) + k0 * (k0 - 1)) / (n * (n - 1)); p1 = d.mean(); Pe = p1 ** 2 + (1 - p1) ** 2
        fk[c] = round(float((P_i.mean() - Pe) / (1 - Pe)), 4) if Pe < 1 else None
    dk = {f"{SYSTEMS[x]}-{SYSTEMS[y]}": round(float(cohen_kappa_score(D[:, x, 0], D[:, y, 0])), 4) for x, y in pairs}
    return {"mean_rank_agreement": round(float(A.mean()), 4), "fleiss_kappa_per_class": fk,
            "person_pair_kappa": dk}


def c4_signal(R, rows, truth, fused, u_saa, seed=20260919):
    from sklearn.model_selection import KFold
    tt = truth.reset_index(drop=True); nA = len(rows)
    P = np.zeros((nA, len(SYSTEMS), len(CLASSES)))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(np.arange(nA)):
        P[te] = apply_calibrators(fit_calibrators(R[rows[tr]], tt.iloc[tr]), R[rows[te]])
    sig = signals(P, fused, u_saa)
    err = ((fused >= 0.5).astype(int) != truth["has_scene"].values).astype(int)
    d, ci, p = delong(err, sig["s1"], sig["s5"])
    return {"per_class_disagreement_auroc": round(float(roc_auc_score(err, sig["s1"])), 4),
            "plain_confidence_auroc": round(float(roc_auc_score(err, sig["s5"])), 4),
            "difference": round(d, 4), "delong_ci95": [round(ci[0], 4), round(ci[1], 4)]}


def training_boxes():
    """How many boxes of each class each detector was trained on. The deployed detector's per-class
    quality tracks this count, which is what the rebuild changes."""
    names = {0: "person", 1: "child", 2: "horse", 3: "building", 4: "weapon", 5: "vehicle", 6: "tree",
             7: "clothing", 8: "text", 9: "animal"}
    old = {c: 0 for c in CLASSES}
    for split in ("train", "val"):
        for f in (BASE / f"data/yolo11_v2_augmented/labels/{split}").glob("*.txt"):
            for line in f.read_text().split("\n"):
                if line.strip():
                    c = names.get(int(line.split()[0]))
                    if c in old: old[c] += 1
    new = json.loads((BASE / "results/signal_rebuild/student_rebuild/dataset_stats.json").read_text())
    return {"deployed_detector": old, "rebuilt_detector": {c: new["boxes_kept_per_class"][c] for c in CLASSES}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    ids, pos, R, Rn = matrices()
    truth = pd.read_csv(CS / "human_class_truth_300.csv", keep_default_na=False).set_index("image_id")
    rows = np.array([pos[i] for i in truth.index])
    ag = pd.read_csv(BASE / "results/multi_agent/agent_comparison_scores.csv").drop_duplicates("image_id").set_index("image_id")
    fused = (0.6 * ag["existing_pipeline_agent"] + 0.2 * ag["scene_agent"] + 0.2 * ag["vlm_agent"]).loc[truth.index].values
    u_saa = ag[c4.AGENTS4].std(axis=1).loc[truth.index].values
    out = {"experiment": "What deploying the rebuilt detector would change (preview; nothing is deployed)",
           "note": "only the detector's per-class column is swapped; the scene-level pipeline, triage score and risk control are unchanged here",
           "evaluation_pictures": "the 300 hand-annotated pictures; none of them entered the rebuild (they were excluded from the teacher's labelling pool), while 136 of them were in the deployed detector's training set",
           "training_boxes_per_class": training_boxes()}
    for tag, M in (("deployed_detector", R), ("rebuilt_detector", Rn)):
        per, P = per_class_and_fusion(M, rows, truth)
        out[tag] = {"per_class_on_300": per,
                    "macro": {k: round(float(np.mean([v[k] for v in per.values()])), 4)
                              for k in ("detector_auroc", "detector_f1", "fusion_auroc", "fusion_f1")},
                    "agreement": agreement(P, rows, truth, M),
                    "c4_signal_on_300": c4_signal(M, rows, truth, fused, u_saa)}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    o, n = out["deployed_detector"], out["rebuilt_detector"]
    print("macro over the 8 classes (deployed -> rebuilt)")
    for k, v in o["macro"].items(): print(f"  {k:16s} {v:.3f} -> {n['macro'][k]:.3f}")
    print(f"\nmean rank agreement {o['agreement']['mean_rank_agreement']:.3f} -> {n['agreement']['mean_rank_agreement']:.3f}")
    print("Fleiss kappa per class:")
    for c in CLASSES:
        print(f"  {c:9s} {o['agreement']['fleiss_kappa_per_class'][c]:+.3f} -> {n['agreement']['fleiss_kappa_per_class'][c]:+.3f}")
    print("\nC4 signal on the 300:")
    for tag, r in (("deployed", o["c4_signal_on_300"]), ("rebuilt ", n["c4_signal_on_300"])):
        print(f"  {tag}: disagreement {r['per_class_disagreement_auroc']:.3f} vs confidence {r['plain_confidence_auroc']:.3f}"
              f"  difference {r['difference']:+.3f} CI {r['delong_ci95']}")
    tb = out["training_boxes_per_class"]
    print("\nper class, detector AUROC / F1 (deployed -> rebuilt), with the training boxes behind each")
    for c in CLASSES:
        print(f"  {c:9s} {o['per_class_on_300'][c]['detector_auroc']:.3f}/{o['per_class_on_300'][c]['detector_f1']:.2f}"
              f" -> {n['per_class_on_300'][c]['detector_auroc']:.3f}/{n['per_class_on_300'][c]['detector_f1']:.2f}"
              f"   boxes {tb['deployed_detector'][c]:6,d} -> {tb['rebuilt_detector'][c]:6,d}")


if __name__ == "__main__":
    raise SystemExit(main())
