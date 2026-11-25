#!/usr/bin/env python3
"""E5 of self_annotation_package/more_experiments/PLAN.md: does restoration help the scene matcher?

Reads restoration_scores.npz (run_restoration_scores.py): SigLIP sigmoids for the five scene prompts and
seven "no scene" prompts on the original and on the restored scans. The scene matcher's report is the
five scene sigmoids and the largest "no scene" sigmoid, as in the corrected design.

  Task S (six answers, 992 pictures, 200 book-disjoint redraws): scene matcher alone, mapped
  (multinomial, planned) -- macro-F1 and balanced accuracy -- and zero-shot.
  First level: AUROC of scene presence on the same redraws (scene matcher alone, logistic mapping).
  Archive: mappings fitted on the 992, AUROC of scene presence on the 693 labelled archive pictures.
Paired differences, restored minus original.

    python run_restoration_ablation.py [--verify]
"""
import argparse, json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

import run_best_config_repeated as rb
import run_scene_classes as rs
import run_final_check as fc
import run_confirm_check as cc
from run_more_experiments import redraws

OUT = rb.SR / "restoration_ablation.json"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    z = np.load(rb.SR / "restoration_scores.npz", allow_pickle=True)
    ids = [str(x) for x in z["ids"]]; pos = {i: k for k, i in enumerate(ids)}
    feats = {c: np.column_stack([z[c][:, :5], z[c][:, 5:].max(1)]) for c in ("restored", "original")}
    d, _ = rs.pool_labels(); d = d[d.ppn_id.isin(pos)].reset_index(drop=True)
    ok = ~np.isnan(feats["original"][d.ppn_id.map(pos).values]).any(1) & ~np.isnan(feats["restored"][d.ppn_id.map(pos).values]).any(1)
    d = d[ok].reset_index(drop=True)
    rows = d.ppn_id.map(pos).values; y = d["one rule"].map({a_: i for i, a_ in enumerate(rs.ANSWERS)}).values
    pres = (y != rs.ANSWERS.index("none")).astype(int); K = rs.K
    t1, _ = fc.labels(); t1 = t1[t1.label != "unsure"]; t2, _ = cc.labelled(cc.CC / "items.csv", cc.CC / "labels.csv")
    arch_ids = list(t1.ppn_id) + list(t2.ppn_id); arch_y = np.r_[(t1.label == "scene").astype(int).values, t2.y.values]
    keep = np.array([i in pos and not np.isnan(feats["original"][pos[i]]).any() for i in arch_ids])
    xr = np.array([pos[i] for i, k in zip(arch_ids, keep) if k]); xy = arch_y[keep]
    acc = {c: {"macro_f1": [], "balanced_accuracy": [], "zero_shot_macro_f1": [], "zero_shot_balanced_accuracy": [], "presence_auroc": []}
           for c in feats}
    for tr, cal, te in redraws(d.ppn.values):
        for c, F in feats.items():
            P = rs.to_probs(F, rows[tr], y[tr], rows); pred = P.argmax(1); yt = y[te]
            zs = F[rows] / np.clip(F[rows].sum(1, keepdims=True), 1e-9, None); pz = zs.argmax(1)
            k = acc[c]
            k["macro_f1"].append(float(f1_score(yt, pred[te], labels=range(K), average="macro", zero_division=0)))
            k["balanced_accuracy"].append(float(np.mean([(pred[te][yt == j] == j).mean() for j in range(K) if (yt == j).any()])))
            k["zero_shot_macro_f1"].append(float(f1_score(yt, pz[te], labels=range(K), average="macro", zero_division=0)))
            k["zero_shot_balanced_accuracy"].append(float(np.mean([(pz[te][yt == j] == j).mean() for j in range(K) if (yt == j).any()])))
            m = LogisticRegression(max_iter=5000).fit(F[rows[tr]], pres[tr])
            k["presence_auroc"].append(float(roc_auc_score(pres[te], m.predict_proba(F[rows[te]])[:, 1])))
    out = {"plan": "self_annotation_package/more_experiments/PLAN.md (commit 01682a7)",
           "sanity_max_abs_diff_restored_vs_stored": round(float(z["sanity_max_abs_diff"][0]), 6),
           "pictures_task_s": int(len(d)), "archive_pictures": int(len(xr)),
           "per_condition": {c: {m: rb.summarise(v) for m, v in k.items()} for c, k in acc.items()},
           "restored_minus_original": {m: rb.paired(np.array(acc["restored"][m]) - np.array(acc["original"][m])) for m in acc["restored"]}}
    arch = {}
    for c, F in feats.items():
        m = LogisticRegression(max_iter=5000).fit(F[rows], pres)
        arch[c] = round(float(roc_auc_score(xy, m.predict_proba(F[xr])[:, 1])), 4)
    out["archive_presence_auroc"] = arch
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
