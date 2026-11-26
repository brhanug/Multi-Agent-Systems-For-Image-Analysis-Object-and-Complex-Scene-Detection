#!/usr/bin/env python3
"""Evaluate the fine-tuned scene matcher and describer (PLAN_finetune.md, fixed before any run).

The out-of-fold reports of run_finetune_agents.py replace the zero-shot ones; everything downstream is the
thesis's pipeline over the same 200 book-disjoint redraws:
  scene class (992 pictures, "one rule"): the joint mapping on all 27 numbers (the system), the scene
      matcher alone and the describer alone; balanced accuracy, and approval at alpha = 0.05;
  scene presence (1,250 pictures): the corrected design (each agent mapped to P(scene), 0.6/0.2/0.2,
      confidence triage); ranking AUROC and approval at alpha = 0.05.
Compared, paired per redraw: fine-tuned against zero-shot scored with the same code (the plan's reference;
SigLIP zero-shot = the stored scores, LLaVA zero-shot = llava_zeroshot_fold*.jsonl). The thesis's stored
zero-shot reports are evaluated too, as a check that the same-code reference reproduces the thesis.

    python evaluate_finetune.py [--verify]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import run_best_config_repeated as rb
import run_scene_classes as rs
from run_more_experiments import redraws
from run_scene_classes_joint import joint_probs

FT = rb.SR / "finetune"
OUT = FT / "finetune_results.json"
SCENES = ["teaching", "family", "playing", "landscape", "drawing"]
OBJECTS = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]
K = 6


def read_folds(prefix, idx, n, kind):
    M = np.full((len(idx), n), np.nan); seen = 0
    for f in sorted(FT.glob(f"{prefix}_fold*.jsonl")):
        for line in open(f):
            r = json.loads(line); k = idx.get(r["image_id"])
            if k is None: continue
            if kind == "siglip":
                M[k] = [r["scene"][s] for s in SCENES] + [r["neg"]]
            else:
                M[k] = [r["p_yes"][c] if r["p_yes"][c] is not None else np.nan for c in SCENES + OBJECTS]
            seen += 1
    return M, seen


def scene_class(V, d, rows, y):
    keep = {n: {"balanced_accuracy": [], "approved@0.05": [], "risk@0.05": []} for n in ("joint mapping", "scene matcher alone", "describer alone")}
    Xj = np.column_stack([V[k] for k in ("object", "scene", "vlm")])
    for tr, cal, te in redraws(d.ppn.values):
        for n, X in (("joint mapping", Xj), ("scene matcher alone", V["scene"]), ("describer alone", V["vlm"])):
            P = joint_probs(X, rows[tr], y[tr], rows, False); pred = P.argmax(1); yt, pt = y[te], pred[te]
            keep[n]["balanced_accuracy"].append(float(np.mean([(pt[yt == c] == c).mean() for c in range(K) if (yt == c).any()])))
            u = 1 - P.max(1); e = (pred != y).astype(int); lam = rs.lam_top1(u[cal], e[cal], 0.05); nc = cal.sum()
            feas = (nc / (nc + 1.0)) * np.mean(e[cal] * (u[cal] < lam)) + 1.0 / (nc + 1.0) <= 0.05
            auto = (u[te] < lam) if feas else np.zeros(te.sum(), bool)
            keep[n]["approved@0.05"].append(float(auto.mean())); keep[n]["risk@0.05"].append(float(np.mean(e[te] * auto)))
    return keep


def presence(V, sp, rows):
    import run_corrected_tier1_end_to_end as e2e
    y = sp.has_scene.values; books = sp.ppn.values; uniq = np.unique(books)
    keep = {"scene_auroc": [], "approved@0.05": [], "risk@0.05": []}
    rng = np.random.default_rng(rb.SEED)
    for _ in range(rb.N_DRAWS):
        perm = rng.permutation(uniq); n_tr, n_cal = int(len(uniq) * 0.6), int(len(uniq) * 0.2)
        b_tr, b_cal = set(perm[:n_tr]), set(perm[n_tr:n_tr + n_cal])
        tr = np.array([b in b_tr for b in books]); cal = np.array([b in b_cal for b in books]); te = ~(tr | cal)
        P = np.column_stack([e2e.to_probability(V[k], rows[tr], y[tr], rows) for k in ("object", "scene", "vlm_objects")])
        saa = P @ rb.W; err = ((saa >= 0.5).astype(int) != y).astype(int); u = 1 - np.abs(saa - 0.5)
        lam = rb.lam_for(u[cal], err[cal], 0.05); auto = u[te] < lam
        keep["scene_auroc"].append(float(roc_auc_score(y[te], saa[te])))
        keep["approved@0.05"].append(float(auto.mean())); keep["risk@0.05"].append(float(np.mean(err[te] * auto)))
    return keep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _o, scene_abs = e2e.native_vectors()
    sig_ft, n_sig = read_folds("siglip", idx, 6, "siglip")
    lla_ft, n_ft = read_folds("llava", idx, 13, "llava")
    lla_zs, n_zs = read_folds("llava_zeroshot", idx, 13, "llava")
    thesis_vlm = np.column_stack([rs.llava_scene(idx), nat["vlm"]])
    variants = {
        "zero-shot (thesis's stored reports)": {"object": nat["object"], "scene": scene_abs, "vlm": thesis_vlm},
        "zero-shot (same code as fine-tuned)": {"object": nat["object"], "scene": scene_abs, "vlm": lla_zs},
        "fine-tuned": {"object": nat["object"], "scene": sig_ft, "vlm": lla_ft},
        "SigLIP fine-tuned only": {"object": nat["object"], "scene": sig_ft, "vlm": lla_zs},
        "LLaVA fine-tuned only": {"object": nat["object"], "scene": scene_abs, "vlm": lla_ft}}
    for V in variants.values():
        V["vlm_objects"] = V["vlm"][:, 5:]
    d, _ = rs.pool_labels(); d = d[d.ppn_id.isin(idx)].reset_index(drop=True)
    yc = d["one rule"].map({x: i for i, x in enumerate(rs.ANSWERS)}).values; rc = d.ppn_id.map(idx).values
    sp = pd.read_csv(rb.SPLIT); sp = sp[sp.ppn_id.isin(idx)].reset_index(drop=True); rp = sp.ppn_id.map(idx).values
    for name, V in variants.items():
        for k in ("scene", "vlm"):
            assert not np.isnan(V[k][rp]).any(), f"{name}: {k} missing for a pool picture"
    sc = {n: scene_class(V, d, rc, yc) for n, V in variants.items()}
    pr = {n: presence(V, sp, rp) for n, V in variants.items()}
    ref, ft = "zero-shot (same code as fine-tuned)", "fine-tuned"
    diff = lambda A, B: rb.paired(np.array(A) - np.array(B))
    out = {"plan": "self_annotation_package/more_experiments/PLAN_finetune.md (fixed before any run)",
           "cross_fitting": "5 book folds over the 1,250-picture pool, seed 20261125; every picture scored by a model that never saw its book",
           "pictures_scored": {"siglip fine-tuned": n_sig, "llava fine-tuned": n_ft, "llava zero-shot (same code)": n_zs},
           "scene_class": {n: {s: {m: rb.summarise(x) for m, x in r.items()} for s, r in v.items()} for n, v in sc.items()},
           "presence": {n: {m: rb.summarise(x) for m, x in v.items()} for n, v in pr.items()},
           "fine_tuned_minus_zero_shot": {
               "PRIMARY scene class, joint mapping, balanced accuracy": diff(sc[ft]["joint mapping"]["balanced_accuracy"], sc[ref]["joint mapping"]["balanced_accuracy"]),
               "scene class, scene matcher alone, balanced accuracy": diff(sc[ft]["scene matcher alone"]["balanced_accuracy"], sc[ref]["scene matcher alone"]["balanced_accuracy"]),
               "scene class, describer alone, balanced accuracy": diff(sc[ft]["describer alone"]["balanced_accuracy"], sc[ref]["describer alone"]["balanced_accuracy"]),
               "scene class, joint mapping, approved at 0.05": diff(sc[ft]["joint mapping"]["approved@0.05"], sc[ref]["joint mapping"]["approved@0.05"]),
               "presence, ranking AUROC": diff(pr[ft]["scene_auroc"], pr[ref]["scene_auroc"]),
               "presence, approved at 0.05": diff(pr[ft]["approved@0.05"], pr[ref]["approved@0.05"])}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for n in variants:
        j = out["scene_class"][n]
        print(f"{n:38s} class BA joint {j['joint mapping']['balanced_accuracy']['mean']:.3f}  scene {j['scene matcher alone']['balanced_accuracy']['mean']:.3f}  "
              f"describer {j['describer alone']['balanced_accuracy']['mean']:.3f}  approved {j['joint mapping']['approved@0.05']['mean']:.3f} | "
              f"presence AUROC {out['presence'][n]['scene_auroc']['mean']:.3f} approved {out['presence'][n]['approved@0.05']['mean']:.3f}")
    for k, v in out["fine_tuned_minus_zero_shot"].items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
