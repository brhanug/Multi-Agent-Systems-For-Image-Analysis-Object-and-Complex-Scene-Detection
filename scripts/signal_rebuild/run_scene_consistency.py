#!/usr/bin/env python3
"""Does one consistent rule for "scene" change the corrected design's results?

In the 1,250-picture pool the expert CVAT labels count 796 of 801 pictures as scenes, sheet music,
covers and borders included, while the 2026 relabelling rounds (449 pictures) label the same kinds of
picture "no scene"; 191 of the 196 "no scene" labels come from those rounds. A pre-sort flagged 324
expert "scene" pictures that may break the 2026 rule; with 40 pictures it kept as scenes they make the 364
of self_annotation_package/scene_consistency/candidates.csv. They are relabelled under the 2026 rule
(people, animals, a landscape or a room = scene) by

  claude   a vision-language model (Claude), claude_labels.csv -- model labels, reported as such
  user     a person on serve_scene_consistency.py, labels.csv: all 364, or the random 60 of
           verify_60.csv, which then measure agreement with the model labels

Reported:
  labels       the user's labels per pre-sort category, and the pre-sort's agreement with them
               (flagged categories against "no scene", hidden controls against "scene")
  variants     the corrected design (three agents with the "no scene" score, 0.6/0.2/0.2) over the
               same 200 book-disjoint redraws as best_config_repeated.json, under
                 as labelled          the labels used throughout
                 consistent           the user's "no scene" applied; "unsure" keeps its old label
                 consistent, unsure   as consistent, with "unsure" also counted as no scene
               per variant: ranking (AUROC); at line 0.5 balanced accuracy, approval and realised risk
               under conformal risk control at 0.05, the share of redraws over 0.05, and approval at
               0.025; at the fitted line balanced accuracy, no-scene recall, approval at 0.05 and the
               share of "no scene" calls approved, with how many of those are wrong (pooled)

    python run_scene_consistency.py --labels claude|user [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import run_best_config_repeated as rb
import run_scene_definition_check as sd
from run_decision_threshold_check import fitted_line

HERE = Path("/data/brhanu/thesis_project/self_annotation_package/scene_consistency")
OUT = {"claude": rb.SR / "scene_consistency_claude.json", "user": rb.SR / "scene_consistency.json"}
FLAGGED = list("MCBIDE")          # music, cover, border, initial, diagram, emblem
CATS = {"M": "sheet music", "C": "cover, blank or marbled paper", "B": "border or ornament",
        "I": "initial without figures", "D": "diagram, map or table", "E": "emblem, coat of arms or stamp",
        "O": "single object", "U": "unsure", "S": "scene (hidden control)"}


def read_labels(who):
    cand = pd.read_csv(HERE / "candidates.csv")
    lab = pd.read_csv(HERE / ("claude_labels.csv" if who == "claude" else "labels.csv")).drop_duplicates("ppn_id", keep="last")
    m = cand.merge(lab[["ppn_id", "label"]], on="ppn_id", how="left")
    missing = int(m.label.isna().sum())
    if missing:
        raise SystemExit(f"{missing} of {len(m)} pictures are not labelled yet")
    return m


def evaluate(sp, idx, V, e2e):
    rows = sp.ppn_id.map(idx).values; y = sp.has_scene.values; books = sp.ppn.values; uniq = np.unique(books)
    R = {k: [] for k in ("auroc", "balanced_accuracy_05", "approved_05", "approved_0025", "balanced_accuracy_fitted",
                         "no_scene_recall_fitted", "approved_fitted", "no_scene_calls_approved_fitted")}
    risk, ns_wrong, ns_appr = [], 0, 0
    rng = np.random.default_rng(rb.SEED)
    for _ in range(rb.N_DRAWS):
        perm = rng.permutation(uniq)
        n_tr, n_cal = int(len(uniq) * 0.6), int(len(uniq) * 0.2)
        b_tr, b_cal = set(perm[:n_tr]), set(perm[n_tr:n_tr + n_cal])
        tr = np.array([b in b_tr for b in books]); cal = np.array([b in b_cal for b in books]); te = ~(tr | cal)
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        P = np.column_stack([e2e.to_probability(V[k], rows[tr], y[tr], rows) for k in ("object", "scene", "vlm")])
        p = P @ rb.W
        R["auroc"].append(float(roc_auc_score(y[te], p[te])))
        c = (p >= 0.5).astype(int); err = (c != y).astype(int); u = 1 - np.abs(p - 0.5)
        R["balanced_accuracy_05"].append(float(balanced_accuracy_score(y[te], c[te])))
        a = sd.approve(u[cal], err[cal], u[te], 0.05)
        R["approved_05"].append(float(a.mean())); risk.append(float(np.mean(err[te] * a)))
        R["approved_0025"].append(float(sd.approve(u[cal], err[cal], u[te], 0.025).mean()))
        L = fitted_line(p[tr], y[tr]); c = (p >= L).astype(int); err = (c != y).astype(int); u = 1 - np.abs(p - L)
        ns = c[te] == 0
        R["balanced_accuracy_fitted"].append(float(balanced_accuracy_score(y[te], c[te])))
        R["no_scene_recall_fitted"].append(float((ns & (y[te] == 0)).sum() / max(1, (y[te] == 0).sum())))
        a = sd.approve(u[cal], err[cal], u[te], 0.05)
        R["approved_fitted"].append(float(a.mean()))
        R["no_scene_calls_approved_fitted"].append(float((a & ns).sum() / max(1, ns.sum())))
        ns_wrong += int((err[te] * (a & ns)).sum()); ns_appr += int((a & ns).sum())
    out = {"no_scene_pictures": int((y == 0).sum()), "draws": len(risk)}
    out.update({k: rb.summarise(v) for k, v in R.items()})
    out["realised_risk_05"] = round(float(np.mean(risk)), 4)
    out["share_of_redraws_over_0.05"] = round(float(np.mean(np.array(risk) > 0.05)), 4)
    out["wrong_among_approved_no_scene_calls_fitted_pooled"] = round(ns_wrong / max(1, ns_appr), 4)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--labels", choices=["claude", "user"], required=True); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    m = read_labels(a.labels)
    by_cat = {CATS[k]: {l: int(n) for l, n in g.label.value_counts().items()} for k, g in m.groupby("claude_presort")}
    flagged = m[m.claude_presort.isin(FLAGGED)]; ctrl = m[m.group == "control"]
    labels = {"pictures": len(m), "by_presort_category": by_cat,
              "flagged_labelled_no_scene": f"{int((flagged.label == 'none').sum())}/{len(flagged)}",
              "controls_labelled_scene": f"{int((ctrl.label == 'scene').sum())}/{len(ctrl)}",
              "labeller": "Claude (vision-language model)" if a.labels == "claude" else "user"}
    if a.labels == "claude" and (HERE / "labels.csv").exists():
        v = pd.read_csv(HERE / "verify_60.csv")[["ppn_id"]]
        u = pd.read_csv(HERE / "labels.csv").drop_duplicates("ppn_id", keep="last")[["ppn_id", "label"]]
        j = v.merge(u, on="ppn_id").merge(m[["ppn_id", "label"]], on="ppn_id", suffixes=("_user", "_claude"))
        j = j[j.label_user != "unsure"]
        labels["agreement_with_user_on_verify_60"] = {"pictures": len(j), "agree": int((j.label_user == j.label_claude).sum())}

    idx, nat, _o, scene_abs = e2e.native_vectors(); V = {**nat, "scene": scene_abs}
    sp0 = pd.read_csv(rb.SPLIT); sp0 = sp0[sp0.ppn_id.isin(idx)].copy()
    none = set(m.ppn_id[m.label == "none"]); unsure = set(m.ppn_id[m.label == "unsure"])
    variants = {}
    for name, flip in (("as labelled", set()), ("consistent", none), ("consistent, unsure as no scene", none | unsure)):
        sp = sp0.copy(); sp.loc[sp.ppn_id.isin(flip), "has_scene"] = 0
        variants[name] = evaluate(sp, idx, V, e2e)
    out = {"experiment": "Corrected design under one consistent rule for scene", "seed": rb.SEED,
           "labels": labels, "variants": variants}
    out_path = OUT[a.labels]
    if a.verify:
        same = json.loads(out_path.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(labels, indent=2))
    for v, r in variants.items():
        print(f"{v:32s} no-scene {r['no_scene_pictures']:4d}  AUROC {r['auroc']['mean']:.3f}  "
              f"line 0.5: bal.acc {r['balanced_accuracy_05']['mean']:.3f} approved {r['approved_05']['mean']:.3f} "
              f"risk {r['realised_risk_05']} over 0.05 in {r['share_of_redraws_over_0.05']}  "
              f"fitted: bal.acc {r['balanced_accuracy_fitted']['mean']:.3f} approved {r['approved_fitted']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
