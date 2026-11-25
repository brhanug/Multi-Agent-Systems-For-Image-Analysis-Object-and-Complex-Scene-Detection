"""
Three formulation questions an examiner can reasonably ask, answered on real data.

A. Does the uncertainty formulation matter? Compare the deployed 4-term spread
   against variants that drop the derived term or replace the operator, on each
   cohort and separately on the positive and negative subsets.

B. Are the fusion weights (0.6, 0.2, 0.2) defensible? They were fixed by hand and
   never tuned, so compare them against uniform, mild, and Shapley-inspired
   alternatives.

C. What does each agent contribute to error DETECTION, not merely to the spread?
   The existing macro ablation reports the change in mean sigma, which is a
   change in the uncertainty distribution rather than in predictive performance.

    python run_formulation_sensitivity.py [--verify]
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "core_pipeline"))
from saa_decision import SAA_THRESHOLD  # deliberate 0.55; see saa_decision.py
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = Path("/data/brhanu/thesis_project")
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
GOLD = BASE / "human_spatial_audit/user_annotations_800.csv"
OUT = BASE / "results/multi_agent/formulation_sensitivity.json"
O, A, S, V = "existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"
SEED, N_BOOT, BUDGET = 0, 2000, 0.20


from picture_ids import full_id as norm_id, read_expert_annotations   # book and page, never the page alone


def metrics(y, s):
    if len(np.unique(y)) < 2:
        return {"note": "single class", "n": int(len(y)), "n_err": int(y.sum())}
    k = max(1, int(round(BUDGET * len(y))))
    top = np.argsort(-s)[:k]                       # highest uncertainty gets reviewed
    return {"roc_auc": round(float(roc_auc_score(y, s)), 4),
            "pr_auc": round(float(average_precision_score(y, s)), 4),
            f"recall_at_{int(BUDGET*100)}pct": round(float(y[top].sum() / y.sum()), 4),
            "n": int(len(y)), "n_err": int(y.sum())}


def cohorts():
    ag = pd.read_csv(AGENTS).drop_duplicates("image_id")
    out = {}
    # corrected held-out test
    sp = pd.read_csv(SPLIT)
    j = sp.merge(ag, left_on="ppn_id", right_on="image_id", how="inner")
    j["fus"] = 0.6 * j[O] + 0.2 * j[S] + 0.2 * j[V]
    j["err"] = ((j.fus >= SAA_THRESHOLD).astype(int) != j.has_scene).astype(int)
    out["corrected_test"] = j[j.split == "test"].reset_index(drop=True)
    # C1 in-sample
    g = read_expert_annotations(GOLD); g["cvat_id"] = g.index
    g = g[g.cvat_id <= 800].copy()
    g["has_scene"] = (g["n_scene_labels"] > 0).astype(int)
    g["image_id"] = g["image_id"].apply(norm_id)
    a2 = ag.copy(); a2["image_id"] = a2["image_id"].apply(norm_id)
    c1 = g.merge(a2.drop_duplicates("image_id"), on="image_id", how="inner")
    c1["fus"] = 0.6 * c1[O] + 0.2 * c1[S] + 0.2 * c1[V]
    c1["err"] = ((c1.fus >= SAA_THRESHOLD).astype(int) != c1.has_scene).astype(int)
    out["C1_in_sample"] = c1
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    coh = cohorts()
    res = {"experiment": "Formulation sensitivity: uncertainty operator, fusion weights, "
                         "and per-agent contribution to error detection",
           "review_budget_for_recall": BUDGET}

    # ---------- A. uncertainty formulation ----------
    def variants(d):
        return {
            "std(Object, Agreement, Scene, VLM)  [deployed]": d[[O, A, S, V]].std(axis=1, ddof=1).values,
            "std(Object, Scene, VLM)  [derived term dropped]": d[[O, S, V]].std(axis=1, ddof=1).values,
            "1 - mean(Object, Scene, VLM)  [pure confidence]": (1 - d[[O, S, V]].mean(axis=1)).values,
            "1 - Object  [single model]": (1 - d[O]).values,
            "MAD(Object, Scene, VLM)": (d[[O, S, V]] - d[[O, S, V]].median(axis=1).values[:, None]).abs().median(axis=1).values,
            "range(Object, Scene, VLM)": (d[[O, S, V]].max(axis=1) - d[[O, S, V]].min(axis=1)).values,
        }
    res["A_uncertainty_formulation"] = {}
    for cname, d in coh.items():
        block = {}
        for vname, sig in variants(d).items():
            block[vname] = {"all": metrics(d.err.values, sig),
                            "positive_subset": metrics(d.err.values[d.has_scene == 1], sig[d.has_scene == 1]),
                            "negative_subset": metrics(d.err.values[d.has_scene == 0], sig[d.has_scene == 0])}
        res["A_uncertainty_formulation"][cname] = block

    # ---------- B. fusion weight sensitivity ----------
    W = {"0.6 / 0.2 / 0.2  [deployed]": (0.6, 0.2, 0.2),
         "1/3 / 1/3 / 1/3  [uniform]": (1/3, 1/3, 1/3),
         "0.5 / 0.25 / 0.25": (0.5, 0.25, 0.25),
         "0.4 / 0.3 / 0.3": (0.4, 0.3, 0.3),
         "0.28 / 0.19 / 0.33  [Shapley-inspired]": (0.28, 0.19, 0.33)}
    res["B_fusion_weight_sensitivity"] = {}
    for cname, d in coh.items():
        block = {}
        for wname, (wo, ws, wv) in W.items():
            f = wo * d[O] + ws * d[S] + wv * d[V]
            pred = (f >= 0.5).astype(int)
            acc = float((pred == d.has_scene).mean())
            tp = int(((pred == 1) & (d.has_scene == 1)).sum()); fp = int(((pred == 1) & (d.has_scene == 0)).sum())
            fn = int(((pred == 0) & (d.has_scene == 1)).sum())
            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            tn = int(((pred == 0) & (d.has_scene == 0)).sum())
            block[wname] = {"scene_accuracy": round(acc, 4),
                            "precision": round(prec, 4), "recall": round(rec, 4),
                            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
                            "auc_of_fusion_for_has_scene": round(float(roc_auc_score(d.has_scene, f)), 4),
                            "n_errors_at_0.5": int((pred != d.has_scene).sum()),
                            "confusion": {"TP": tp, "FN": fn, "TN": tn, "FP": fp},
                            "error_composition": f"{fn} missed scenes / {fp} false positives"}
        res["B_fusion_weight_sensitivity"][cname] = block

    # ---------- C. per-agent ablation, measured on error detection ----------
    res["C_agent_ablation_on_error_detection"] = {}
    for cname, d in coh.items():
        full = d[[O, A, S, V]].std(axis=1, ddof=1).values
        base = metrics(d.err.values, full)
        block = {"full_4_term": base | {"mean_sigma": round(float(full.mean()), 4)}}
        for drop in (O, A, S, V):
            rest = [c for c in (O, A, S, V) if c != drop]
            sig = d[rest].std(axis=1, ddof=1).values
            m = metrics(d.err.values, sig)
            block[f"remove {drop}"] = m | {
                "mean_sigma": round(float(sig.mean()), 4),
                "delta_roc_auc": round(m.get("roc_auc", float('nan')) - base.get("roc_auc", float('nan')), 4),
                "delta_pr_auc": round(m.get("pr_auc", float('nan')) - base.get("pr_auc", float('nan')), 4),
                "delta_mean_sigma": round(float(sig.mean()) - float(full.mean()), 4)}
        res["C_agent_ablation_on_error_detection"][cname] = block

    # ---------- a structural limit no weighting can pass ----------
    res["D_structural_blind_spot_of_the_fusion"] = {}
    for cname, d in coh.items():
        sub = d[(d[O] == 0) & (d.has_scene == 1)]
        res["D_structural_blind_spot_of_the_fusion"][cname] = {
            "n_scene_positive_with_A_object_zero": int(len(sub)),
            "of_total_scene_positive": int((d.has_scene == 1).sum()),
            "max_reachable_fusion": round(float(max((0.2*1.0, (1/3)*1.0))), 4),
            "explanation": "A_vlm = 1[A_object > 0], so when the object signal is zero two of the "
                           "three fusion terms are zero and only A_scene can vote. Its maximum "
                           "contribution is w_scene, so unless w_scene >= 0.5 these images can never "
                           "reach the 0.5 threshold under ANY weighting. This is a consequence of the "
                           "shared-source dependency, not of the weight choice."}

    if args.verify:
        prev = json.loads(OUT.read_text())
        same = json.dumps(prev, sort_keys=True) == json.dumps(res, sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
