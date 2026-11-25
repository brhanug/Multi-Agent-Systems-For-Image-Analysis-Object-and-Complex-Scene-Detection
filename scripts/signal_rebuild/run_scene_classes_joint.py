#!/usr/bin/env python3
"""The scene classes with the learned combination (one joint mapping) as the system's combination.

E6 (PLAN_combination.md, commit e9b630f, planned before it was run) found that one joint multinomial
mapping on all 27 agent reports gives a clearly higher balanced accuracy than the fixed 0.6/0.2/0.2 that
Task S (class_tasks/PLAN.md, commit d9a7fa1) used. The thesis therefore adopts the joint mapping as its
scene-class combination. This script repeats every Task S measure for that choice, on the same 992
pictures, labels ("one rule"), 200 redraws and conformal procedures, so that the scene-class results of
the thesis describe the system it recommends. It is an analysis chosen after E6's result and is
reported as such. Also with class-balanced fitting (the extra analysis of Task S) and with Mondrian
per-class sets (E3's method).

    python run_scene_classes_joint.py [--verify]
"""
import argparse, json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

import run_best_config_repeated as rb
import run_scene_classes as rs
from run_more_experiments import redraws

OUT = rb.SR / "scene_classes_joint.json"
ANSWERS, K = rs.ANSWERS, rs.K


def joint_probs(X, rows_fit, y_fit, rows_all, balanced):
    m = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced" if balanced else None).fit(X[rows_fit], y_fit)
    P = np.zeros((len(rows_all), K)); P[:, m.classes_] = m.predict_proba(X[rows_all]); return P


def mondrian(P, y_cal, alpha=0.10):
    q = np.full(K, -np.inf)
    for k in range(K):
        s = np.sort(P[y_cal == k, k]); m = int(np.floor(alpha * (len(s) + 1)))
        if m >= 1: q[k] = s[m - 1]
    return q


def run(V, d, rows, y, balanced):
    Xj = np.column_stack([V[k] for k in ("object", "scene", "vlm")])
    names = ("joint mapping", "describer alone")
    keep = {n: {"macro_f1": [], "accuracy": [], "balanced_accuracy": [], "approved@0.05": [], "risk@0.05": [],
                "wrong_among_approved@0.05": [], "set_coverage@0.1": [], "set_size@0.1": [], "single_class_sets@0.1": [],
                "mondrian_coverage@0.1": [], "mondrian_size@0.1": []} for n in names}
    per_class = {a: {"precision": [], "recall": []} for a in ANSWERS}
    set_cov_class = {a: [] for a in ANSWERS}; mon_cov_class = {a: [] for a in ANSWERS}
    conf = np.zeros((K, K), int); appr = np.zeros((2, K)); given = np.zeros(K)
    for tr, cal, te in redraws(d.ppn.values):
        Ps = {"joint mapping": joint_probs(Xj, rows[tr], y[tr], rows, balanced),
              "describer alone": joint_probs(V["vlm"], rows[tr], y[tr], rows, balanced)}
        for n, P in Ps.items():
            k = keep[n]; pred = P.argmax(1); yt = y[te]; pt = pred[te]
            k["macro_f1"].append(float(f1_score(yt, pt, labels=range(K), average="macro", zero_division=0)))
            k["accuracy"].append(float((pt == yt).mean()))
            k["balanced_accuracy"].append(float(np.mean([(pt[yt == c] == c).mean() for c in range(K) if (yt == c).any()])))
            u = 1 - P.max(1); e = (pred != y).astype(int); lam = rs.lam_top1(u[cal], e[cal], 0.05); nc = cal.sum()
            feas = (nc / (nc + 1.0)) * np.mean(e[cal] * (u[cal] < lam)) + 1.0 / (nc + 1.0) <= 0.05
            auto = (u[te] < lam) if feas else np.zeros(te.sum(), bool)
            k["approved@0.05"].append(float(auto.mean())); k["risk@0.05"].append(float(np.mean(e[te] * auto)))
            k["wrong_among_approved@0.05"].append(float(e[te][auto].mean()) if auto.any() else 0.0)
            g = rs.set_threshold(P[cal], y[cal], 0.10); S = rs.make_sets(P[te], g); cov = S[np.arange(len(yt)), yt]
            k["set_coverage@0.1"].append(float(cov.mean())); k["set_size@0.1"].append(float(S.sum(1).mean()))
            k["single_class_sets@0.1"].append(float((S.sum(1) == 1).mean()))
            q = mondrian(P[cal], y[cal]); M = P[te] >= q[None, :]; M[np.arange(len(yt)), pt] = True
            mc = M[np.arange(len(yt)), yt]
            k["mondrian_coverage@0.1"].append(float(mc.mean())); k["mondrian_size@0.1"].append(float(M.sum(1).mean()))
            if n != "joint mapping":
                continue
            pr, rc, _f, sup = precision_recall_fscore_support(yt, pt, labels=range(K), zero_division=0)
            for c in range(K):
                if sup[c] > 0:   # as Task S: averaged over draws where the class occurs, precision 0 if never given
                    per_class[ANSWERS[c]]["recall"].append(float(rc[c])); per_class[ANSWERS[c]]["precision"].append(float(pr[c]))
                    set_cov_class[ANSWERS[c]].append(float(cov[yt == c].mean())); mon_cov_class[ANSWERS[c]].append(float(mc[yt == c].mean()))
                given[c] += (pt == c).sum(); appr[0, c] += (auto & (pt == c)).sum(); appr[1, c] += (auto & (pt == c) & (e[te] == 1)).sum()
            conf += confusion_matrix(yt, pt, labels=range(K))
    mean = lambda x: round(float(np.mean(x)), 4) if x else None
    rowsum = conf.sum(1, keepdims=True)
    return {"variants": {n: {m: rb.summarise(x) for m, x in k.items()} for n, k in keep.items()},
            "joint_minus_describer": {m: rb.paired(np.array(keep["joint mapping"][m]) - np.array(keep["describer alone"][m]))
                                      for m in ("macro_f1", "balanced_accuracy", "approved@0.05", "set_size@0.1", "mondrian_size@0.1")},
            "per_class": {a: {q: mean(v) for q, v in z.items()} for a, z in per_class.items()},
            "set_coverage_per_class@0.1": {a: mean(v) for a, v in set_cov_class.items()},
            "mondrian_coverage_per_class@0.1": {a: mean(v) for a, v in mon_cov_class.items()},
            "confusion_pooled": {"rows_true_cols_predicted": ANSWERS, "matrix": conf.tolist(),
                                 "row_shares": np.round(conf / np.maximum(rowsum, 1), 4).tolist()},
            "approved_by_answer_given@0.05": {
                ANSWERS[c]: {"answers_given": int(given[c]),
                             "share_of_all_approvals": round(float(appr[0, c] / max(1, appr[0].sum())), 4),
                             "wrong_among_approved": round(float(appr[1, c] / max(1, appr[0, c])), 4) if appr[0, c] else None}
                for c in range(K)}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _o, scene_abs = e2e.native_vectors()
    V = {"object": nat["object"], "scene": scene_abs, "vlm": np.column_stack([rs.llava_scene(idx), nat["vlm"]])}
    d, _ = rs.pool_labels(); d = d[d.ppn_id.isin(idx)].reset_index(drop=True)
    y = d["one rule"].map({x: i for i, x in enumerate(ANSWERS)}).values; rows = d.ppn_id.map(idx).values
    out = {"what": "Task S measures with the joint mapping as the scene-class combination (chosen after E6)",
           "after": "PLAN_combination.md (commit e9b630f)", "pictures": int(len(d)), "labels": "one rule",
           "results": {"planned fit": run(V, d, rows, y, False), "class-balanced fit": run(V, d, rows, y, True)}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for v, r in out["results"].items():
        print("==", v)
        for n, m in r["variants"].items(): print(f"  {n:16s}", {q: x["mean"] for q, x in m.items()})
        print("  joint-describer", {q: (x["mean"], x["p2.5"], x["p97.5"], x.get("share_positive")) for q, x in r["joint_minus_describer"].items()})
        print("  per class", r["per_class"]); print("  sets/class", r["set_coverage_per_class@0.1"]); print("  mondrian/class", r["mondrian_coverage_per_class@0.1"])
        print("  approvals", r["approved_by_answer_given@0.05"]); print("  confusion rows", r["confusion_pooled"]["row_shares"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
