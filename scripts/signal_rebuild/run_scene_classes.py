#!/usr/bin/env python3
"""Task S of self_annotation_package/class_tasks/PLAN.md (commit d9a7fa1): which scene class a picture shows.

Six answers: teaching, family, playing, landscape, drawing, no scene. Pictures: the 801 expert-reviewed
pictures of the labelled pool and the 191 pool pictures labelled "no scene" in the 2026 rounds. Labels
"one rule" (primary: expert pictures the author labelled "no scene" under the scene rule become "no
scene") and "as labelled".

Each agent's report is mapped to a probability over the six answers by a multinomial logistic
regression on the training books; the three are combined 0.6/0.2/0.2; the answer is the most probable
class, the uncertainty 1 - (largest probability). Conformal risk control approves single answers
(loss: approved and wrong) and builds prediction sets (loss: true class not in the set).
200 book-disjoint redraws, 60/20/20, seed 20260926.

    python run_scene_classes.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

import run_best_config_repeated as rb

BASE = Path("/data/brhanu/thesis_project")
OUT = rb.SR / "scene_classes.json"
C1 = BASE / "human_spatial_audit/user_annotations_800.csv"
LLAVA_SCENE = rb.SR / "class_scores/llava_scene_scores"
ANSWERS = ["teaching", "family", "playing", "landscape", "drawing", "none"]
SCENES = ANSWERS[:5]
ALPHAS = [0.05, 0.10]
GRID = np.linspace(0.001, 1.0, 1000)
K = len(ANSWERS)


def pool_labels():
    import run_final_check as fc
    sp = pd.read_csv(rb.SPLIT)
    c = pd.read_csv(C1).iloc[:801].copy()
    stem = c.raw_name.map(lambda s: Path(str(s)).stem); c["ppn_id"] = stem.str.replace("_", "/", n=1)
    c = c.drop_duplicates("ppn_id").set_index("ppn_id")
    _, person = fc.labels()
    pl = person.drop_duplicates("ppn_id", keep="last").set_index("ppn_id").label
    ex = sp[sp.label_source == "cvat_expert_review"].copy()
    ex["as labelled"] = ex.ppn_id.map(c.scene_labels).fillna("none").values
    ex["one rule"] = ex["as labelled"]
    ex.loc[ex.ppn_id.map(pl).values == "none", "one rule"] = "none"
    neg = sp[(sp.label_source != "cvat_expert_review") & (sp.has_scene == 0)].copy()
    neg["as labelled"] = "none"; neg["one rule"] = "none"
    left_out = int(((sp.label_source != "cvat_expert_review") & (sp.has_scene == 1)).sum())
    d = pd.concat([ex, neg]).reset_index(drop=True)
    return d, left_out


def llava_scene(idx):
    V = np.full((len(idx), len(SCENES)), np.nan)
    for f in sorted(LLAVA_SCENE.glob("*.jsonl")):
        for line in open(f):
            r = json.loads(line); k = idx.get(r["image_id"])
            if k is not None:
                V[k] = [r["p_yes"][s] for s in SCENES]
    return V


BALANCED = False   # --balanced: an extra analysis added after the planned run (class-balanced mapping)


def to_probs(X, rows_fit, y_fit, rows_all):
    """Multinomial mapping fitted on the training books; classes absent there get probability 0."""
    m = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced" if BALANCED else None).fit(X[rows_fit], y_fit)
    P = np.zeros((len(rows_all), K)); P[:, m.classes_] = m.predict_proba(X[rows_all])
    return P


def lam_top1(u, e, alpha):
    return rb.lam_for(u, e, alpha)


def set_threshold(P, y, alpha):
    """Smallest lambda whose sets {k: p_k >= 1 - lambda} (plus the top class) keep the conformal bound on
    the miss rate. The loss falls as lambda grows, so we scan upward."""
    n = len(y); top = P.argmax(1)
    p_true = P[np.arange(n), y]
    for g in GRID:
        miss = ~((p_true >= 1 - g) | (top == y))
        if (n / (n + 1.0)) * miss.mean() + 1.0 / (n + 1.0) <= alpha:
            return g
    return 1.0


def make_sets(P, g):
    S = P >= 1 - g; S[np.arange(len(P)), P.argmax(1)] = True
    return S


def run(d, label, V, idx):
    y_all = d[label].map({a: i for i, a in enumerate(ANSWERS)}).values
    rows = d.ppn_id.map(idx).values; books = d.ppn.values; uniq = np.unique(books)
    variants = ["combined", "equal weights", "object finder alone", "scene matcher alone", "describer alone",
                "scene matcher zero-shot", "majority answer"]
    keep = {v: {"macro_f1": [], "accuracy": [], "balanced_accuracy": [],
                **{f"approved@{a}": [] for a in ALPHAS}, **{f"risk@{a}": [] for a in ALPHAS},
                **{f"wrong_among_approved@{a}": [] for a in ALPHAS},
                **{f"set_coverage@{a}": [] for a in ALPHAS}, **{f"set_size@{a}": [] for a in ALPHAS},
                **{f"single_class_sets@{a}": [] for a in ALPHAS}} for v in variants}
    per_class = {v: {a: {"precision": [], "recall": [], "f1": [], "support": []} for a in ANSWERS} for v in ("combined", "describer alone")}
    set_cov_class = {a: [] for a in ANSWERS}
    conf = np.zeros((K, K), int)
    appr_by_pred = np.zeros((2, K))                 # combined, alpha 0.05, pooled: [approved, approved and wrong] per answer given
    given_by_pred = np.zeros(K)
    rng = np.random.default_rng(rb.SEED)
    used = 0
    for _ in range(rb.N_DRAWS):
        perm = rng.permutation(uniq); n_tr, n_cal = int(len(uniq) * 0.6), int(len(uniq) * 0.2)
        b_tr, b_cal = set(perm[:n_tr]), set(perm[n_tr:n_tr + n_cal])
        tr = np.array([b in b_tr for b in books]); cal = np.array([b in b_cal for b in books]); te = ~(tr | cal)
        used += 1
        Pk = {k: to_probs(V[k], rows[tr], y_all[tr], rows) for k in ("object", "scene", "vlm")}
        Ps = {"combined": 0.6 * Pk["object"] + 0.2 * Pk["scene"] + 0.2 * Pk["vlm"],
              "equal weights": (Pk["object"] + Pk["scene"] + Pk["vlm"]) / 3,
              "object finder alone": Pk["object"], "scene matcher alone": Pk["scene"], "describer alone": Pk["vlm"]}
        zs = V["scene"][rows]                      # five sigmoids and the "no scene" score, no training
        Ps["scene matcher zero-shot"] = zs / np.clip(zs.sum(1, keepdims=True), 1e-9, None)
        maj = np.bincount(y_all[tr], minlength=K).argmax(); M = np.zeros((len(rows), K)); M[:, maj] = 1.0
        Ps["majority answer"] = M
        for v, P in Ps.items():
            pred = P.argmax(1); yt = y_all[te]
            k = keep[v]
            k["macro_f1"].append(float(f1_score(yt, pred[te], labels=range(K), average="macro", zero_division=0)))
            k["accuracy"].append(float((pred[te] == yt).mean()))
            rec = [float((pred[te][yt == c] == c).mean()) for c in range(K) if (yt == c).any()]
            k["balanced_accuracy"].append(float(np.mean(rec)))
            u = 1 - P.max(1); e = (pred != y_all).astype(int)
            for a in ALPHAS:
                lam = lam_top1(u[cal], e[cal], a)
                n = cal.sum(); feas = (n / (n + 1.0)) * np.mean(e[cal] * (u[cal] < lam)) + 1.0 / (n + 1.0) <= a
                auto = (u[te] < lam) if feas else np.zeros(te.sum(), bool)
                k[f"approved@{a}"].append(float(auto.mean())); k[f"risk@{a}"].append(float(np.mean(e[te] * auto)))
                k[f"wrong_among_approved@{a}"].append(float(e[te][auto].mean()) if auto.any() else 0.0)
                if v == "combined" and a == 0.05:
                    pt = pred[te]
                    for c in range(K):
                        given_by_pred[c] += (pt == c).sum(); appr_by_pred[0, c] += (auto & (pt == c)).sum()
                        appr_by_pred[1, c] += (auto & (pt == c) & (e[te] == 1)).sum()
                g = set_threshold(P[cal], y_all[cal], a); S = make_sets(P[te], g)
                cov = S[np.arange(te.sum()), yt]
                k[f"set_coverage@{a}"].append(float(cov.mean())); k[f"set_size@{a}"].append(float(S.sum(1).mean()))
                k[f"single_class_sets@{a}"].append(float((S.sum(1) == 1).mean()))
                if v == "combined" and a == 0.10:
                    for c in range(K):
                        if (yt == c).any(): set_cov_class[ANSWERS[c]].append(float(cov[yt == c].mean()))
            if v in per_class:
                pr, rc, f1, sup = precision_recall_fscore_support(yt, pred[te], labels=range(K), zero_division=0)
                for c in range(K):
                    if sup[c] > 0:
                        for q, val in (("precision", pr[c]), ("recall", rc[c]), ("f1", f1[c])):
                            per_class[v][ANSWERS[c]][q].append(float(val))
                    per_class[v][ANSWERS[c]]["support"].append(int(sup[c]))
            if v == "combined":
                conf += confusion_matrix(yt, pred[te], labels=range(K))
        # paired differences are computed below from the stored lists
    out = {"draws": used, "counts": {a: int((d[label] == a).sum()) for a in ANSWERS}}
    out["variants"] = {v: {m: rb.summarise(x) for m, x in k.items() if x} for v, k in keep.items()}
    out["paired_combined_minus"] = {
        v: {m: rb.paired(np.array(keep["combined"][m]) - np.array(keep[v][m]))
            for m in ("macro_f1", "balanced_accuracy", "approved@0.05")}
        for v in variants if v != "combined"}
    out["per_class"] = {v: {a: {q: (round(float(np.mean(x)), 4) if x else None) for q, x in z.items()}
                            for a, z in pc.items()} for v, pc in per_class.items()}
    out["set_coverage_per_class@0.1"] = {a: (round(float(np.mean(x)), 4) if x else None) for a, x in set_cov_class.items()}
    out["confusion_pooled"] = {"rows_true_cols_predicted": ANSWERS, "matrix": conf.tolist()}
    out["combined_approved_by_answer_given@0.05"] = {
        ANSWERS[c]: {"answers_given": int(given_by_pred[c]),
                     "share_of_these_approved": round(float(appr_by_pred[0, c] / max(1, given_by_pred[c])), 4),
                     "share_of_all_approvals": round(float(appr_by_pred[0, c] / max(1, appr_by_pred[0].sum())), 4),
                     "wrong_among_approved": round(float(appr_by_pred[1, c] / max(1, appr_by_pred[0, c])), 4)}
        for c in range(K)}
    return out


def main():
    global BALANCED, OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--balanced", action="store_true", help="extra analysis, not in the plan: class-balanced mappings")
    a = ap.parse_args()
    if a.balanced:
        BALANCED = True; OUT = rb.SR / "scene_classes_balanced_extra.json"
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _o, scene_abs = e2e.native_vectors()
    ls = llava_scene(idx)
    V = {"object": nat["object"], "scene": scene_abs, "vlm": np.column_stack([ls, nat["vlm"]])}
    d, left_out = pool_labels()
    d = d[d.ppn_id.isin(idx)].reset_index(drop=True)
    assert not np.isnan(V["vlm"][d.ppn_id.map(idx).values]).any(), "a pool picture has no LLaVA scene scores"
    res = {lab: run(d, lab, V, idx) for lab in ("one rule", "as labelled")}
    out = {"experiment": "Task S: the scene class (five classes and no scene) on the labelled pool",
           "plan": "self_annotation_package/class_tasks/PLAN.md (commit d9a7fa1)"
                   + ("; EXTRA, not in the plan: class-balanced mappings" if BALANCED else ""),
           "pictures": int(len(d)), "books": int(d.ppn.nunique()),
           "pool_scene_pictures_without_class_left_out": left_out,
           "answers": ANSWERS, "seed": rb.SEED, "results": res}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for lab, r in res.items():
        print(f"\n== {lab}  {r['counts']}")
        for v, m in r["variants"].items():
            print(f"  {v:24s} macroF1 {m['macro_f1']['mean']:.3f} [{m['macro_f1']['p2.5']:.3f},{m['macro_f1']['p97.5']:.3f}]  acc {m['accuracy']['mean']:.3f}  "
                  f"balacc {m['balanced_accuracy']['mean']:.3f}  appr@.05 {m['approved@0.05']['mean']:.3f} (risk {m['risk@0.05']['mean']:.3f}, wrong {m['wrong_among_approved@0.05']['mean']:.3f})  "
                  f"set@.10 cov {m['set_coverage@0.1']['mean']:.3f} size {m['set_size@0.1']['mean']:.2f} single {m['single_class_sets@0.1']['mean']:.2f}")
        for v, z in r["paired_combined_minus"].items():
            print(f"  combined - {v:24s} " + "  ".join(f"{m} {s['mean']:+.3f} [{s['p2.5']:+.3f},{s['p97.5']:+.3f}]" for m, s in z.items()))
        print("  per class (combined):", {a: (z['recall'], z['precision']) for a, z in r["per_class"]["combined"].items()})
        print("  confusion", r["confusion_pooled"]["matrix"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
