#!/usr/bin/env python3
"""The best configuration, over 200 redraws: are its gains a property of the pipeline or of one split?

The corrected pipeline's headline numbers -- scene AUROC 0.863 with the absence router, test error
0.153, 70.7% auto-approved at a risk target of 0.05 under confidence-scored triage -- come from the one
committed split of 249 test pictures. A single split cannot say how much of that is the split. This
redraws it 200 times over the 1,250 labelled pictures, books partitioned 60/20/20 so that no book is
ever on two sides, and on every draw refits the agents' mappings on its train books, calibrates the
risk threshold on its calibration books and measures on its test books.

Reported, each as the mean with the 2.5th and 97.5th percentiles over draws:
  scene AUROC and test error rate, for the deployed pipeline, the corrected one and corrected + absence
  auto-approval and realised risk under both triage scores, disagreement and confidence
and, per draw, the PAIRED differences that carry the claims: corrected + absence against deployed,
the absence router against its absence, and confidence triage against disagreement triage.

    python run_best_config_repeated.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
OUT = SR / "best_config_repeated.json"
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
A4 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
GRID = np.linspace(0.001, 1.0, 1000)
ALPHAS = [0.05, 0.10, 0.15]
N_DRAWS, SEED = 200, 20260926
W = np.array([0.6, 0.2, 0.2])


def lam_for(u, e, alpha):
    n = len(u)
    risk = (e[None, :] * (u[None, :] < GRID[:, None])).mean(axis=1)
    ok = (n / (n + 1.0)) * risk + 1.0 / (n + 1.0) <= alpha
    return float(GRID[ok].max()) if ok.any() else 0.0  # no cut-off keeps the promise: approve nothing


def summarise(xs):
    xs = np.asarray(xs, dtype=float)
    return {"mean": round(float(xs.mean()), 4),
            "p2.5": round(float(np.percentile(xs, 2.5)), 4),
            "p97.5": round(float(np.percentile(xs, 97.5)), 4)}


def paired(xs):
    s = summarise(xs)
    s["share_of_draws_above_zero"] = round(float((np.asarray(xs) > 0).mean()), 4)
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e

    idx, nat, _old, scene_abs = e2e.native_vectors()
    sp = pd.read_csv(SPLIT)
    sp = sp[sp.ppn_id.isin(idx)].copy()
    rows_all = sp.ppn_id.map(idx).values
    y = sp.has_scene.values
    books = sp.ppn.values
    uniq = np.unique(books)

    ag = pd.read_csv(AGENTS).drop_duplicates("image_id").set_index("image_id").loc[sp.ppn_id]
    d_saa = (0.6 * ag.existing_pipeline_agent + 0.2 * ag.scene_agent + 0.2 * ag.vlm_agent).values
    d_err = ((d_saa >= 0.5).astype(int) != y).astype(int)
    d_scores = {"disagreement": ag[A4].values.std(axis=1), "confidence": 1 - np.abs(d_saa - 0.5)}

    variants = {"corrected": nat, "corrected + absence": {**nat, "scene": scene_abs}}
    names = ["deployed"] + list(variants)
    acc = {v: {"scene_auroc": [], "test_error": [],
               **{f"{s}@{al}": {"approved": [], "risk": []}
                  for s in ("disagreement", "confidence") for al in ALPHAS}} for v in names}

    rng = np.random.default_rng(SEED)
    for _ in range(N_DRAWS):
        perm = rng.permutation(uniq)
        n_tr, n_cal = int(len(uniq) * 0.6), int(len(uniq) * 0.2)
        b_tr, b_cal = set(perm[:n_tr]), set(perm[n_tr:n_tr + n_cal])
        tr = np.array([b in b_tr for b in books]); cal = np.array([b in b_cal for b in books])
        te = ~(tr | cal)

        per = {"deployed": (d_saa, d_err, d_scores)}
        for v, V in variants.items():
            P = np.column_stack([e2e.to_probability(V[k], rows_all[tr], y[tr], rows_all)
                                 for k in ("object", "scene", "vlm")])
            saa = P @ W
            err = ((saa >= 0.5).astype(int) != y).astype(int)
            per[v] = (saa, err, {"disagreement": P.std(axis=1), "confidence": 1 - np.abs(saa - 0.5)})

        for v, (saa, err, scores) in per.items():
            acc[v]["scene_auroc"].append(float(roc_auc_score(y[te], saa[te])))
            acc[v]["test_error"].append(float(err[te].mean()))
            for s, u in scores.items():
                for al in ALPHAS:
                    lam = lam_for(u[cal], err[cal], al)
                    auto = u[te] < lam
                    k = acc[v][f"{s}@{al}"]
                    k["approved"].append(float(auto.mean()))
                    k["risk"].append(float(np.mean(err[te] * auto)))

    A = lambda v, k: np.array(acc[v][k])
    Ap = lambda v, k: np.array(acc[v][k]["approved"])
    out = {"experiment": "The best configuration over 200 book-disjoint redraws",
           "pictures": int(len(sp)), "books": int(len(uniq)), "draws": N_DRAWS,
           "split": "by book, 60/20/20 train/calibration/test, refitted on every draw",
           "per_pipeline": {v: {"scene_auroc": summarise(acc[v]["scene_auroc"]),
                                "test_error": summarise(acc[v]["test_error"]),
                                "triage": {k: {"auto_approved": summarise(acc[v][k]["approved"]),
                                               "mean_realised_risk": round(float(np.mean(acc[v][k]["risk"])), 4),
                                               "risk_standard_error": round(float(np.std(acc[v][k]["risk"], ddof=1)
                                                                                  / np.sqrt(N_DRAWS)), 4)}
                                           for k in acc[v] if "@" in k}}
                            for v in names},
           "paired_differences": {
               "scene AUROC, corrected + absence minus deployed":
                   paired(A("corrected + absence", "scene_auroc") - A("deployed", "scene_auroc")),
               "scene AUROC, absence router's own gain":
                   paired(A("corrected + absence", "scene_auroc") - A("corrected", "scene_auroc")),
               "test error, deployed minus corrected + absence":
                   paired(A("deployed", "test_error") - A("corrected + absence", "test_error")),
               **{f"auto-approval at {al}, confidence minus disagreement triage (corrected + absence)":
                      paired(Ap("corrected + absence", f"confidence@{al}") - Ap("corrected + absence", f"disagreement@{al}"))
                  for al in ALPHAS},
               **{f"auto-approval at {al}, corrected + absence (confidence) minus deployed (disagreement)":
                      paired(Ap("corrected + absence", f"confidence@{al}") - Ap("deployed", f"disagreement@{al}"))
                  for al in ALPHAS}}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out))
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))

    f = lambda s: f"{s['mean']:.3f} [{s['p2.5']:.3f}, {s['p97.5']:.3f}]"
    print(f"{N_DRAWS} redraws, {len(uniq)} books, {len(sp)} labelled pictures\n")
    print(f"{'':22s} {'scene AUROC':>24s} {'test error':>24s}")
    for v in names:
        r = out["per_pipeline"][v]
        print(f"  {v:20s} {f(r['scene_auroc']):>24s} {f(r['test_error']):>24s}")
    print("\nauto-approved (mean [95% of draws]), mean realised risk in brackets")
    for v in names:
        for s in ("disagreement", "confidence"):
            cells = "  ".join(f"a={al}: {100*out['per_pipeline'][v]['triage'][f'{s}@{al}']['auto_approved']['mean']:4.1f}% "
                              f"({out['per_pipeline'][v]['triage'][f'{s}@{al}']['mean_realised_risk']:.3f})"
                              for al in ALPHAS)
            print(f"  {v:20s} {s:12s} {cells}")
    print("\npaired differences (mean [95% of draws], share of draws above zero)")
    for k, s in out["paired_differences"].items():
        print(f"  {k}\n      {f(s)}   {100*s['share_of_draws_above_zero']:.1f}%")


if __name__ == "__main__":
    raise SystemExit(main())
