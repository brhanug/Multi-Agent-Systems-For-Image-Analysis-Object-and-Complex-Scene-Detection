#!/usr/bin/env python3
"""What does "scene" mean for the system: a real scene, or any illustration?

In the expert-reviewed labels the scene / no-scene target counts a picture as a scene if any scene
type was given, and "drawing" was given far more than the four real scene types (teaching, family,
playing, landscape). Counting "drawing" as "no scene", as suggested for decision 1, therefore
changes the target of the main decision, not only the scene-type results. This measures the
corrected design under both definitions, on the expert-reviewed pictures of the labelled pool,
whose scene types are known:

  broad    a scene if any scene type was given (the definition used throughout)
  narrow   a scene only if teaching, family, playing or landscape was given

For each definition the corrected design (three agents, mappings refitted on that definition,
0.6/0.2/0.2, confidence triage) is evaluated over 200 book-disjoint redraws (60/20/20): scene
ranking (AUROC), error rate, and auto-approval and realised risk under conformal risk control at
0.05 and 0.10. The deployed design (its stored fusion score, routed by disagreement as deployed) is
scored on the same test books for comparison; its score is fixed, so only its cut-off is refitted.

    python run_scene_definition_check.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run_best_config_repeated as rb

BASE = Path("/data/brhanu/thesis_project")
OUT = rb.SR / "scene_definition_check.json"
C1 = BASE / "human_spatial_audit/user_annotations_800.csv"
REAL = ["teaching", "family", "playing", "landscape"]
ALPHAS = [0.05, 0.10]


def approve(u_cal, e_cal, u_te, alpha):
    """Conformal risk control; when no cut-off keeps the promise, nothing is approved (pictures whose
    uncertainty is exactly 0 would otherwise pass every cut-off)."""
    lam = rb.lam_for(u_cal, e_cal, alpha); n = len(u_cal)
    feasible = (n / (n + 1.0)) * np.mean(e_cal * (u_cal < lam)) + 1.0 / (n + 1.0) <= alpha
    return (u_te < lam) if feasible else np.zeros(len(u_te), bool)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _old, scene_abs = e2e.native_vectors()
    V = {**nat, "scene": scene_abs}

    c = pd.read_csv(C1).iloc[:801].copy()               # the expert-reviewed rows (the pool's first source)
    stem = c.raw_name.map(lambda s: Path(str(s)).stem)
    c["ppn_id"] = stem.str.replace("_", "/", n=1); c["ppn"] = stem.str.split("_").str[0]
    lab = c.scene_labels.fillna("").str.lower()
    c["broad"] = (c.n_scene_labels > 0).astype(int)
    c["narrow"] = lab.map(lambda s: int(any(r in s for r in REAL)))
    c = c[c.ppn_id.isin(idx)].drop_duplicates("ppn_id").reset_index(drop=True)
    rows = c.ppn_id.map(idx).values; books = c.ppn.values; uniq = np.unique(books)
    counts = {"pictures": int(len(c)), "books": int(len(uniq)),
              "broad_positives": int(c.broad.sum()), "narrow_positives": int(c.narrow.sum()),
              "drawing_only": int(((c.broad == 1) & (c.narrow == 0)).sum())}

    ag = pd.read_csv(rb.AGENTS).drop_duplicates("image_id").set_index("image_id").reindex(c.ppn_id)
    have_old = ag.existing_pipeline_agent.notna().values
    d_saa = (0.6 * ag.existing_pipeline_agent + 0.2 * ag.scene_agent + 0.2 * ag.vlm_agent).fillna(0).values
    d_u = ag[rb.A4].fillna(0).values.std(axis=1)
    counts["with_deployed_scores"] = int(have_old.sum())
    res = {}
    for name in ("broad", "narrow"):
        y = c[name].values
        acc = {"auroc": [], "error": [], **{f"approved@{al}": [] for al in ALPHAS}, **{f"risk@{al}": [] for al in ALPHAS},
               "deployed_auroc": [], "deployed_error": [], **{f"deployed_approved@{al}": [] for al in ALPHAS},
               **{f"deployed_risk@{al}": [] for al in ALPHAS}}
        rng = np.random.default_rng(rb.SEED)
        for _ in range(rb.N_DRAWS):
            perm = rng.permutation(uniq)
            n_tr, n_cal = int(len(uniq) * 0.6), int(len(uniq) * 0.2)
            b_tr, b_cal = set(perm[:n_tr]), set(perm[n_tr:n_tr + n_cal])
            tr = np.array([b in b_tr for b in books]); cal = np.array([b in b_cal for b in books]); te = ~(tr | cal)
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            P = np.column_stack([e2e.to_probability(V[k], rows[tr], y[tr], rows) for k in ("object", "scene", "vlm")])
            s = P @ rb.W
            err = ((s >= 0.5).astype(int) != y).astype(int); u = 1 - np.abs(s - 0.5)
            acc["auroc"].append(float(roc_auc_score(y[te], s[te]))); acc["error"].append(float(err[te].mean()))
            for al in ALPHAS:
                auto = approve(u[cal], err[cal], u[te], al)
                acc[f"approved@{al}"].append(float(auto.mean())); acc[f"risk@{al}"].append(float(np.mean(err[te] * auto)))
            t2, c2 = te & have_old, cal & have_old
            if len(set(y[t2])) == 2:
                de = ((d_saa >= 0.5).astype(int) != y).astype(int)
                acc["deployed_auroc"].append(float(roc_auc_score(y[t2], d_saa[t2]))); acc["deployed_error"].append(float(de[t2].mean()))
                for al in ALPHAS:
                    auto = approve(d_u[c2], de[c2], d_u[t2], al)
                    acc[f"deployed_approved@{al}"].append(float(auto.mean())); acc[f"deployed_risk@{al}"].append(float(np.mean(de[t2] * auto)))
        res[name] = {"draws_used": len(acc["auroc"]), "prevalence": round(float(y.mean()), 4),
                     **{k: (rb.summarise(v) if "risk" not in k else round(float(np.mean(v)), 4)) for k, v in acc.items() if v}}
    out = {"experiment": "The corrected design with 'scene' meaning any illustration (broad) or a real scene type (narrow)",
           "pictures_source": "expert-reviewed CVAT labels (first 801 rows), restricted to pictures the agents scored",
           "counts": counts, "draws": rb.N_DRAWS, "seed": rb.SEED, "results": res}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(counts))
    for k, r in res.items():
        print(f"{k:7s} prevalence {r['prevalence']:.3f}  AUROC {r['auroc']['mean']:.3f} [{r['auroc']['p2.5']:.3f},{r['auroc']['p97.5']:.3f}]  "
              f"error {r['error']['mean']:.3f}  " + "  ".join(f"approve@{al} {r[f'approved@{al}']['mean']:.3f} (risk {r[f'risk@{al}']})" for al in ALPHAS))
        if "deployed_auroc" in r:
            print(f"   deployed: AUROC {r['deployed_auroc']['mean']:.3f}  error {r['deployed_error']['mean']:.3f}  " + "  ".join(
                f"approve@{al} {r[f'deployed_approved@{al}']['mean']:.3f} (risk {r[f'deployed_risk@{al}']})" for al in ALPHAS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
