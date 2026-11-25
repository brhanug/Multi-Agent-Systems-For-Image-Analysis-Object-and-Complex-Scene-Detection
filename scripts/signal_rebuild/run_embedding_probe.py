#!/usr/bin/env python3
"""Can a picture's SigLIP embedding recognise "no scene" better than the corrected design?

The corrected design calls nearly every picture a scene, and with a fitted line most of its "no scene"
calls are wrong (two_sided_risk_control.json). Its agents reduce each picture to a few class scores.
This asks whether the full SigLIP image embedding (1,152 numbers, stored for all 12,110 pictures by the
cross-archive linker) carries the missing information: a logistic regression on the embedding,
class-balanced, fitted on the training books of each of the same 200 book-disjoint redraws as
best_config_repeated.json, against the corrected design on the same test books. No new labels and no
model inference are needed.

Reported per variant: ranking (AUROC), balanced accuracy and no-scene recall at the variant's own line
(0.5 for the balanced probe; the fitted line for the corrected design), and under conformal risk control
at alpha = 0.05 (uncertainty 1 - |p - line|) the share approved, the realised risk, the share of
"no scene" calls approved, and how many approved "no scene" calls are wrong (pooled over redraws).

    python run_embedding_probe.py [--verify]
"""
import argparse, json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_best_config_repeated as rb
import run_scene_definition_check as sd
from run_decision_threshold_check import fitted_line

OUT = rb.SR / "embedding_probe.json"
EMB = rb.SR / "siglip_embeddings.npz"
ALPHA = 0.05


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _o, scene_abs = e2e.native_vectors(); V = {**nat, "scene": scene_abs}
    emb = np.load(EMB, allow_pickle=True); epos = {str(i): k for k, i in enumerate(emb["ids"])}; X_all = emb["vecs"]
    sp = pd.read_csv(rb.SPLIT); sp = sp[sp.ppn_id.isin(idx) & sp.ppn_id.isin(epos)].copy()
    rows = sp.ppn_id.map(idx).values; X = X_all[sp.ppn_id.map(epos).values]
    y = sp.has_scene.values; books = sp.ppn.values; uniq = np.unique(books)
    keys = ["auroc", "balanced_accuracy", "no_scene_recall", "approved", "risk", "no_scene_calls_approved"]
    acc = {v: {k: [] for k in keys} for v in ("corrected design, fitted line", "embedding probe", "corrected agents + embedding probe")}
    pooled = {v: [0, 0] for v in acc}                   # wrong / approved "no scene" calls
    rng = np.random.default_rng(rb.SEED)
    for _ in range(rb.N_DRAWS):
        perm = rng.permutation(uniq)
        n_tr, n_cal = int(len(uniq) * 0.6), int(len(uniq) * 0.2)
        b_tr, b_cal = set(perm[:n_tr]), set(perm[n_tr:n_tr + n_cal])
        tr = np.array([b in b_tr for b in books]); cal = np.array([b in b_cal for b in books]); te = ~(tr | cal)
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        P = np.column_stack([e2e.to_probability(V[k], rows[tr], y[tr], rows) for k in ("object", "scene", "vlm")])
        p_cd = P @ rb.W
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced")).fit(X[tr], y[tr])
        p_pr = probe.predict_proba(X)[:, 1]
        both = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced")).fit(
            np.column_stack([X[tr], P[tr]]), y[tr])
        p_bo = both.predict_proba(np.column_stack([X, P]))[:, 1]
        for v, p, line in (("corrected design, fitted line", p_cd, fitted_line(p_cd[tr], y[tr])),
                           ("embedding probe", p_pr, 0.5), ("corrected agents + embedding probe", p_bo, 0.5)):
            c = (p >= line).astype(int); err = (c != y).astype(int); u = 1 - np.abs(p - line)
            auto = sd.approve(u[cal], err[cal], u[te], ALPHA)
            ns = c[te] == 0; d = acc[v]
            d["auroc"].append(float(roc_auc_score(y[te], p[te])))
            d["balanced_accuracy"].append(float(balanced_accuracy_score(y[te], c[te])))
            d["no_scene_recall"].append(float((ns & (y[te] == 0)).sum() / max(1, (y[te] == 0).sum())))
            d["approved"].append(float(auto.mean())); d["risk"].append(float(np.mean(err[te] * auto)))
            d["no_scene_calls_approved"].append(float((auto & ns).sum() / max(1, ns.sum())))
            pooled[v][0] += int((err[te] * (auto & ns)).sum()); pooled[v][1] += int((auto & ns).sum())
    out = {"experiment": "SigLIP image embedding against the corrected design for scene / no scene",
           "pictures": int(len(y)), "draws": rb.N_DRAWS, "seed": rb.SEED, "alpha": ALPHA,
           "results": {v: {k: (round(float(np.mean(x)), 4) if k == "risk" else rb.summarise(x)) for k, x in d.items()}
                       for v, d in acc.items()},
           "wrong_among_approved_no_scene_pooled": {v: round(n0 / max(1, n), 4) for v, (n0, n) in pooled.items()},
           "approved_no_scene_calls_pooled_count": {v: n for v, (_, n) in pooled.items()}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for v, r in out["results"].items():
        print(f"{v:36s} AUROC {r['auroc']['mean']:.3f}  bal.acc {r['balanced_accuracy']['mean']:.3f}  no-scene recall {r['no_scene_recall']['mean']:.3f}  "
              f"approved {r['approved']['mean']:.3f} (risk {r['risk']})  no-scene calls approved {r['no_scene_calls_approved']['mean']:.3f}  "
              f"wrong among them {out['wrong_among_approved_no_scene_pooled'][v]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
