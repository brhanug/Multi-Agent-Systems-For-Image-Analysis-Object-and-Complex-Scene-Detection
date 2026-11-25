#!/usr/bin/env python3
"""The corrected design's decision line: fixed at 0.5, or fitted on the training books.

The corrected design calls a picture a scene when its combined probability p reaches 0.5. Because 84.5%
of the labelled pictures are scenes and averaging three probabilities pulls p towards the middle, p
rarely falls below 0.5, so the design calls nearly everything a scene: its error rate then equals that
of always answering "scene". This compares, over the same 200 book-disjoint redraws:

  fixed 0.5   the decision line as in the rest of the thesis
  fitted      the line that maximises balanced accuracy on the training books of each redraw
  deployed    the old design as deployed (its own score at 0.5, disagreement triage)

reporting on the test books: error rate, balanced accuracy, recall of no-scene pictures, the always-
"scene" error for reference, and under conformal risk control (uncertainty = closeness to the design's
own line, 1 - |p - line|) the share approved, the realised risk, and how many approved pictures are
no-scene calls. Run for the scene definition used throughout (1,250 pooled pictures) and for real
scene types only (801 expert-reviewed pictures, as in run_scene_definition_check.py).

    python run_decision_threshold_check.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import run_best_config_repeated as rb
import run_scene_definition_check as sd

OUT = rb.SR / "decision_threshold_check.json"
ALPHAS = [0.05, 0.10]
GRID_T = np.linspace(0.05, 0.95, 181)


def fitted_line(p, y):
    best, bt = -1, 0.5
    for t in GRID_T:
        c = (p >= t).astype(int)
        if 0 < c.sum() < len(c):
            b = balanced_accuracy_score(y, c)
            if b > best: best, bt = b, float(t)
    return bt


def evaluate(rows, y, books, V, e2e, old=None):
    uniq = np.unique(books)
    keys = ["error", "balanced_accuracy", "no_scene_recall", "always_majority_error", "auroc", "line"]
    acc = {v: {k: [] for k in keys + [f"approved@{a}" for a in ALPHAS] + [f"risk@{a}" for a in ALPHAS]
               + [f"approved_no_scene_calls@{a}" for a in ALPHAS]} for v in ("fixed 0.5", "fitted", "deployed")}
    pooled = {v: {a: [0, 0] for a in ALPHAS} for v in acc}
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
        variants = {"fixed 0.5": (p, 0.5), "fitted": (p, fitted_line(p[tr], y[tr]))}
        if old is not None:
            variants["deployed"] = (old, 0.5)
        majority = int(y[tr].mean() >= 0.5)
        for v, (s, t) in variants.items():
            c = (s >= t).astype(int); err = (c != y).astype(int)
            u = 1 - np.abs(s - t) if v != "deployed" else old_u
            k = acc[v]
            k["error"].append(float(err[te].mean())); k["balanced_accuracy"].append(float(balanced_accuracy_score(y[te], c[te])))
            k["no_scene_recall"].append(float(((c[te] == 0) & (y[te] == 0)).sum() / max(1, (y[te] == 0).sum())))
            k["always_majority_error"].append(float((y[te] != majority).mean())); k["auroc"].append(float(roc_auc_score(y[te], s[te])))
            k["line"].append(float(t))
            for a in ALPHAS:
                auto = sd.approve(u[cal], err[cal], u[te], a)
                k[f"approved@{a}"].append(float(auto.mean())); k[f"risk@{a}"].append(float(np.mean(err[te] * auto)))
                k[f"approved_no_scene_calls@{a}"].append(float(((c[te] == 0) & auto).sum() / max(1, auto.sum())))
                pooled[v][a][0] += int(((c[te] == 0) & auto).sum()); pooled[v][a][1] += int(auto.sum())
    res = {v: {k: (rb.summarise(x) if "risk" not in k else round(float(np.mean(x)), 4)) for k, x in d.items() if x}
           for v, d in acc.items() if d["error"]}
    for v in res:                                 # the per-redraw mean above is inflated by redraws with few approvals
        for a in ALPHAS:
            n0, n = pooled[v][a]; res[v][f"approved_no_scene_calls_pooled@{a}"] = round(n0 / max(1, n), 4)
    return res


def main():
    global old_u
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _o, scene_abs = e2e.native_vectors(); V = {**nat, "scene": scene_abs}

    sp = pd.read_csv(rb.SPLIT); sp = sp[sp.ppn_id.isin(idx)].copy()
    ag = pd.read_csv(rb.AGENTS).drop_duplicates("image_id").set_index("image_id").loc[sp.ppn_id]
    old = (0.6 * ag.existing_pipeline_agent + 0.2 * ag.scene_agent + 0.2 * ag.vlm_agent).values
    old_u = ag[rb.A4].values.std(axis=1)
    broad = evaluate(sp.ppn_id.map(idx).values, sp.has_scene.values, sp.ppn.values, V, e2e, old)

    c = pd.read_csv(sd.C1).iloc[:801].copy()
    stem = c.raw_name.map(lambda s: Path(str(s)).stem)
    c["ppn_id"] = stem.str.replace("_", "/", n=1); c["ppn"] = stem.str.split("_").str[0]
    c["narrow"] = c.scene_labels.fillna("").str.lower().map(lambda s: int(any(r in s for r in sd.REAL)))
    c = c[c.ppn_id.isin(idx)].drop_duplicates("ppn_id").reset_index(drop=True)
    ag2 = pd.read_csv(rb.AGENTS).drop_duplicates("image_id").set_index("image_id").reindex(c.ppn_id)
    old2 = (0.6 * ag2.existing_pipeline_agent + 0.2 * ag2.scene_agent + 0.2 * ag2.vlm_agent).fillna(0).values
    old_u = ag2[rb.A4].fillna(0).values.std(axis=1)
    narrow = evaluate(c.ppn_id.map(idx).values, c.narrow.values, c.ppn.values, V, e2e, old2)

    out = {"experiment": "Corrected design: decision line fixed at 0.5 against fitted on the training books",
           "draws": rb.N_DRAWS, "seed": rb.SEED, "line_fit": "maximum balanced accuracy on the training books",
           "broad_scene_definition_1250": broad, "real_scene_types_801": narrow}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for name, res in (("broad (1,250)", broad), ("real scene types (801)", narrow)):
        print(f"\n{name}")
        for v, r in res.items():
            print(f"  {v:10s} line {r['line']['mean']:.3f}  error {r['error']['mean']:.3f} (always-majority {r['always_majority_error']['mean']:.3f})  "
                  f"bal.acc {r['balanced_accuracy']['mean']:.3f}  no-scene recall {r['no_scene_recall']['mean']:.3f}  "
                  + "  ".join(f"approve@{al} {r[f'approved@{al}']['mean']:.3f} (risk {r[f'risk@{al}']}, no-scene calls {r[f'approved_no_scene_calls@{al}']['mean']:.3f})" for al in ALPHAS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
