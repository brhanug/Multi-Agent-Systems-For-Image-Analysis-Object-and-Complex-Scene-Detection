#!/usr/bin/env python3
"""D3: does Kosmos-2.5 add negative evidence the scene agent does not already have?

The pilot left Kosmos one possible role: evidence that a picture is a text page, read from a
repetition-discounted substance score of its as-run output (0.811 at separating text from no-text
pictures on the 53 Task D pictures). But the scene agent already carries SigLIP's evidence that a
picture is printed text, sheet music, an ornament or similar -- the absence router. If Kosmos adds
nothing on top of that, it can leave the pipeline entirely.

The substance score is computed from the existing corpus-wide Kosmos output (no model is run),
added to the scene agent as a seventh number, and the pipeline is compared with and without it
over 200 book-disjoint redraws of the 1,250 labelled pictures, paired, with the agents' mappings
refitted on every draw.

    python run_kosmos_negative_evidence.py [--verify]
"""
import argparse, glob, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
OUT = SR / "kosmos_negative_evidence.json"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
W = np.array([0.6, 0.2, 0.2])
N_DRAWS, SEED = 200, 20260926


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    from run_kosmos_generation_pilot import substance

    idx, nat, _old, scene_abs = e2e.native_vectors()
    sub = np.zeros((len(idx), 1)); seen = 0
    for f in sorted(glob.glob(str(SR / "kosmos_full/shard_*.jsonl"))):
        for line in open(f):
            r = json.loads(line); k = idx.get(r["image_id"])
            if k is not None:
                sub[k, 0] = np.log1p(substance(r.get("kosmos_output") or "")); seen += 1
    assert seen == len(idx), f"Kosmos output for {seen} of {len(idx)} pictures"
    scene_k = np.hstack([scene_abs, sub])

    sp = pd.read_csv(SPLIT); sp = sp[sp.ppn_id.isin(idx)].copy()
    ra, y, books = sp.ppn_id.map(idx).values, sp.has_scene.values, sp.ppn.values
    variants = {"corrected + absence": {**nat, "scene": scene_abs},
                "corrected + absence + Kosmos": {**nat, "scene": scene_k}}

    def run(tr, te, V):
        P = np.column_stack([e2e.to_probability(V[k], ra[tr], y[tr], ra) for k in ("object", "scene", "vlm")])
        s = P @ W
        return (roc_auc_score(y[te], P[te, 1]), roc_auc_score(y[te], s[te]),
                float(((s >= 0.5).astype(int) != y)[te].mean()))

    rng = np.random.default_rng(SEED); ub = np.unique(books)
    acc = {v: {"scene_agent_auroc": [], "pipeline_auroc": [], "test_error": []} for v in variants}
    for _ in range(N_DRAWS):
        perm = rng.permutation(ub)
        b_tr, b_te = set(perm[:int(len(ub) * 0.6)]), set(perm[int(len(ub) * 0.8):])
        tr = np.array([b in b_tr for b in books]); te = np.array([b in b_te for b in books])
        for v, V in variants.items():
            ag, pa, er = run(tr, te, V)
            acc[v]["scene_agent_auroc"].append(ag); acc[v]["pipeline_auroc"].append(pa); acc[v]["test_error"].append(er)

    f = lambda xs: {"mean": round(float(np.mean(xs)), 4), "p2.5": round(float(np.percentile(xs, 2.5)), 4),
                    "p97.5": round(float(np.percentile(xs, 97.5)), 4)}
    base, kos = variants
    diff = lambda k, sign=1: [sign * (b - c) for b, c in zip(acc[kos][k], acc[base][k])]
    paired = {}
    for k, sign in (("scene_agent_auroc", 1), ("pipeline_auroc", 1), ("test_error", -1)):
        d = diff(k, sign); s = f(d); s["share_above_zero"] = round(float((np.array(d) > 0).mean()), 4); paired[k] = s
    out = {"experiment": "D3 -- does Kosmos add negative evidence beyond SigLIP's absence prompts?",
           "kosmos_feature": "log(1 + repetition-discounted substance) of the as-run corpus-wide output",
           "draws": N_DRAWS, "split": "by book, 60/20/20, mappings refitted every draw",
           "per_variant": {v: {k: f(x) for k, x in d.items()} for v, d in acc.items()},
           "paired_gain_from_kosmos": paired}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out))
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for v, d in out["per_variant"].items():
        print(f"{v:30s} scene agent {d['scene_agent_auroc']['mean']:.3f}   pipeline {d['pipeline_auroc']['mean']:.3f}   "
              f"error {d['test_error']['mean']:.3f}")
    print("\npaired gain from adding Kosmos (mean [95% of draws], share above zero)")
    for k, s in paired.items():
        print(f"  {k:20s} {s['mean']:+.4f} [{s['p2.5']:+.4f}, {s['p97.5']:+.4f}]  {100*s['share_above_zero']:.1f}%")


if __name__ == "__main__":
    raise SystemExit(main())
