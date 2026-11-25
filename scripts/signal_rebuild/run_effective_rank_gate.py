#!/usr/bin/env python3
"""AR10: effective rank of the Tier-1 signals, deployed against corrected, as an ingestion gate.

Disagreement between agents can only carry information if the agents span more than one direction.
The thesis summarises that by the effective rank of the agents' covariance -- the perplexity of its
eigenvalue spectrum, exp(-sum p_i log p_i) with p_i = lambda_i / sum lambda -- and the architecture
track proposes it as a gate: below 2, block the disagreement path for that collection.

It has only been computed on the deployed signals. This computes it on the corrected ones, over the
whole corpus, both as each agent's calibrated probability of a scene (the quantity fusion averages)
and per object class across the three systems (the quantity per-class agreement would read).

    python run_effective_rank_gate.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
OUT = SR / "effective_rank_gate.json"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
GATE = 2.0


def erank(X):
    X = np.asarray(X, float)
    lam = np.clip(np.linalg.eigvalsh(np.cov(X, rowvar=False)), 0, None)
    if lam.sum() == 0: return 1.0
    p = lam / lam.sum(); p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    import tier1_signals

    idx, nat, _old, scene_abs = e2e.native_vectors()
    sp = pd.read_csv(SPLIT); tr = sp[(sp.split == "train") & sp.ppn_id.isin(idx)]
    rows_tr, y_tr = tr.ppn_id.map(idx).values, tr.has_scene.values
    allr = np.arange(len(idx))

    out = {"experiment": "AR10 -- effective rank of the Tier-1 signals as an ingestion gate",
           "definition": "perplexity of the covariance spectrum, exp(-sum p log p); maximum = number of signals",
           "gate": GATE, "pictures": int(len(allr))}

    ag = pd.read_csv(AGENTS).drop_duplicates("image_id").set_index("image_id").reindex(list(idx))
    four = ag[["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]].values
    three = ag[["existing_pipeline_agent", "scene_agent", "vlm_agent"]].values
    out["deployed"] = {"four signals (with the derived agreement term)": round(erank(four), 4),
                       "three measured signals": round(erank(three), 4), "of_a_maximum_of": [4, 3]}

    res = {}
    for tag, V in (("corrected", nat), ("corrected + absence", {**nat, "scene": scene_abs})):
        P = np.column_stack([e2e.to_probability(V[k], rows_tr, y_tr, allr) for k in ("object", "scene", "vlm")])
        res[tag] = {"three calibrated P(scene)": round(erank(P), 4),
                    "pairwise correlation": {f"{x}/{y}": round(float(np.corrcoef(P[:, i], P[:, j])[0, 1]), 4)
                                             for (i, x), (j, y) in [((0, "object"), (1, "scene")),
                                                                    ((0, "object"), (2, "vlm")),
                                                                    ((1, "scene"), (2, "vlm"))]},
                    "passes_gate": bool(erank(P) >= GATE)}
    out["corrected"] = res

    sig = tier1_signals.load("corrected")
    m = np.load(SR / "class_scores/class_score_matrix.npz", allow_pickle=True)
    mc, ms = [str(x) for x in m["classes"]], [str(x) for x in m["systems"]]
    sig_raw = np.nan_to_num(m["raw"][:, ms.index("siglip")].astype(float))
    per = {}
    for ci, c in enumerate(tier1_signals.CLASSES):
        X = np.column_stack([sig["object"][:, ci], sig_raw[:, mc.index(c)], sig["vlm"][:, ci]])
        # rank-transform so a system's scale does not decide its share of the variance
        X = np.column_stack([pd.Series(X[:, k]).rank(pct=True).values for k in range(3)])
        per[c] = round(erank(X), 4)
    out["per_class_three_systems_rank_scaled"] = {"by_class": per,
                                                  "range": [min(per.values()), max(per.values())],
                                                  "classes_passing_gate": [c for c, v in per.items() if v >= GATE]}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out))
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("deployed", "corrected", "per_class_three_systems_rank_scaled")}, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
