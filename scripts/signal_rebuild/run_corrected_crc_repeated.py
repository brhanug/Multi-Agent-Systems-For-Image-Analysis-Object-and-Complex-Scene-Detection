#!/usr/bin/env python3
"""Does the corrected pipeline's risk guarantee hold, or did one split just fall badly?

The end-to-end run reported a bound violation at the two tightest targets on the committed split.
That is not yet evidence against the guarantee: conformal risk control bounds the risk in expectation
over draws of the calibration set, so any single split can exceed the target. The question is whether
the violations sit at the rate the bound allows.

So the split is drawn again, 200 times, over the 1,250 labelled pictures. Books are partitioned into
train, calibration and test in the proportions of the committed split (60/20/20), so no book is ever
on two sides -- pages of one book are correlated and would otherwise break the exchangeability the
bound assumes. Each draw fits the agents' mappings on its train books, calibrates lambda on its
calibration books, and measures the realised risk on its test books.

Reported per target: the mean realised risk, the share of draws whose risk exceeds the target, and
the mean auto-approval. The deployed pipeline is put through the identical procedure.

    python run_corrected_crc_repeated.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
OUT = SR / "corrected_crc_repeated.json"
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
A4 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
GRID = np.linspace(0.001, 1.0, 1000)
ALPHAS = [0.01, 0.02, 0.05, 0.10, 0.15]
N_DRAWS, SEED = 200, 20260920
SHARES = (0.60, 0.20)   # train, calibration; the rest is test


def lam_for(u_cal, e_cal, alpha):
    n = len(u_cal)
    risk = np.array([np.mean(e_cal * (u_cal < g)) for g in GRID])
    ok = (n / (n + 1.0)) * risk + 1.0 / (n + 1.0) <= alpha
    return float(GRID[ok].max()) if ok.any() else 0.0  # no cut-off keeps the promise: approve nothing


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e

    idx, nat, old_det, _ = e2e.native_vectors()
    sp = pd.read_csv(SPLIT)
    sp = sp[sp.ppn_id.isin(idx)].copy()
    sp["row"] = sp.ppn_id.map(idx)
    rows_all = sp.row.values
    y_all = sp.has_scene.values
    books = sp.ppn.values
    uniq = np.unique(books)

    ag = pd.read_csv(AGENTS).drop_duplicates("image_id").set_index("image_id").loc[sp.ppn_id]
    dep_saa = (0.6 * ag.existing_pipeline_agent + 0.2 * ag.scene_agent + 0.2 * ag.vlm_agent).values
    dep_unc = ag[A4].values.std(axis=1)
    dep_err = ((dep_saa >= 0.5).astype(int) != y_all).astype(int)

    rng = np.random.default_rng(SEED)
    acc = {tag: {al: {"risk": [], "approved": [], "violated": []} for al in ALPHAS}
           for tag in ("deployed", "corrected")}

    for _ in range(N_DRAWS):
        perm = rng.permutation(uniq)
        n_tr = int(len(uniq) * SHARES[0]); n_cal = int(len(uniq) * SHARES[1])
        b_tr, b_cal = set(perm[:n_tr]), set(perm[n_tr:n_tr + n_cal])
        tr = np.array([b in b_tr for b in books])
        cal = np.array([b in b_cal for b in books])
        te = ~(tr | cal)
        if not (tr.any() and cal.any() and te.any()) or len(np.unique(y_all[tr])) < 2:
            continue

        P = np.column_stack([e2e.to_probability(M, rows_all[tr], y_all[tr], rows_all)
                             for M in (nat["object"], nat["scene"], nat["vlm"])])
        saa = P @ np.array([0.6, 0.2, 0.2])
        unc = P.std(axis=1)
        err = ((saa >= 0.5).astype(int) != y_all).astype(int)

        for tag, u, e in (("deployed", dep_unc, dep_err), ("corrected", unc, err)):
            for al in ALPHAS:
                lam = lam_for(u[cal], e[cal], al)
                auto = u[te] < lam
                risk = float(np.mean(e[te] * auto))
                k = acc[tag][al]
                k["risk"].append(risk); k["approved"].append(float(auto.mean()))
                k["violated"].append(bool(risk > al))

    out = {"experiment": "Does the corrected pipeline's risk guarantee hold over repeated splits?",
           "why": ("Conformal risk control bounds the risk in expectation over draws of the calibration "
                   "set, so a single split exceeding the target is not evidence against it. This draws "
                   "the split 200 times and reports how often the target is exceeded."),
           "pictures": int(len(sp)), "books": int(len(uniq)), "draws": int(len(acc["deployed"][0.01]["risk"])),
           "split": "by book, 60/20/20 train/calibration/test, no book on two sides",
           "results": {tag: {str(al): {"mean_realised_risk": round(float(np.mean(v["risk"])), 4),
                                       "share_of_draws_over_target": round(float(np.mean(v["violated"])), 4),
                                       "mean_auto_approval": round(float(np.mean(v["approved"])), 4)}
                             for al, v in byal.items()} for tag, byal in acc.items()}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out))
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))

    print(f"{out['draws']} draws over {out['books']} books, {out['pictures']} labelled pictures\n")
    print(f"{'target':>7s}  {'pipeline':10s} {'mean risk':>10s} {'draws over target':>18s} {'auto-approved':>14s}")
    for al in ALPHAS:
        for tag in ("deployed", "corrected"):
            r = out["results"][tag][str(al)]
            print(f"{al:7.2f}  {tag:10s} {r['mean_realised_risk']:10.4f} "
                  f"{r['share_of_draws_over_target']:17.1%} {r['mean_auto_approval']:13.1%}")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
