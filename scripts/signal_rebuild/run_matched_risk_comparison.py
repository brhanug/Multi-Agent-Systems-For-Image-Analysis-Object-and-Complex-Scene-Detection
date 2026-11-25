#!/usr/bin/env python3
"""Old against corrected at the same REALISED risk, not only the same promised risk.

The headline compares the two pipelines at the same risk target: the corrected one approves 76.9%
at a target of 0.05 against 61.1% for the old one. But the old one realises only 0.017 at that
target (its uncertainty takes 8 values, so it cannot spend the budget), while the corrected one
realises 0.051. A reader may fairly ask which approves more at the same error actually delivered.

Over the same 200 book-disjoint redraws as best_config_repeated.json (same seed, same splits):

  promised  each pipeline calibrated by conformal risk control at a finer grid of targets; mean
            approval and mean realised risk per target, then approval read off at matched realised
            risk by interpolation along each pipeline's own curve
  best case on every draw, the most a pipeline can approve on its test books while its realised
            risk stays at or below r (the threshold chosen on the test books themselves): the
            ranking's own limit, free of calibration; paired per draw

Pipelines: old (disagreement triage, as deployed), old with confidence triage, corrected + absence
(confidence triage).

    python run_matched_risk_comparison.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

import run_best_config_repeated as rb

OUT = rb.SR / "matched_risk_comparison.json"
TARGETS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15]
MATCH = [0.01, 0.02, 0.03, 0.05, 0.1]


def best_case(u, err, r):
    """Largest share approvable (whole tie groups of u, lowest first) with mean(err * approved) <= r."""
    n = len(u); order = np.argsort(u, kind="stable"); us, es = u[order], err[order]
    cum = np.cumsum(es) / n
    ends = np.r_[np.where(np.diff(us) != 0)[0], n - 1]          # last index of each tie group
    ok = ends[cum[ends] <= r + 1e-12]
    return float((ok[-1] + 1) / n) if len(ok) else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _old, scene_abs = e2e.native_vectors()
    sp = pd.read_csv(rb.SPLIT); sp = sp[sp.ppn_id.isin(idx)].copy()
    rows_all = sp.ppn_id.map(idx).values; y = sp.has_scene.values; books = sp.ppn.values; uniq = np.unique(books)
    ag = pd.read_csv(rb.AGENTS).drop_duplicates("image_id").set_index("image_id").loc[sp.ppn_id]
    d_saa = (0.6 * ag.existing_pipeline_agent + 0.2 * ag.scene_agent + 0.2 * ag.vlm_agent).values
    d_err = ((d_saa >= 0.5).astype(int) != y).astype(int)
    V = {**nat, "scene": scene_abs}
    names = ["old (disagreement)", "old (confidence)", "corrected + absence (confidence)"]
    prom = {p: {t: {"approved": [], "risk": []} for t in TARGETS} for p in names}
    best = {p: {r: [] for r in MATCH} for p in names}

    rng = np.random.default_rng(rb.SEED)
    for _ in range(rb.N_DRAWS):
        perm = rng.permutation(uniq)
        n_tr, n_cal = int(len(uniq) * 0.6), int(len(uniq) * 0.2)
        b_tr, b_cal = set(perm[:n_tr]), set(perm[n_tr:n_tr + n_cal])
        tr = np.array([b in b_tr for b in books]); cal = np.array([b in b_cal for b in books]); te = ~(tr | cal)
        P = np.column_stack([e2e.to_probability(V[k], rows_all[tr], y[tr], rows_all) for k in ("object", "scene", "vlm")])
        saa = P @ rb.W; c_err = ((saa >= 0.5).astype(int) != y).astype(int)
        per = {"old (disagreement)": (ag[rb.A4].values.std(axis=1), d_err),
               "old (confidence)": (1 - np.abs(d_saa - 0.5), d_err),
               "corrected + absence (confidence)": (1 - np.abs(saa - 0.5), c_err)}
        for p, (u, err) in per.items():
            for t in TARGETS:
                lam = rb.lam_for(u[cal], err[cal], t); auto = u[te] < lam
                prom[p][t]["approved"].append(float(auto.mean())); prom[p][t]["risk"].append(float(np.mean(err[te] * auto)))
            for r in MATCH:
                best[p][r].append(best_case(u[te], err[te], r))

    curve = {p: [{"target": t, "approved": round(float(np.mean(prom[p][t]["approved"])), 4),
                  "realised_risk": round(float(np.mean(prom[p][t]["risk"])), 4)} for t in TARGETS] for p in names}

    def at_realised(p, r):
        pts = sorted((c["realised_risk"], c["approved"]) for c in curve[p])
        xs, ys = np.array([x for x, _ in pts]), np.array([y_ for _, y_ in pts])
        if r < xs[0] or r > xs[-1]: return None
        return round(float(np.interp(r, xs, np.maximum.accumulate(ys))), 4)

    out = {"experiment": "Old against corrected pipeline at matched realised risk",
           "pictures": int(len(sp)), "books": int(len(uniq)), "draws": rb.N_DRAWS, "seed": rb.SEED,
           "promised_curve": curve,
           "approval_at_matched_realised_risk": {str(r): {p: at_realised(p, r) for p in names} for r in MATCH},
           "best_case_approval": {str(r): {p: rb.summarise(best[p][r]) for p in names} for r in MATCH},
           "best_case_paired_corrected_minus_old_disagreement": {
               str(r): rb.paired(np.array(best[names[2]][r]) - np.array(best[names[0]][r])) for r in MATCH}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print("promised curve (target: approved / realised)")
    for p in names:
        print(f"  {p:34s} " + "  ".join(f"{c['target']}:{100*c['approved']:.1f}/{c['realised_risk']:.3f}" for c in curve[p]))
    print("\napproval at matched realised risk (interpolated on the promised curve)")
    for r, v in out["approval_at_matched_realised_risk"].items(): print(f"  {r}: {v}")
    print("\nbest case (threshold set on the test books), mean [2.5, 97.5]")
    for r, v in out["best_case_approval"].items():
        print(f"  {r}: " + " | ".join(f"{p} {s['mean']:.3f} [{s['p2.5']:.3f},{s['p97.5']:.3f}]" for p, s in v.items()))
    for r, s in out["best_case_paired_corrected_minus_old_disagreement"].items():
        print(f"  paired corrected - old @ {r}: {s['mean']:+.3f} [{s['p2.5']:+.3f},{s['p97.5']:+.3f}] above 0 in {100*s['share_of_draws_above_zero']:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
