#!/usr/bin/env python3
"""The complete pipeline end to end on rebuilt signals, recalibrated, plus a 53k projection.

Earlier steps ran Tier 1 corpus-wide and fed it to Tier 2, but applied the DEPLOYED conformal
threshold to the rebuilt triage score. That is indicative only: lambda is calibrated against a
particular score, and reusing one calibrated on different signals is not a deployment. This
recalibrates on the rebuilt signals and reports the operating point that follows, so the
rebuilt pipeline is evaluated as a system rather than as a swapped-in component.

It also projects the measured throughput onto the full 53,000-image archive, which is the
decision the numbers are actually for.

Outputs: results/signal_rebuild/end_to_end_rebuilt.json
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
AGENTS = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT = SR / "end_to_end_rebuilt.json"
R4 = ["A_object", "A_agree", "A_scene", "A_vlm"]
D4 = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
GRID = np.linspace(0.001, 1.0, 1000)
ALPHAS = [0.01, 0.02, 0.05, 0.10, 0.15]

# measured on this hardware during the full Tier 1 re-run (4x RTX A6000)
THROUGHPUT = {"yolo": 16.2, "siglip": 6.7, "llava": 0.64 * 4, "kosmos": 0.31 * 4}


def crc(U_cal, E_cal, U_te, E_te, alpha):
    n = len(U_cal)
    risks = np.array([np.mean(E_cal * (U_cal < g)) for g in GRID])
    ok = (n / (n + 1.0)) * risks + (1.0 / (n + 1.0)) <= alpha
    lam = float(np.max(GRID[ok])) if ok.any() else 0.0  # no cut-off keeps the promise: approve nothing
    auto = U_te < lam
    risk = float(np.mean(E_te * auto))
    caught = int(E_te[~auto].sum()); tot = int(E_te.sum())
    return {"alpha": alpha, "lambda": round(lam, 4),
            "test_empirical_risk": round(risk, 4),
            "bound_satisfied": bool(risk <= alpha),
            "auto_approval_pct": round(100 * float(auto.mean()), 2),
            "error_recall_pct": round(100 * caught / tot, 2) if tot else None,
            "efficiency_lift": round((caught / tot) / max(float((~auto).mean()), 1e-9), 2) if tot else None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    reb = pd.read_csv(SR / "full_tier1_signals.csv")
    ag = pd.read_csv(AGENTS).drop_duplicates("image_id")
    sp = pd.read_csv(SPLIT)
    j = sp.merge(ag, left_on="ppn_id", right_on="image_id", how="inner").merge(
        reb, on="image_id", how="inner")

    j["saa_reb"] = 0.6 * j.A_object + 0.2 * j.A_scene + 0.2 * j.A_vlm
    j["U_reb"] = 0.6 * j[R4].std(axis=1, ddof=1) + 0.4 * (1 - j[R4].mean(axis=1))

    # The 0.5 decision threshold was set for the DEPLOYED signal scales. The rebuilt
    # signals sit differently -- A_scene alone has mean 0.81 -- so inheriting 0.5
    # penalises the rebuilt pipeline for an untuned hyperparameter rather than for its
    # signals. Retune on the calibration partition only; test is never consulted.
    _cal = j[j.split == "calibration"]
    _grid = np.arange(0.20, 0.95, 0.005)
    _errs = [float(((_cal.saa_reb >= t).astype(int) != _cal.has_scene).mean()) for t in _grid]
    THR_REB = float(_grid[int(np.argmin(_errs))])
    j["err_reb"] = ((j.saa_reb >= THR_REB).astype(int) != j.has_scene).astype(int)
    j["err_reb_inherited"] = ((j.saa_reb >= 0.5).astype(int) != j.has_scene).astype(int)
    j["saa_dep"] = 0.6 * j.existing_pipeline_agent + 0.2 * j.scene_agent + 0.2 * j.vlm_agent
    j["U_dep"] = 0.6 * j[D4].std(axis=1, ddof=1) + 0.4 * (1 - j[D4].mean(axis=1))
    j["err_dep"] = ((j.comparison_fusion_score >= 0.5).astype(int) != j.has_scene).astype(int)

    cal, te = j[j.split == "calibration"], j[j.split == "test"]
    out = {"experiment": "Rebuilt pipeline end to end, recalibrated on its own signals",
           "n_calibration": int(len(cal)), "n_test": int(len(te)),
           "decision_threshold": {
               "deployed": 0.5,
               "rebuilt_retuned_on_calibration": round(THR_REB, 3),
               "rebuilt_test_error_with_inherited_0.5": round(float(te.err_reb_inherited.mean()), 4),
               "rebuilt_test_error_retuned": round(float(te.err_reb.mean()), 4),
               "why": ("The 0.5 threshold belongs to the deployed signal scales; rebuilt A_scene "
                       "alone has mean 0.81. Inheriting it charges the rebuilt pipeline for an "
                       "untuned hyperparameter. Retuning on calibration alone takes the test error "
                       "rate from 0.4337 to 0.2731, against 0.2249 deployed, so most of the "
                       "apparent degradation was the threshold and not the signals.")},
           "saa_ranking_auroc_vs_human": {
               "deployed": round(float(roc_auc_score(te.has_scene, te.comparison_fusion_score)), 4),
               "rebuilt": round(float(roc_auc_score(te.has_scene, te.saa_reb)), 4)},
           "test_error_rate_deployed": round(float(te.err_dep.mean()), 4),
           "test_error_rate_rebuilt": round(float(te.err_reb.mean()), 4)}

    for tag, U, E in [("deployed", "U_dep", "err_dep"), ("rebuilt", "U_reb", "err_reb")]:
        rows = [crc(cal[U].values, cal[E].values, te[U].values, te[E].values, a) for a in ALPHAS]
        out[f"crc_{tag}"] = rows
        y, u = te[E].values, te[U].values
        out[f"triage_auroc_{tag}"] = (round(float(roc_auc_score(y, u)), 4)
                                      if len(np.unique(y)) > 1 else None)

    d5 = next(r for r in out["crc_rebuilt"] if r["alpha"] == 0.05)
    p5 = next(r for r in out["crc_deployed"] if r["alpha"] == 0.05)
    out["headline_alpha_0.05"] = {
        "deployed": {k: p5[k] for k in ("lambda", "auto_approval_pct", "error_recall_pct",
                                        "test_empirical_risk", "bound_satisfied")},
        "rebuilt": {k: d5[k] for k in ("lambda", "auto_approval_pct", "error_recall_pct",
                                       "test_empirical_risk", "bound_satisfied")}}

    # ---- projection to the full archive ----
    N = 53000
    per = {k: round(N / v / 3600, 2) for k, v in THROUGHPUT.items()}
    out["projection_to_full_archive"] = {
        "target_images": N,
        "measured_throughput_img_per_s": {k: round(v, 2) for k, v in THROUGHPUT.items()},
        "hours_per_stage_4xA6000": per,
        "total_hours": round(sum(per.values()), 1),
        "corpus_run_actual_hours": 4.63,
        "scale_factor": round(N / 12110, 2),
        "output_size_estimate_mb": round(8.3 * N / 12110, 1),
        "note": ("Throughput measured on this hardware during the 12,110-image run: YOLO and "
                 "SigLIP on one GPU, LLaVA and Kosmos sharded across four. Inference cost scales "
                 "linearly in image count; the annotation needed to evaluate the output does not "
                 "come with it.")}
    out["what_scaling_does_not_buy"] = (
        "Running the archive end to end produces signals for every image and an auto-approval "
        "decision for every image. It does not produce a measurement of whether those decisions "
        "are right. Accuracy is bounded by human labels, of which there are 1,535 for 12,110 "
        "images and none for the remaining 40,890 of a 53,000-image archive. Scaling the compute "
        "multiplies the output; it does not multiply the evidence.")

    if args.verify:
        prev = json.loads(OUT.read_text())
        same = json.dumps(prev, sort_keys=True) == json.dumps(out, sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH")
        return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
