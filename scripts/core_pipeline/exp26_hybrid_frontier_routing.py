#!/usr/bin/env python3
"""
exp26_hybrid_frontier_routing.py
---------------------------------
Illustrative extrapolation of the cost-efficiency of using local agents for
primary triage and commercial frontier models only for escalated,
high-uncertainty images. Sweeps the routing threshold based on SAA
uncertainty score.

IMPORTANT -- this is a calibrated illustration, not a per-image benchmark:
no frontier model is queried per routed image here. A single correction
probability is derived from the one real, measured comparison this thesis
has (results/multi_agent/coordinator_comparison.json: local Contradiction
F1=0.812 vs. frontier F1=0.965, a C4 n=100 benchmark using live Gemini/Claude
API calls in compare_coordinators.py) and applied uniformly via a seeded
Bernoulli draw. An earlier version of this script instead drew a per-image
"GPT-4o probability" from np.random.beta() parameters hand-picked to
reproduce pre-decided target statistics ("simulate frontier model
predictions to match thesis stats") -- i.e. a fabricated per-image result
with no image-level frontier evaluation behind it at all. That has been
replaced with the single-scalar, real-measurement-calibrated version below.

Outputs:
  results/multi_agent/exp26_hybrid_routing_results.json
  results/figures/exp26_hybrid_routing_curve.png
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")
GOLD_CSV = BASE / "human_spatial_audit/user_annotations_800.csv"
SCORES_CSV = BASE / "results/multi_agent/agent_comparison_scores.csv"
OUT_DIR = BASE / "results/multi_agent"
FIG_DIR = BASE / "results/figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def to_ppn_id(raw_id):
    name = str(raw_id).replace("images/", "").rsplit(".", 1)[0]
    parts = name.split("_", 1)
    if len(parts) > 1:
        return f"{parts[0]}/{parts[1]}"
    return name

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total_ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        total_ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(total_ece / max(len(y_true), 1))

def main():
    print("=" * 70)
    print("EXP 26: HYBRID FRONTIER-AGENT AUDITING (SELECTIVE API ROUTING)")
    print("=" * 70)

    # 1. Load data
    gold = pd.read_csv(GOLD_CSV)
    gold["cvat_id"] = gold.index
    gold = gold[gold["cvat_id"] <= 800].copy()
    gold["gold_has_scene"] = (gold["n_scene_labels"] > 0).astype(int)
    gold["ppn_id"] = gold["raw_name"].apply(to_ppn_id)

    agents = pd.read_csv(SCORES_CSV).drop_duplicates("image_id")
    joined = gold.merge(agents, left_on="ppn_id", right_on="image_id", how="inner")
    
    # 2. Deterministic sample of 100 images
    subset = joined.sample(n=100, random_state=100).copy().reset_index(drop=True)
    y_true = subset["gold_has_scene"].values
    
    # Compute local SAA fusion and uncertainty
    alpha, beta, gamma = 0.6, 0.2, 0.2
    subset["saa"] = alpha * subset["existing_pipeline_agent"] + beta * subset["scene_agent"] + gamma * subset["vlm_agent"]
    subset["saa_pred"] = (subset["saa"] >= 0.5).astype(int)
    subset["is_error"] = (subset["saa_pred"] != subset["gold_has_scene"]).astype(int)
    
    agent_cols = ["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]
    subset["saa_uncertainty"] = subset[agent_cols].std(axis=1).values
    
    # Normalize SAA uncertainty to [0, 1]
    u_max = subset["saa_uncertainty"].max()
    subset["saa_uncertainty_norm"] = subset["saa_uncertainty"] / u_max if u_max > 0 else subset["saa_uncertainty"]
    
    y_err = subset["is_error"].values
    u_saa = subset["saa_uncertainty_norm"].values

    # 3. Derive a single correction-probability scalar from the one real,
    #    measured local-vs-frontier comparison available (see module
    #    docstring). This is a coarse approximation -- frontier models don't
    #    correct every error at a fixed rate in reality -- but it is
    #    calibrated to a real number instead of fabricated per-image.
    coord_path = OUT_DIR / "coordinator_comparison.json"
    if coord_path.exists():
        with open(coord_path) as f:
            coord = json.load(f)
        local_f1 = next(v["Contradiction F1-Score"] for k, v in coord.items() if "Local" in k)
        frontier_f1 = next(v["Contradiction F1-Score"] for k, v in coord.items() if "Frontier" in k)
    else:
        print("  WARNING: coordinator_comparison.json not found; cannot calibrate "
              "correction probability to a real measurement. Skipping.")
        return
    # Fraction of the local-frontier F1 gap closed, applied as a flat
    # per-routed-error correction probability.
    correction_prob = float(np.clip((frontier_f1 - local_f1) / max(1.0 - local_f1, 1e-6), 0.0, 1.0))
    print(f"  Correction probability calibrated from real measurement: "
          f"local F1={local_f1}, frontier F1={frontier_f1} -> p_correct={correction_prob:.4f}")
    rng = np.random.default_rng(100)
    gpt4o_prob = np.where(rng.random(len(subset)) < correction_prob, 1.0, 0.0)

    # 4. Sweep selective routing percentiles
    percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    routing_data = []
    
    # Base API cost: $15.00 per 1000 images
    api_rate = 15.00 / 1000
    
    for p in percentiles:
        # Route top p% most uncertain images to GPT-4o
        threshold = np.percentile(u_saa, 100 - p)
        routed_mask = u_saa >= threshold
        
        # Cost
        n_routed = routed_mask.sum()
        cost = n_routed * api_rate
        
        # Error correction: the frontier model corrects the coordinator errors
        # Assume frontier model corrects errors with its predictive probability
        corrected_errors = y_err.copy()
        
        # For routed images, the corrected error depends on frontier VLM outcome
        # If VLM identifies error, error is corrected (is_error goes to 0)
        # We assume the VLM successfully detects and corrects the error with probability = gpt4o_prob
        for i in range(len(subset)):
            if routed_mask[i] and y_err[i] == 1:
                if gpt4o_prob[i] > 0.5:
                    corrected_errors[i] = 0
                    
        remaining_errors = corrected_errors.sum()
        total_errors_orig = y_err.sum()
        recall = (total_errors_orig - remaining_errors) / total_errors_orig if total_errors_orig > 0 else 1.0
        
        routing_data.append({
            "routed_percent": p,
            "routed_count": int(n_routed),
            "api_cost_per_1k": round(cost * 1000, 2),
            "error_recall": round(recall, 4),
            "remaining_errors": int(remaining_errors),
        })
        
        print(f"  Route top {p:3d}% | Routed: {n_routed:3d} | Cost/1k: ${cost*1000:5.2f} | Error Recall: {recall:.4f}")

    # Save results
    results = {
        "experiment": "Experiment 26: Hybrid Frontier-Agent Auditing (Selective Routing)",
        "provenance": "ILLUSTRATIVE EXTRAPOLATION, not a per-image frontier benchmark. "
                       "SAA uncertainty, gold labels, and routing costs are real; the "
                       "per-routed-error correction outcome is a single scalar probability "
                       "calibrated to the real, measured local-vs-frontier Contradiction F1 "
                       "gap (coordinator_comparison.json), applied via a seeded Bernoulli "
                       "draw -- not a real per-image frontier model query.",
        "correction_probability_calibrated_from_real_f1_gap": round(correction_prob, 4),
        "sample_n": len(subset),
        "total_errors": int(y_err.sum()),
        "routing_sweep": routing_data
    }
    
    out_json = OUT_DIR / "exp26_hybrid_routing_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {out_json}")

    # Plot trade-off curve
    plt.figure(figsize=(8, 5))
    x_cost = [r["api_cost_per_1k"] for r in routing_data]
    y_rec = [r["error_recall"] * 100 for r in routing_data]
    p_labels = [f"{r['routed_percent']}%" for r in routing_data]
    
    plt.plot(x_cost, y_rec, marker="o", color="#d95f02", linewidth=2, label="Hybrid Routing Frontier")
    for i, txt in enumerate(p_labels):
        if i % 2 == 0 or i == len(p_labels) - 1:
            plt.annotate(txt, (x_cost[i], y_rec[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
            
    plt.title("Exp 26: Selective Routing Trade-off (Illustrative, Calibrated to Measured F1 Gap)", fontsize=11, fontweight="bold")
    plt.xlabel("API Cost per 1k Images ($)", fontsize=11)
    plt.ylabel("Coordinator Error Correction Rate (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(-5, 105)
    plt.xlim(-max(x_cost) * 0.05, max(x_cost) * 1.05)  # was hardcoded (-1, 16), mismatched
    # against real costs up to ~$1500/1k -- a pre-existing axis bug that
    # silently flattened the whole curve into the left few pixels regardless
    # of the y-values plotted; fixed to auto-scale to the actual cost range.
    
    fig_path = FIG_DIR / "exp26_hybrid_routing_curve.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure saved to: {fig_path}")

if __name__ == "__main__":
    main()
