#!/usr/bin/env python3
"""
Scene Complexity Index (SCI) — formal definition and measurement.

Defines a multi-dimensional complexity score per image using data-driven
weights derived from Random Forest feature importance on the annotated cohort:

  C(image) = 0.48 * object_density_norm     (α: spatial detector saturation)
           + 0.50 * spatial_overlap_proxy    (β: inter-agent spatial std)
           + 0.00 * scene_classification_entropy  (γ: scene entropy, data-driven weight=0)
           + 0.02 * inter_agent_disagreement (δ: global disagreement signal)

Note: gamma=0.00 (scene entropy excluded) is derived from RF feature importance
  showing object_density (0.4822) and spatial_overlap (0.4989) account for ~98%
  of predictive power. scene_entropy importance=0.0001 is below noise threshold.
  This is reported explicitly in the thesis (Section SCI) to distinguish from
  early theoretical weights (0.35/0.25/0.25/0.15).

Then analyses:
  - Does fusion advantage (Δ = fusion - mono) correlate with SCI?
  - At which SCI bin does multi-agent win vs monolithic?
  - Spearman correlation: SCI vs Δ(fusion - mono)

Outputs:
  results/multi_agent/scene_complexity_index.csv
  results/multi_agent/scene_complexity_summary.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ALPHA, BETA, GAMMA, DELTA_W = 0.48, 0.50, 0.00, 0.02
N_BINS = 5
BIN_LABELS = ["very_low","low","medium","high","very_high"]

def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 3: return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d  = rx - ry
    return float(1.0 - 6*np.sum(d**2)/(n*(n**2-1)))

def normalize_id(raw: str) -> str:
    from pathlib import Path
    parts = Path(str(raw)).parts
    if len(parts) >= 2:
        if parts[-2] not in ["original", "restored", "images", "metadata", "results", "CycleGAN", "pix2pix"]:
            return f"{parts[-2]}/{Path(parts[-1]).stem}"
    return Path(str(raw)).stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--scores-csv", default="results/multi_agent/agent_comparison_scores.csv")
    parser.add_argument("--manifest",   default="final_dataset/metadata/manifest.csv")
    parser.add_argument("--output-dir", default="results/multi_agent")
    args = parser.parse_args()

    base    = Path(args.base_dir).resolve()
    scores  = pd.read_csv(base / args.scores_csv if not Path(args.scores_csv).is_absolute() else args.scores_csv)
    manifest= pd.read_csv(base / args.manifest   if not Path(args.manifest).is_absolute()   else args.manifest)
    out_dir = base / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge manifest for extra features
    manifest["img_key"] = manifest["image_id"].apply(normalize_id)
    scores["img_key"]   = scores["image_id"].apply(normalize_id)
    df = scores.merge(manifest[["img_key","vqa_total_objects","vqa_scene_confidence"]], on="img_key", how="left")

    # ── Component 1: Object density (normalised to [0,1]) ────────────────────
    # Use existing_pipeline_agent which already encodes object count / saturation
    obj_density = df["existing_pipeline_agent"].to_numpy(dtype=float)

    # ── Component 2: Spatial overlap proxy ──────────────────────────────────
    # High agreement_agent means boxes AGREE (low overlap/ambiguity)
    # High disagreement = high spatial complexity
    agent_cols = ["existing_pipeline_agent","agreement_agent","scene_agent",
                  "vlm_agent","restoration_agent","document_agent"]
    agent_matrix = df[[c for c in agent_cols if c in df.columns]].to_numpy(dtype=float)
    spatial_overlap = np.std(agent_matrix[:,:2], axis=1)          # std of object+agreement → spatial ambiguity
    spatial_overlap = (spatial_overlap - spatial_overlap.min()) / (spatial_overlap.max() - spatial_overlap.min() + 1e-9)

    # ── Component 3: Scene classification entropy ────────────────────────────
    # Low scene confidence = high semantic ambiguity = higher complexity
    scene_conf = df["vqa_scene_confidence"].fillna(0.5).to_numpy(dtype=float)
    scene_entropy = 1.0 - np.clip(scene_conf, 0, 1)   # inverse confidence = entropy proxy

    # ── Component 4: Inter-agent disagreement ────────────────────────────────
    inter_disagree = np.std(agent_matrix, axis=1)
    inter_disagree = (inter_disagree - inter_disagree.min()) / (inter_disagree.max() - inter_disagree.min() + 1e-9)

    # ── Composite SCI ────────────────────────────────────────────────────────
    sci = (ALPHA * obj_density
         + BETA  * spatial_overlap
         + GAMMA * scene_entropy
         + DELTA_W * inter_disagree)
    sci = np.clip(sci, 0, 1)

    df["scene_complexity_index"]  = sci
    df["sci_obj_density"]          = obj_density
    df["sci_spatial_overlap"]      = spatial_overlap
    df["sci_scene_entropy"]        = scene_entropy
    df["sci_inter_disagreement"]   = inter_disagree
    df["delta_fusion_vs_mono"]     = df["comparison_fusion_score"] - df["monolithic_pipeline_agent"]

    # Bin into 5 levels using QUANTILE-based bins so every level is populated.
    # (Equal-width bins with fixed [0,1] breakpoints leave high/very_high empty
    #  when the SCI distribution is left-skewed, producing NaN in high_delta.)
    # When duplicates='drop' reduces the edge count, trim labels to match.
    try:
        df["sci_bin"] = pd.qcut(sci, q=N_BINS, labels=BIN_LABELS, duplicates="drop")
    except ValueError:
        # Fewer unique quantile edges than labels → compute actual bin count and trim
        _, edges = pd.qcut(sci, q=N_BINS, retbins=True, duplicates="drop")
        n_actual_bins = len(edges) - 1
        df["sci_bin"] = pd.qcut(sci, q=N_BINS, labels=BIN_LABELS[:n_actual_bins], duplicates="drop")



    # Spearman: SCI vs Δ(fusion - mono)
    spear = spearman_r(sci, df["delta_fusion_vs_mono"].to_numpy(dtype=float))

    # Per-bin statistics
    bin_rows = []
    for label in BIN_LABELS:
        sub = df[df["sci_bin"] == label]
        if len(sub) == 0: continue
        bin_rows.append({
            "sci_bin":           label,
            "n_images":          len(sub),
            "sci_mean":          round(float(sub["scene_complexity_index"].mean()), 4),
            "monolithic_mean":   round(float(sub["monolithic_pipeline_agent"].mean()), 4),
            "fusion_mean":       round(float(sub["comparison_fusion_score"].mean()), 4),
            "delta_fus_mono":    round(float(sub["delta_fusion_vs_mono"].mean()), 4),
            "disagreement_mean": round(float(sub["inter_agent_disagreement"].mean() if "inter_agent_disagreement" in sub else 0), 4),
        })

    bin_df = pd.DataFrame(bin_rows)

    print("\n📊 Scene Complexity Index (SCI) — Bin Analysis:")
    print(f"   Spearman r (SCI vs Δ fusion-mono): {spear:.4f}")
    for _, r in bin_df.iterrows():
        print(f"  {r['sci_bin']:10s}: n={r['n_images']:5d} | SCI={r['sci_mean']:.3f} | "
              f"mono={r['monolithic_mean']:.3f} | fus={r['fusion_mean']:.3f} | Δ={r['delta_fus_mono']:+.3f}")

    # Key finding: does fusion win at high complexity?
    # Guard against empty bins (quantile bins guarantee coverage but just in case)
    high_bins = ["high", "very_high"]
    low_bins  = ["very_low", "low"]
    high_sub  = bin_df[bin_df["sci_bin"].isin(high_bins)]["delta_fus_mono"]
    low_sub   = bin_df[bin_df["sci_bin"].isin(low_bins)]["delta_fus_mono"]
    high_delta = float(high_sub.mean()) if len(high_sub) > 0 else float("nan")
    low_delta  = float(low_sub.mean())  if len(low_sub)  > 0 else float("nan")
    # fusion wins at high complexity only if both values are finite
    import math as _math
    fusion_wins_complex = (not _math.isnan(high_delta) and not _math.isnan(low_delta)
                           and high_delta > low_delta)

    # Save CSV
    sci_path = out_dir / "scene_complexity_index.csv"
    df.to_csv(sci_path, index=False)
    print(f"✅ Wrote {sci_path}")

    # Save summary
    summary = {
        "sci_formula": "0.48*object_density + 0.50*spatial_overlap + 0.00*scene_entropy + 0.02*inter_disagreement",
        "sci_formula_note": "Weights derived from RF feature importance (object_density=0.4822, spatial_overlap=0.4989, scene_entropy=0.0001, disagreement=0.0188). gamma=0.00 because scene_entropy importance below noise threshold.",
        "sci_components": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA, "delta": DELTA_W},
        "n_images": int(len(df)),
        "sci_stats": {
            "mean":   round(float(sci.mean()), 4),
            "std":    round(float(sci.std()), 4),
            "median": round(float(np.median(sci)), 4),
        },
        "spearman_sci_vs_delta_fusion_mono": round(spear, 4),
        "fusion_advantage_high_sci_bins":  None if _math.isnan(high_delta) else round(high_delta, 4),
        "fusion_advantage_low_sci_bins":   None if _math.isnan(low_delta)  else round(low_delta, 4),
        "fusion_wins_at_high_complexity":  fusion_wins_complex,
        "per_bin": bin_rows,
        "rq7_conclusion": (
            f"Multi-agent fusion provides "
            f"{'greater' if fusion_wins_complex else ('similar' if not _math.isnan(high_delta) else 'undetermined')} advantage "
            f"at high scene complexity (Δ={'N/A' if _math.isnan(high_delta) else f'{high_delta:+.4f}'}) "
            f"vs low complexity (Δ={'N/A' if _math.isnan(low_delta) else f'{low_delta:+.4f}'}). "
            f"Spearman r(SCI, Δ) = {spear:.4f} "
            f"({'positive correlation confirms' if spear > 0 else 'negative correlation refutes'} "
            f"the hypothesis that complexity mediates multi-agent advantage). "
            f"Bins computed using quantile-based binning (5 equal-frequency bins) "
            f"to ensure all complexity levels are populated."
        ),
    }
    summary_path = out_dir / "scene_complexity_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Wrote {summary_path}")

if __name__ == "__main__":
    main()
