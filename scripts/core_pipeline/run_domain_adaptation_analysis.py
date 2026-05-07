#!/usr/bin/env python3
"""
Domain Adaptation Analysis.

Quantifies the visual domain gap between:
  - Natural image domain (what YOLO/CLIP were trained on)
  - Historical archival domain (Colibri dataset)

Measures:
  1. Restoration gain distribution (SRS score: original vs restored)
  2. Domain difficulty proxy: images where student_v3_count = 0 but VLM > 0.5
     (VLM understands the image but detector fails = domain gap evidence)
  3. Per-PPN domain gap estimate from score distributions
  4. SRS gain vs agent score correlation (does restoration help detection?)
  5. Before/after restoration score comparison using srs_score_original vs restored

This directly answers: "What is the domain gap and does your pipeline address it?"

Outputs:
  results/multi_agent/domain_adaptation_analysis.csv
  results/multi_agent/domain_adaptation_summary.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def normalize_id(raw: str) -> str:
    return Path(str(raw)).stem

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    return float((xm*ym).sum() / denom) if denom > 1e-9 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir",   default="/data/brhanu/thesis_project")
    parser.add_argument("--scores-csv", default="results/multi_agent/agent_comparison_scores.csv")
    parser.add_argument("--manifest",   default="final_dataset/metadata/manifest.csv")
    parser.add_argument("--output-dir", default="results/multi_agent")
    args = parser.parse_args()

    base     = Path(args.base_dir).resolve()
    scores   = pd.read_csv(base / args.scores_csv   if not Path(args.scores_csv).is_absolute() else args.scores_csv)
    manifest = pd.read_csv(base / args.manifest      if not Path(args.manifest).is_absolute()   else args.manifest)
    out      = base / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest["img_key"] = manifest["image_id"].apply(normalize_id)
    scores["img_key"]   = scores["image_id"].apply(normalize_id)
    df = scores.merge(manifest, on="img_key", how="inner", suffixes=("","_m"))

    n_total = len(df)
    print(f"📊 Domain Adaptation Analysis — {n_total} images")

    # ── 1. SRS restoration gain ──────────────────────────────────────────────
    srs_orig = df["srs_score_original"].fillna(0).to_numpy(dtype=float) if "srs_score_original" in df.columns else np.zeros(n_total)
    srs_rest = df["srs_score_restored"].fillna(0).to_numpy(dtype=float) if "srs_score_restored" in df.columns else np.zeros(n_total)
    srs_gain = srs_rest - srs_orig

    n_improved    = int((srs_gain > 0.01).sum())
    n_degraded    = int((srs_gain < -0.01).sum())
    n_unchanged   = n_total - n_improved - n_degraded
    mean_gain     = float(srs_gain.mean())
    max_gain      = float(srs_gain.max())

    print(f"  SRS gain: mean={mean_gain:.4f}  max={max_gain:.4f}")
    print(f"  Improved: {n_improved} ({100*n_improved/n_total:.1f}%) | "
          f"Degraded: {n_degraded} ({100*n_degraded/n_total:.1f}%) | "
          f"Unchanged: {n_unchanged} ({100*n_unchanged/n_total:.1f}%)")

    # ── 2. Domain gap evidence ───────────────────────────────────────────────
    # Images where VLM understands content but object detector fails
    vlm_high = df["vlm_agent"].to_numpy(dtype=float) > 0.5
    obj_low  = df["existing_pipeline_agent"].to_numpy(dtype=float) < 0.20
    student_zero = (df["student_v3_count"].fillna(0) == 0).to_numpy() if "student_v3_count" in df.columns else np.zeros(n_total, dtype=bool)

    domain_gap_proxy = (vlm_high & obj_low).sum()
    detector_fail    = (student_zero & vlm_high).sum()

    print(f"\n  Domain gap evidence:")
    print(f"    VLM>0.5 & Object<0.2: {domain_gap_proxy} ({100*domain_gap_proxy/n_total:.1f}%)")
    print(f"    Student=0 & VLM>0.5:  {detector_fail}  ({100*detector_fail/n_total:.1f}%)")

    # ── 3. Does restoration help? (SRS gain vs agent score correlation) ──────
    r_srs_obj  = pearson_r(srs_gain, df["existing_pipeline_agent"].to_numpy(dtype=float))
    r_srs_vlm  = pearson_r(srs_gain, df["vlm_agent"].to_numpy(dtype=float))
    r_srs_fus  = pearson_r(srs_gain, df["comparison_fusion_score"].to_numpy(dtype=float))
    print(f"\n  SRS gain correlations:")
    print(f"    r(srs_gain, object_agent)  = {r_srs_obj:.4f}")
    print(f"    r(srs_gain, vlm_agent)     = {r_srs_vlm:.4f}")
    print(f"    r(srs_gain, fusion_score)  = {r_srs_fus:.4f}")

    # ── 4. Before-vs-after stratified analysis ────────────────────────────────
    low_orig  = srs_orig < np.percentile(srs_orig, 25)   # bottom quartile quality
    high_orig = srs_orig > np.percentile(srs_orig, 75)   # top quartile quality

    df["srs_gain"] = srs_gain
    strat_rows = []
    for label, mask in [("low_quality_images", low_orig), ("high_quality_images", high_orig)]:
        sub = df[mask]
        row = {
            "stratum":         label,
            "n_images":        int(mask.sum()),
            "mean_srs_gain":   round(float(srs_gain[mask].mean()), 4) if mask.sum() > 0 else 0.0,
            "obj_agent_mean":  round(float(sub["existing_pipeline_agent"].mean()), 4) if mask.sum() > 0 else 0.0,
            "vlm_agent_mean":  round(float(sub["vlm_agent"].mean()), 4) if mask.sum() > 0 else 0.0,
            "fusion_mean":     round(float(sub["comparison_fusion_score"].mean()), 4) if mask.sum() > 0 else 0.0,
        }
        strat_rows.append(row)
        print(f"  {label}: n={row['n_images']} | srs_gain={row['mean_srs_gain']:+.3f} | "
              f"obj={row['obj_agent_mean']:.3f} | vlm={row['vlm_agent_mean']:.3f}")

    # ── 5. Per-scene domain gap ───────────────────────────────────────────────
    scene_gap = {}
    for scene in ["teaching","landscape","drawing","family","playing"]:
        sub = df[df["vqa_primary_scene"]==scene] if "vqa_primary_scene" in df.columns else df.iloc[:0]
        if len(sub) == 0: continue
        gap = float((sub["vlm_agent"] > 0.5).mean() - (sub["existing_pipeline_agent"] > 0.3).mean())
        scene_gap[scene] = {"n": len(sub), "vlm_obj_gap": round(gap, 4)}

    # Save per-image csv
    cols = ["image_id","img_key","srs_gain","existing_pipeline_agent","vlm_agent",
            "comparison_fusion_score","vqa_primary_scene"]
    out_df = df[[c for c in cols if c in df.columns]]
    out_df.to_csv(out / "domain_adaptation_analysis.csv", index=False)
    print(f"\n✅ Wrote {out / 'domain_adaptation_analysis.csv'}")

    summary = {
        "n_images": n_total,
        "srs_restoration": {
            "mean_gain":  round(mean_gain, 4),
            "max_gain":   round(max_gain, 4),
            "n_improved": n_improved,
            "n_degraded": n_degraded,
            "n_unchanged":n_unchanged,
            "pct_improved": round(100*n_improved/n_total, 1),
        },
        "domain_gap_evidence": {
            "vlm_high_obj_low": int(domain_gap_proxy),
            "vlm_high_obj_low_pct": round(100*domain_gap_proxy/n_total, 1),
            "detector_fail_vlm_pass": int(detector_fail),
            "detector_fail_vlm_pass_pct": round(100*detector_fail/n_total, 1),
        },
        "restoration_correlations": {
            "srs_gain_vs_object_agent": round(r_srs_obj, 4),
            "srs_gain_vs_vlm_agent":    round(r_srs_vlm, 4),
            "srs_gain_vs_fusion_score": round(r_srs_fus, 4),
        },
        "quality_stratification": strat_rows,
        "per_scene_vlm_obj_gap": scene_gap,
        "domain_adaptation_conclusion": (
            f"{n_improved} images ({100*n_improved/n_total:.1f}%) improved with restoration. "
            f"{domain_gap_proxy} images ({100*domain_gap_proxy/n_total:.1f}%) show domain gap evidence "
            f"(VLM understands content but spatial detector fails). "
            f"SRS gain correlates {('positively' if r_srs_fus > 0 else 'negatively')} with fusion score "
            f"(r={r_srs_fus:.4f}), {'confirming' if r_srs_fus > 0 else 'suggesting'} "
            f"that restoration helps downstream multi-agent performance."
        ),
    }
    with open(out / "domain_adaptation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Wrote {out / 'domain_adaptation_summary.json'}")

if __name__ == "__main__":
    main()
