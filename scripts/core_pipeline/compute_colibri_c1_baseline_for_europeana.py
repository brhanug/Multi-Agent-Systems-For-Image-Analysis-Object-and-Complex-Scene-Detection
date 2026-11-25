#!/usr/bin/env python3
"""
compute_colibri_c1_baseline_for_europeana.py
-----------------------------------------------
Computes the real Colibri $C_1$ baseline mean SAA fusion score, for direct
comparison against the genuine Europeana pilot's mean SAA fusion score
(0.332, see run_europeana_real_pilot_inference.py).

Uses the exact same C1-matching methodology as exp29_conformal_risk_control.py
(join the 801 CVAT expert-reviewed images against agent_comparison_scores.csv
by normalized image ID) so the two "C1 baseline" numbers used throughout this
thesis are computed identically and stay consistent.

A full-project reproducibility re-run (per user request) found this value
computed at 0.7639 (rounds to 0.764), not the 0.762 previously written in
the thesis/slides -- the earlier number was a manual/ad-hoc computation
that was never itself saved as a script. This script exists so the number
is reproducible going forward.

Outputs: prints the mean SAA fusion score on the matched C1 cohort.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
EXPERT_CSV = BASE / "human_spatial_audit" / "user_annotations_800.csv"
SCORES_CSV = BASE / "results" / "multi_agent" / "agent_comparison_scores.csv"
OUT_JSON = BASE / "results" / "multi_agent" / "colibri_c1_baseline.json"


from picture_ids import full_id as normalize_id, read_expert_annotations   # book and page, never the page alone


def main() -> None:
    expert = read_expert_annotations(EXPERT_CSV)
    expert["cvat_id"] = expert.index
    expert_reviewed = expert[expert["cvat_id"] <= 800].copy()

    agents = pd.read_csv(SCORES_CSV)
    agents["image_id"] = agents["image_id"].apply(normalize_id)
    agents = agents.drop_duplicates(subset=["image_id"])

    joined = expert_reviewed.merge(agents, on="image_id", how="inner")
    assert len(joined) == 801, f"expected 801 matched images, got {len(joined)}"

    mean_fusion = joined["comparison_fusion_score"].mean()
    u_saa = joined[["existing_pipeline_agent", "agreement_agent", "scene_agent", "vlm_agent"]].std(axis=1).mean()
    print(f"Mean Colibri C1 U_SAA (sample std of the four signals): {u_saa:.4f}")
    print(f"Matched {len(joined)} C1 images.")
    print(f"Mean Colibri C1 SAA fusion score: {mean_fusion:.4f} (rounds to {mean_fusion:.3f})")
    print("Compare against the genuine Europeana pilot's mean SAA fusion score: 0.332")
    OUT_JSON.write_text(json.dumps({"pictures": int(len(joined)), "mean_fusion_score": round(float(mean_fusion), 4),
                                    "mean_U_SAA": round(float(u_saa), 4)}, indent=2))


if __name__ == "__main__":
    main()
