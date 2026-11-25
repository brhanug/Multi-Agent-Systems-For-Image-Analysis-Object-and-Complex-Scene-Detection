#!/usr/bin/env python3
"""
compute_real_interannotator_kappa.py
--------------------------------------
Computes a genuine Cohen's Kappa inter-annotator agreement statistic for
scene-presence/category judgments, using two annotation passes that were
already collected independently, by two different people, through two
different tools, for two different reasons -- not a purpose-built
double-annotation study.

  Rater A (C1, CVAT expert-reviewed cohort): scene-presence review.
  Rater B (C2, n=300): Label Studio spatial-audit task. The Label Studio
    task data for each image was ONLY the raw image URL (verified directly
    against the Label Studio SQLite task table) -- Rater B was never shown
    Rater A's label.

CRITICAL: human_spatial_audit/user_annotations_800.csv has 2,000 rows, but
only the first 801 (cvat_id 0-800) are Rater A's genuine CVAT review (796
scene-positive + 5 negative, matching the documented C1 cohort). Rows 801
onward (1,199 rows) are a DIFFERENT, unverified pool -- every single one
has n_scene_labels=0, n_boxes=0, scene_labels=NaN with zero variation
across all 1,199 rows, and no script anywhere in this codebase documents
how they were produced. That is not consistent with genuine human review
of 1,199 additional images (some would show a scene); it is consistent
with catalog records defaulted to "no scene" without ever being looked at.

An earlier version of this script did not filter on cvat_id and matched
176 of C2's 300 images against this fake range, silently comparing Rater
B's real judgments against a meaningless default rather than a second
human's judgment for those 176 -- producing an artifactual "58.7% vs
20.0%" base-rate story and a deflated kappa. This version restricts the
comparison to the 124 images that are genuinely double-reviewed (matched
against cvat_id<=800 only).

Ground truth for rows 801+ is a separate, unresolved data-integrity
question (see the thesis and slides for the disclosed limitation) --
it must NOT be used as if it were human-verified anywhere, including here.

Outputs:
  results/multi_agent/real_interannotator_kappa.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

BASE = Path("/data/brhanu/thesis_project")
C1_CSV = BASE / "human_spatial_audit" / "user_annotations_800.csv"
C2_CSV = BASE / "human_spatial_audit" / "user_annotations_300.csv"
OUT_JSON = BASE / "results" / "multi_agent" / "real_interannotator_kappa.json"

REAL_REVIEW_CUTOFF = 800  # inclusive; rows beyond this are the unverified pool


def main() -> None:
    c1 = pd.read_csv(C1_CSV)
    c2 = pd.read_csv(C2_CSV)

    c1["cvat_id"] = c1.index
    c1["full_id"] = (
        c1["raw_name"].str.replace("images/", "", regex=False).str.replace(".jpg", "", regex=False)
    )

    merged_all = c2.merge(c1, left_on="image_id", right_on="full_id", how="inner", suffixes=("_c2", "_c1"))
    merged = merged_all[merged_all["cvat_id"] <= REAL_REVIEW_CUTOFF].copy()
    n_excluded = len(merged_all) - len(merged)

    print(f"C2 images matched to C1 pool at all: {len(merged_all)}")
    print(f"Of those, genuinely double-reviewed (cvat_id<={REAL_REVIEW_CUTOFF}): {len(merged)}")
    print(f"Excluded as unverified-pool matches: {n_excluded}")

    rater_a = merged["scene_labels"].fillna("none")  # C1 / CVAT, real review only
    rater_b = merged["scene"].fillna("none")          # C2 / Label Studio

    categories = sorted(set(rater_a) | set(rater_b))
    kappa = cohen_kappa_score(rater_a, rater_b, labels=categories)

    rater_a_bin = (rater_a != "none").astype(int)
    rater_b_bin = (rater_b != "none").astype(int)
    a_has_variance = rater_a_bin.nunique() > 1
    kappa_binary = cohen_kappa_score(rater_a_bin, rater_b_bin) if a_has_variance else None
    pct_agree_binary = float((rater_a_bin == rater_b_bin).mean())

    cm = confusion_matrix(rater_a, rater_b, labels=categories)

    result = {
        "n_c2_total": 300,
        "n_matched_to_c1_pool": int(len(merged_all)),
        "n_genuinely_double_reviewed": int(len(merged)),
        "n_excluded_unverified_pool": int(n_excluded),
        "categories": categories,
        "categorical_cohens_kappa": round(float(kappa), 4),
        "binary_has_scene_cohens_kappa": round(float(kappa_binary), 4) if kappa_binary is not None else None,
        "binary_kappa_note": None if a_has_variance else (
            "Undefined/degenerate: Rater A has zero 'no scene' judgments in this "
            "124-image overlap (all 5 of C1's real negatives fall outside it), so "
            "Rater A's binary column has no variance and kappa is not meaningful."
        ),
        "binary_pct_agreement": round(pct_agree_binary, 4),
        "rater_a_no_scene_rate": round(float((rater_a == "none").mean()), 4),
        "rater_b_no_scene_rate": round(float((rater_b == "none").mean()), 4),
        "confusion_matrix": cm.tolist(),
        "rater_a": "C1 / CVAT expert scene-presence review, cvat_id<=800 ONLY "
                   "(the genuinely reviewed subset)",
        "rater_b": "C2 / Label Studio spatial-audit task (user_annotations_300.csv)",
        "note": "Two different people, two different tools, verified independent. "
                "Restricted to the 124 C2 images that land in C1's genuinely "
                f"reviewed range; {n_excluded} of C2's 300 images matched only "
                "against an unverified default-labeled pool and are excluded, "
                "not treated as a second rater's judgment.",
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))

    print(f"\nCategorical Cohen's Kappa (5 scene types + none): {result['categorical_cohens_kappa']}")
    print(f"Binary Cohen's Kappa (has_scene vs no_scene): {result['binary_has_scene_cohens_kappa']} "
          f"{'(' + result['binary_kappa_note'] + ')' if result['binary_kappa_note'] else ''}")
    print(f"Binary raw agreement: {result['binary_pct_agreement']:.1%}")
    print(f"\nConfusion matrix (rows=Rater A/C1, cols=Rater B/C2), order={categories}:")
    print(pd.DataFrame(cm, index=categories, columns=categories))
    print(f"\n✅ Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
