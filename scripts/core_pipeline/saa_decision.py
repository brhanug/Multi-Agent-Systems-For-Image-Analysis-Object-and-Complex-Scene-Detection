#!/usr/bin/env python3
"""The canonical SAA scene-present decision rule.

Why this module exists
----------------------
The deployed weighting w = (0.6, 0.2, 0.2) places one cell of the signal space,
(A_object, A_scene, A_vlm) = (1/3, 0.5, 1), at exactly the 0.5 decision threshold:

    0.6*(1/3) + 0.2*(0.5) + 0.2*(1) = 0.2 + 0.1 + 0.2 = 0.5

That cell holds 2,217 of the 12,110 corpus images (18.3%) and 276 of the 1,250
labelled pool (22.1%). Because A_object is stored to six decimal places as 0.333333
rather than as a third, the cell actually evaluates to 0.4999998 and falls on the
negative side of `score >= 0.5` by two parts in ten million. The classification of a
fifth of the corpus therefore rested on a stored decimal truncation, and call sites
that used `> 0.5` rather than `>= 0.5` were only agreeing with the rest by accident.

The fix keeps every published number identical while making the decision deliberate:

  1. Snap A_object back to the exact third it represents, so scores are exact.
  2. Threshold at 0.55, which lies inside the empty interval (0.5, 0.6) separating the
     two nearest attainable scores. The eight attainable SAA values are
     {0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, so the nearest score is 0.05 away
     instead of 0.0000002, and `>=` versus `>` can no longer disagree.

This reproduces the legacy rule on every image in the corpus -- verified by
`check_equivalence()` below and by the regression guard in `verify()`. It is a
robustness fix, not a behaviour change: the negative side of the boundary keeps the
knife-edge cell, which is the conservative choice the deployed system has in fact been
making, now made on purpose and recorded as such.
"""
from __future__ import annotations
import numpy as np

SAA_WEIGHTS = (0.6, 0.2, 0.2)          # (object, scene, vlm) -- unchanged
SAA_THRESHOLD = 0.55                    # deliberate: inside the empty gap (0.5, 0.6)
LEGACY_THRESHOLD = 0.5                  # what the stored-truncation rule used

# The stored `comparison_fusion_score` column is NOT the same number as a recomputation
# of 0.6*A_object + 0.2*A_scene + 0.2*A_vlm from the stored agent columns. On the cell
# (1/3, 0.5, 1) the stored column holds the exact 0.5, while recomputation from the
# truncated A_object = 0.333333 yields 0.4999998. Thresholded at 0.5 the two rules
# therefore disagree on 2,217 of 12,110 images (18.3%) -- the stored column calls them
# scene-present, the recomputation calls them scene-absent. Both rules appear in the
# published results. This constant marks the call sites that consume the stored column,
# so the divergence is explicit rather than accidental; it must not be "unified" without
# re-deriving every downstream number that depends on it.
STORED_COLUMN_THRESHOLD = 0.5
MIN_MARGIN = 0.01                       # guard: no attainable score may sit closer


def exact_object(a_object):
    """Undo the six-decimal storage truncation: snap A_object to the nearest third.

    A_object = min(count/3, 1) by construction, so every true value is k/3 for
    k in {0,1,2,3}. Stored values are 0.0, 0.333333, 0.666667, 1.0.
    """
    a = np.asarray(a_object, dtype=float)
    return np.round(a * 3.0) / 3.0


def saa_score(a_object, a_scene, a_vlm):
    """The fusion score, computed from exact thirds."""
    wo, ws, wv = SAA_WEIGHTS
    return (wo * exact_object(a_object)
            + ws * np.asarray(a_scene, dtype=float)
            + wv * np.asarray(a_vlm, dtype=float))


def scene_present(a_object, a_scene, a_vlm):
    """The canonical binary prediction. Use this instead of a bare `>= 0.5`."""
    return saa_score(a_object, a_scene, a_vlm) >= SAA_THRESHOLD


def legacy_scene_present(a_object, a_scene, a_vlm):
    """The rule as it behaved with stored truncation, for equivalence checking only."""
    wo, ws, wv = SAA_WEIGHTS
    s = (wo * np.asarray(a_object, dtype=float)
         + ws * np.asarray(a_scene, dtype=float)
         + wv * np.asarray(a_vlm, dtype=float))
    return s >= LEGACY_THRESHOLD


def check_equivalence(df, o="existing_pipeline_agent", s="scene_agent", v="vlm_agent"):
    """Confirm the canonical rule matches the legacy rule on every row."""
    new = scene_present(df[o], df[s], df[v])
    old = legacy_scene_present(df[o], df[s], df[v])
    diff = int((new != old).sum())
    return {"n": int(len(df)), "disagreements": diff, "identical": diff == 0}


def check_margin(df, o="existing_pipeline_agent", s="scene_agent", v="vlm_agent"):
    """Guard: no attainable score may sit within MIN_MARGIN of the threshold."""
    vals = np.unique(saa_score(df[o], df[s], df[v]))
    closest = float(np.min(np.abs(vals - SAA_THRESHOLD)))
    return {"attainable_scores": [round(float(x), 6) for x in vals],
            "closest_to_threshold": round(closest, 6),
            "min_required": MIN_MARGIN,
            "safe": bool(closest >= MIN_MARGIN)}


def verify(df):
    """Run both guards. Returns (ok, report)."""
    eq, mg = check_equivalence(df), check_margin(df)
    return bool(eq["identical"] and mg["safe"]), {"equivalence": eq, "margin": mg}


if __name__ == "__main__":
    import sys, json
    from pathlib import Path
    import pandas as pd
    BASE = Path("/data/brhanu/thesis_project")
    ok_all = True
    for name, path in [("full corpus", BASE / "results/multi_agent/agent_comparison_scores.csv")]:
        df = pd.read_csv(path).drop_duplicates("image_id")
        ok, rep = verify(df)
        ok_all &= ok
        print(f"--- {name} ---")
        print(json.dumps(rep, indent=2))
        print("RESULT:", "SAFE AND EQUIVALENT" if ok else "*** FAILED ***")
    sys.exit(0 if ok_all else 1)
