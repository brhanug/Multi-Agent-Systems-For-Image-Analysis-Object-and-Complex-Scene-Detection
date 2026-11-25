#!/usr/bin/env python3
"""
run_europeana_transfer_ground_truth_eval.py
-------------------------------------------
Evaluates the C5 Europeana pilot against the human ground truth collected in
Label Studio ("Europeana REAL Pilot (genuine API images)", 23 of 24 images
labeled), answering the question the thesis previously listed as open:

    does error detection transfer to a new archive WITHOUT recalibration?

Two provenance issues in the pre-existing pilot artifacts are handled here
explicitly rather than silently trusted, because no script in the repository
generates `europeana_real_agent_scores_FINAL.csv`:

  1. `saa_fusion` in that file DOES reproduce the thesis canonical fusion
     formula (0.6*object + 0.2*scene + 0.2*vlm) exactly -- verified below and
     asserted at runtime, so it is safe to use.
  2. `u_saa` in that file does NOT. It is std(object, scene, vlm, ocr) with
     ddof=1, whereas the thesis canonical U_SAA is
     std(object, agreement, scene, vlm) with ddof=0 and
     agreement = 1 - |object - vlm|. This script recomputes U_SAA canonically
     and reports both, never citing the stored column as U_SAA.

Headline finding: the transfer question CANNOT be answered from this pilot,
because three of the four Tier 1 agents returned near-constant values on this
archive (VLM takes 2 distinct values across 24 images; scene stays low), so
saa_fusion correlates r=0.99 with the YOLO object score alone. Both the error
label and any disagreement score are then deterministic functions of one
number, and the resulting ROC-AUC is a degeneracy artifact, not a measurement.
What IS measurable is the classification transfer gap: the unchanged pipeline
misses 65.2% of human-confirmed scenes.

Output: results/external_archive/europeana_real_pilot/europeana_transfer_ground_truth_result.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path("/data/brhanu/thesis_project")
PILOT = BASE / "results/external_archive/europeana_real_pilot"
GT_CSV = PILOT / "europeana_human_annotations_REAL.csv"
SCORES_CSV = PILOT / "europeana_real_agent_scores_FINAL.csv"
OUT_JSON = PILOT / "europeana_transfer_ground_truth_result.json"

DECISION_THRESHOLD = 0.5  # same rule as the thesis: sign(S_fusion >= 0.5)


def main():
    gt = pd.read_csv(GT_CSV)
    sc = pd.read_csv(SCORES_CSV)
    m = gt.merge(sc, on="image_id", suffixes=("_gt", ""))
    assert len(m) == len(gt), f"join lost rows: {len(m)} of {len(gt)}"

    obj, scene, vlm, ocr = (m.existing_pipeline_agent, m.scene_agent,
                            m.vlm_agent, m.document_agent)

    # --- provenance guard: the stored fusion score must match the thesis formula
    saa_recomputed = 0.6 * obj + 0.2 * scene + 0.2 * vlm
    assert np.abs(m.saa_fusion - saa_recomputed).max() < 1e-9, \
        "saa_fusion does not reproduce the canonical 0.6/0.2/0.2 formula"

    # --- canonical U_SAA (NOT the stored u_saa column, which uses OCR + ddof=1)
    agreement = 1 - (obj - vlm).abs()
    u_saa_canonical = np.std(np.vstack([obj, agreement, scene, vlm]), axis=0, ddof=0)

    m["pred_scene"] = (m.saa_fusion >= DECISION_THRESHOLD).astype(int)
    m["error"] = (m.pred_scene != m.has_scene).astype(int)

    n = len(m)
    n_err = int(m.error.sum())
    n_pos = int(m.has_scene.sum())

    # --- degeneracy check: is the ensemble actually multi-modal on this archive?
    variation = {c: {"n_unique": int(m[c].nunique()), "std": round(float(m[c].std()), 4)}
                 for c in ["existing_pipeline_agent", "scene_agent", "vlm_agent", "document_agent"]}
    r_fusion_yolo = float(m.saa_fusion.corr(m.existing_pipeline_agent))
    degenerate = variation["vlm_agent"]["n_unique"] <= 3 and abs(r_fusion_yolo) > 0.95

    result = {
        "n_labeled": n,
        "n_human_scene_positive": n_pos,
        "n_negative_controls": n - n_pos,
        "n_pipeline_errors": n_err,
        "false_negative_rate": round(n_err / n, 4),
        "n_images_yolo_found_nothing": int((m.n_yolo_objects == 0).sum()),
        "agent_variation_across_archive": variation,
        "corr_saa_fusion_with_yolo_alone": round(r_fusion_yolo, 4),
        "ensemble_degenerate_on_this_archive": bool(degenerate),
        "u_saa_canonical_rocauc": round(float(roc_auc_score(m.error, u_saa_canonical)), 4),
        "u_saa_as_stored_rocauc": round(float(roc_auc_score(m.error, m.u_saa)), 4),
        "rocauc_is_interpretable": not bool(degenerate),
        "interpretation": (
            f"All {n_pos} human-labeled Europeana images are scene-positive, so this sample "
            f"contains no negative controls and the error task is entirely false-negative "
            f"detection. Applied with no recalibration, the pipeline misses {n_err} of {n} "
            f"human-confirmed scenes ({n_err/n*100:.1f}%), driven by YOLO returning zero objects "
            f"on {int((m.n_yolo_objects == 0).sum())} images. The error-DETECTION question cannot be "
            f"answered from this pilot: the VLM agent takes only "
            f"{variation['vlm_agent']['n_unique']} distinct values across the archive and the scene "
            f"agent stays low, so saa_fusion correlates r={r_fusion_yolo:.3f} with the YOLO score "
            f"alone. Both the error label and any disagreement score are then functions of that one "
            f"number, making the resulting ROC-AUC a degeneracy artifact rather than a measurement. "
            f"A valid transfer test requires the ensemble to remain genuinely multi-modal on the new "
            f"archive, which it did not here."
        ),
        "provenance_notes": [
            "saa_fusion verified to reproduce the thesis canonical 0.6/0.2/0.2 fusion exactly.",
            "The stored u_saa column is std(object,scene,vlm,ocr) ddof=1, NOT the thesis canonical "
            "std(object,agreement,scene,vlm) ddof=0; canonical U_SAA is recomputed here.",
            "-saa_fusion scores ROC-AUC=1.0 against this error label purely by construction "
            "(the label is that score thresholded at 0.5) and is not a valid baseline.",
        ],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(OUT_JSON, "w"), indent=2)

    print(f"n={n}, all scene-positive={n_pos}, errors={n_err} ({n_err/n*100:.1f}% false negatives)")
    print(f"YOLO found nothing on {result['n_images_yolo_found_nothing']} images")
    print(f"corr(saa_fusion, YOLO alone) = {r_fusion_yolo:.4f} -> ensemble degenerate: {degenerate}")
    print(f"canonical U_SAA ROC-AUC = {result['u_saa_canonical_rocauc']} "
          f"(interpretable: {result['rocauc_is_interpretable']})")
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
