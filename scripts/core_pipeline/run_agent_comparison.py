#!/usr/bin/env python3
"""
Compare existing pipeline agent vs other agents and fusion.

Existing pipeline agent is represented by YOLO student detections
(from aligned detections), while other agents come from agreement,
scene, VLM, document, and restoration signals.

Improvements vs original:
  - Uses shared pipeline_utils to remove copy-pasted loader code.
  - Fusion weights are loaded from config.yaml so they stay in sync.
  - Adds a --config argument (consistent with run_multi_agent_validation.py).
  - Adds explicit weight normalisation so the script is robust to config edits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

# Allow importing from the same package directory
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_utils import (
    discover_agreement_scores,
    load_agreement_scores,
    load_jsonl_map,
    load_object_counts,
    load_scene_scores,
    load_srs_scores,
    load_vqa_scores,
    normalize_image_id,
)


def load_weights_from_config(config_path: Path) -> dict[str, float]:
    """
    Read multi_agent weights from config.yaml.
    Falls back to the original hard-coded defaults if the file is absent.
    """
    defaults = {
        "object": 0.25,
        "agreement": 0.20,
        "scene": 0.15,
        "vlm": 0.15,
        "restoration": 0.15,
        "document": 0.10,
    }
    if not config_path.exists():
        return defaults
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        mac = cfg.get("multi_agent", {})
        return {
            "object": float(mac.get("object_weight", defaults["object"])),
            "agreement": float(mac.get("agreement_weight", defaults["agreement"])),
            "scene": float(mac.get("scene_weight", defaults["scene"])),
            "vlm": float(mac.get("vlm_weight", defaults["vlm"])),
            "restoration": float(mac.get("restoration_weight", defaults["restoration"])),
            "document": float(mac.get("document_weight", defaults["document"])),
        }
    except Exception:
        return defaults


def build_manifest_fallbacks(manifest: pd.DataFrame, object_count_saturation: int, document_char_saturation: int):
    object_counts: dict[str, int] = {}
    scene: dict[str, float] = {}
    srs_raw: dict[str, float] = {}
    kosmos: dict[str, str] = {}
    vqa: dict[str, float] = {}
    for _, row in manifest.iterrows():
        img = normalize_image_id(row.get("image_id", ""))
        try:
            student_count = int(row.get("student_v3_count", 0) or 0)
        except (TypeError, ValueError):
            student_count = 0
        if student_count <= 0:
            try:
                student_count = int(float(row.get("vqa_total_objects", 0) or 0))
            except (TypeError, ValueError):
                student_count = 0
        object_counts[img] = max(student_count, 0)
        try:
            conf = float(row.get("vqa_scene_confidence", 0) or 0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf <= 0:
            scene_label = str(row.get("vqa_primary_scene", "") or "").strip().lower()
            conf = 0.5 if scene_label and scene_label != "nan" else 0.0
        scene[img] = max(0.0, min(1.0, conf))
        try:
            srs_raw[img] = float(row.get("srs_gain", 0.0) or 0.0)
        except (TypeError, ValueError):
            srs_raw[img] = 0.0
        kosmos[img] = str(row.get("kosmos_markdown", "") or "")
        try:
            vqa_total = float(row.get("vqa_total_objects", 0) or 0)
        except (TypeError, ValueError):
            vqa_total = 0.0
        vqa[img] = min(vqa_total / 5.0, 1.0)

    if srs_raw:
        mn, mx = min(srs_raw.values()), max(srs_raw.values())
        srs = {k: 0.5 for k in srs_raw} if (mx - mn < 1e-9) else {k: (v - mn) / (mx - mn) for k, v in srs_raw.items()}
    else:
        srs = srs_raw
    return object_counts, scene, srs, kosmos, vqa


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare existing pipeline and multi-agent signals.")
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--dataset-root", default="final_dataset")
    parser.add_argument("--output-dir", default="results/multi_agent")
    parser.add_argument("--object-count-saturation", type=int, default=3)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (defaults to <base-dir>/../../config.yaml)",
    )
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = base / dataset_root
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve config and load weights
    config_path = Path(args.config) if args.config else Path(__file__).resolve().parents[2] / "config.yaml"
    weights = load_weights_from_config(config_path)

    # Normalise weights so they always sum to 1.0 (robust to partial edits)
    w_total = sum(weights.values())
    if w_total > 0:
        weights = {k: v / w_total for k, v in weights.items()}

    manifest = pd.read_csv(dataset_root / "metadata" / "manifest.csv")
    ids = [normalize_image_id(x) for x in manifest["image_id"].astype(str)]

    raw_object_counts = load_object_counts(base / "results" / "aligned_detections" / "student_iter_3_aligned.json")
    raw_agreement = load_agreement_scores(base / "results" / "agreement_scores_v8" / "agreement_v8_final_results.csv")
    raw_scene = load_scene_scores(base / "results" / "scene_labels" / "scene_labels_clip.json")
    raw_vqa = load_vqa_scores(dataset_root / "metadata" / "vqa_binary_classification.json")
    raw_srs = load_srs_scores(dataset_root / "metadata" / "srs_scores.json")
    raw_kosmos = load_jsonl_map(base / "results" / "kosmos_grounding.jsonl", "image", "kosmos_output")

    fb_object, fb_scene, fb_srs, fb_kosmos, fb_vqa = build_manifest_fallbacks(
        manifest, args.object_count_saturation, 80
    )

    object_counts = raw_object_counts or fb_object
    agreement = raw_agreement or discover_agreement_scores(base)
    scene = raw_scene or fb_scene
    vqa = raw_vqa or fb_vqa
    srs = raw_srs or fb_srs
    kosmos = raw_kosmos or fb_kosmos

    source = lambda measured: "measured" if measured else "proxy"
    object_source = source(raw_object_counts)
    scene_source = source(raw_scene)
    vlm_source = source(raw_vqa)
    restoration_source = source(raw_srs)
    document_source = source(raw_kosmos)
    agreement_source = source(raw_agreement)

    rows = []
    for image_id in ids:
        existing_agent = min(object_counts.get(image_id, 0) / max(args.object_count_saturation, 1), 1.0)
        agreement_agent = max(0.0, min(1.0, agreement.get(image_id, 0.0)))
        scene_agent = max(0.0, min(1.0, scene.get(image_id, 0.0)))
        vlm_agent = max(0.0, min(1.0, vqa.get(image_id, 0.0)))
        restoration_agent = max(0.0, min(1.0, srs.get(image_id, 0.0)))
        document_agent = min(len(str(kosmos.get(image_id, ""))) / 80.0, 1.0)

        if image_id not in agreement:
            agreement_agent = max(0.0, 1.0 - abs(existing_agent - vlm_agent))
            agreement_source_row = "proxy"
        else:
            agreement_source_row = agreement_source

        # Monolithic baseline: no agreement agent (single opaque pipeline)
        # Decoupled enrichment: exclude restoration and document
        mono_w = {k: v for k, v in weights.items() if k not in ("agreement", "restoration", "document")}
        mono_total = sum(mono_w.values()) or 1.0
        monolithic_pipeline_agent = (
            mono_w["object"] / mono_total * existing_agent
            + mono_w["scene"] / mono_total * scene_agent
            + mono_w["vlm"] / mono_total * vlm_agent
        )

        # Coordinator fusion: validation agents only
        fusion_w = {k: v for k, v in weights.items() if k not in ("restoration", "document")}
        fusion_total = sum(fusion_w.values()) or 1.0
        comparison_fusion = (
            fusion_w["object"] / fusion_total * existing_agent
            + fusion_w["agreement"] / fusion_total * agreement_agent
            + fusion_w["scene"] / fusion_total * scene_agent
            + fusion_w["vlm"] / fusion_total * vlm_agent
        )

        rows.append({
            "image_id": image_id,
            "existing_pipeline_agent": round(existing_agent, 6),
            "agreement_agent": round(agreement_agent, 6),
            "scene_agent": round(scene_agent, 6),
            "vlm_agent": round(vlm_agent, 6),
            "restoration_agent": round(restoration_agent, 6),
            "document_agent": round(document_agent, 6),
            "monolithic_pipeline_agent": round(monolithic_pipeline_agent, 6),
            "comparison_fusion_score": round(comparison_fusion, 6),
            "agreement_source": agreement_source_row,
            "object_source": object_source,
            "scene_source": scene_source,
            "vlm_source": vlm_source,
            "restoration_source": restoration_source,
            "document_source": document_source,
        })

    df = pd.DataFrame(rows)
    agent_cols = [
        "existing_pipeline_agent", "agreement_agent", "scene_agent",
        "vlm_agent", "restoration_agent", "document_agent",
        "monolithic_pipeline_agent", "comparison_fusion_score",
    ]
    per_agent = {
        col: {
            "mean": float(df[col].mean()),
            "coverage_ratio": float((df[col] > 0).mean()),
        }
        for col in agent_cols
    }

    df.to_csv(output_dir / "agent_comparison_scores.csv", index=False)
    with open(output_dir / "agent_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_images": int(len(df)),
                "per_agent": per_agent,
                "weights_used": weights,
                "sources": {
                    "agreement_source": agreement_source,
                    "object_source": object_source,
                    "scene_source": scene_source,
                    "vlm_source": vlm_source,
                    "restoration_source": restoration_source,
                    "document_source": document_source,
                },
                "notes": "Monolithic baseline excludes agreement agent; coordinator fusion includes all.",
            },
            f,
            indent=2,
        )

    print(f"Wrote {output_dir / 'agent_comparison_scores.csv'}")
    print(f"Wrote {output_dir / 'agent_comparison_summary.json'}")


if __name__ == "__main__":
    main()
