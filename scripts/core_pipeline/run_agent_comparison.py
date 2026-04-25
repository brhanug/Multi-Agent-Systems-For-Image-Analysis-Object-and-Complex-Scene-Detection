#!/usr/bin/env python3
"""
Compare existing pipeline agent vs other agents and fusion.

Existing pipeline agent is represented by YOLO student detections
(from aligned detections), while other agents come from agreement,
scene, VLM, document, and restoration signals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_image_id(raw: str) -> str:
    return Path(str(raw)).stem


def safe_load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_map(path: Path, key: str, value: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                out[normalize_image_id(obj.get(key, ""))] = obj.get(value, "")
            except json.JSONDecodeError:
                continue
    return out


def load_object_counts(path: Path) -> dict[str, int]:
    data = safe_load_json(path)
    out: dict[str, int] = {}
    if not isinstance(data, dict):
        return out
    for k, dets in data.items():
        out[normalize_image_id(k)] = len(dets) if isinstance(dets, list) else 0
    return out


def load_agreement(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    df = pd.read_csv(path)
    if "image_id" not in df.columns or "final_score" not in df.columns:
        return out
    for _, row in df.iterrows():
        out[normalize_image_id(row["image_id"])] = float(row["final_score"])
    return out


def discover_agreement(base: Path) -> dict[str, float]:
    candidates = [
        base / "results" / "agreement_scores_v8" / "agreement_v8_final_results.csv",
        base / "results" / "agreement_scores_final" / "agreement_v8_final_results.csv",
        base / "results" / "agreement_scores_final" / "agreement_final_results.csv",
    ]
    for c in candidates:
        ag = load_agreement(c)
        if ag:
            return ag
    return {}


def load_scene(path: Path) -> dict[str, float]:
    data = safe_load_json(path)
    out: dict[str, float] = {}
    if not isinstance(data, dict):
        return out
    for k, vals in data.items():
        score = 0.0
        if isinstance(vals, list) and vals:
            top = vals[0]
            if isinstance(top, list) and len(top) >= 2:
                try:
                    score = float(top[1])
                except (TypeError, ValueError):
                    score = 0.0
            elif isinstance(top, dict):
                try:
                    score = float(top.get("score", 0.0))
                except (TypeError, ValueError):
                    score = 0.0
        out[normalize_image_id(k)] = max(0.0, min(1.0, score))
    return out


def load_vqa(path: Path) -> dict[str, float]:
    data = safe_load_json(path)
    out: dict[str, float] = {}
    if data is None:
        return out

    def to_bin(v: Any) -> float | None:
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            lv = v.strip().lower()
            if lv in {"yes", "true", "1"}:
                return 1.0
            if lv in {"no", "false", "0"}:
                return 0.0
        return None

    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            vals = []
            for k, v in row.items():
                if k in {"image_id", "image"}:
                    continue
                b = to_bin(v)
                if b is not None:
                    vals.append(b)
            if vals:
                out[normalize_image_id(row.get("image_id", row.get("image", "")))] = sum(vals) / len(vals)
    elif isinstance(data, dict):
        for k, payload in data.items():
            vals = []
            if isinstance(payload, dict):
                for v in payload.values():
                    b = to_bin(v)
                    if b is not None:
                        vals.append(b)
            else:
                b = to_bin(payload)
                if b is not None:
                    vals.append(b)
            if vals:
                out[normalize_image_id(k)] = sum(vals) / len(vals)
    return out


def load_srs(path: Path) -> dict[str, float]:
    data = safe_load_json(path)
    raw: dict[str, float] = {}
    if not isinstance(data, list):
        return raw
    for row in data:
        if not isinstance(row, dict):
            continue
        img = normalize_image_id(row.get("image_id", row.get("image", "")))
        raw[img] = float(row.get("srs_gain", 0.0))
    if not raw:
        return raw
    mn, mx = min(raw.values()), max(raw.values())
    if mx - mn < 1e-9:
        return {k: 0.5 for k in raw}
    return {k: (v - mn) / (mx - mn) for k, v in raw.items()}


def build_manifest_fallbacks(manifest: pd.DataFrame) -> tuple[dict[str, int], dict[str, float], dict[str, float], dict[str, str], dict[str, float]]:
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

    mn, mx = min(srs_raw.values()), max(srs_raw.values())
    srs = {k: 0.5 for k in srs_raw} if (mx - mn < 1e-9) else {k: (v - mn) / (mx - mn) for k, v in srs_raw.items()}
    return object_counts, scene, srs, kosmos, vqa


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare existing pipeline and multi-agent signals.")
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--dataset-root", default="final_dataset")
    parser.add_argument("--output-dir", default="results/multi_agent")
    parser.add_argument("--object-count-saturation", type=int, default=3)
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = base / dataset_root
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(dataset_root / "metadata" / "manifest.csv")
    ids = [normalize_image_id(x) for x in manifest["image_id"].astype(str)]

    raw_object_counts = load_object_counts(base / "results" / "aligned_detections" / "student_iter_3_aligned.json")
    raw_agreement = load_agreement(base / "results" / "agreement_scores_v8" / "agreement_v8_final_results.csv")
    raw_scene = load_scene(base / "results" / "scene_labels" / "scene_labels_clip.json")
    raw_vqa = load_vqa(dataset_root / "metadata" / "vqa_binary_classification.json")
    raw_srs = load_srs(dataset_root / "metadata" / "srs_scores.json")
    raw_kosmos = load_jsonl_map(base / "results" / "kosmos_grounding.jsonl", "image", "kosmos_output")

    object_counts = raw_object_counts
    agreement = raw_agreement
    scene = raw_scene
    vqa = raw_vqa
    srs = raw_srs
    kosmos = raw_kosmos
    fb_object, fb_scene, fb_srs, fb_kosmos, fb_vqa = build_manifest_fallbacks(manifest)
    if not object_counts:
        object_counts = fb_object
    if not scene:
        scene = fb_scene
    if not srs:
        srs = fb_srs
    if not kosmos:
        kosmos = fb_kosmos
    if not vqa:
        vqa = fb_vqa
    if not agreement:
        agreement = discover_agreement(base)

    object_source = "measured" if raw_object_counts else "proxy"
    scene_source = "measured" if raw_scene else "proxy"
    vlm_source = "measured" if raw_vqa else "proxy"
    restoration_source = "measured" if raw_srs else "proxy"
    document_source = "measured" if raw_kosmos else "proxy"
    agreement_source = "measured" if raw_agreement or agreement else "proxy"

    rows = []
    for image_id in ids:
        existing_agent = min(object_counts.get(image_id, 0) / max(args.object_count_saturation, 1), 1.0)
        agreement_agent = max(0.0, min(1.0, agreement.get(image_id, 0.0)))
        scene_agent = max(0.0, min(1.0, scene.get(image_id, 0.0)))
        vlm_agent = max(0.0, min(1.0, vqa.get(image_id, 0.0)))
        restoration_agent = max(0.0, min(1.0, srs.get(image_id, 0.0)))
        document_agent = min(len(str(kosmos.get(image_id, ""))) / 80.0, 1.0)

        # agreement fallback proxy if agreement artifacts are unavailable
        if image_id not in agreement:
            agreement_agent = max(0.0, 1.0 - abs(existing_agent - vlm_agent))
            agreement_source_row = "proxy"
        else:
            agreement_source_row = agreement_source
        monolithic_pipeline_agent = (
            0.45 * existing_agent
            + 0.20 * scene_agent
            + 0.20 * vlm_agent
            + 0.10 * restoration_agent
            + 0.05 * document_agent
        )
        comparison_fusion = (
            0.30 * existing_agent
            + 0.20 * agreement_agent
            + 0.15 * scene_agent
            + 0.15 * vlm_agent
            + 0.10 * restoration_agent
            + 0.10 * document_agent
        )

        rows.append(
            {
                "image_id": image_id,
                "existing_pipeline_agent": existing_agent,
                "agreement_agent": agreement_agent,
                "scene_agent": scene_agent,
                "vlm_agent": vlm_agent,
                "restoration_agent": restoration_agent,
                "document_agent": document_agent,
                "monolithic_pipeline_agent": monolithic_pipeline_agent,
                "comparison_fusion_score": comparison_fusion,
                "agreement_source": agreement_source_row,
                "object_source": object_source,
                "scene_source": scene_source,
                "vlm_source": vlm_source,
                "restoration_source": restoration_source,
                "document_source": document_source,
                "metric_source": "mixed" if "proxy" in {agreement_source_row, object_source, scene_source, vlm_source, restoration_source, document_source} else "measured",
            }
        )

    df = pd.DataFrame(rows)
    per_agent = {
        "existing_pipeline_agent": {
            "mean": float(df["existing_pipeline_agent"].mean()),
            "coverage_ratio": float((df["existing_pipeline_agent"] > 0).mean()),
        },
        "agreement_agent": {
            "mean": float(df["agreement_agent"].mean()),
            "coverage_ratio": float((df["agreement_agent"] > 0).mean()),
        },
        "scene_agent": {
            "mean": float(df["scene_agent"].mean()),
            "coverage_ratio": float((df["scene_agent"] > 0).mean()),
        },
        "vlm_agent": {
            "mean": float(df["vlm_agent"].mean()),
            "coverage_ratio": float((df["vlm_agent"] > 0).mean()),
        },
        "restoration_agent": {
            "mean": float(df["restoration_agent"].mean()),
            "coverage_ratio": float((df["restoration_agent"] > 0).mean()),
        },
        "document_agent": {
            "mean": float(df["document_agent"].mean()),
            "coverage_ratio": float((df["document_agent"] > 0).mean()),
        },
        "monolithic_pipeline_agent": {
            "mean": float(df["monolithic_pipeline_agent"].mean()),
            "coverage_ratio": float((df["monolithic_pipeline_agent"] > 0).mean()),
        },
        "comparison_fusion_score": {
            "mean": float(df["comparison_fusion_score"].mean()),
            "coverage_ratio": float((df["comparison_fusion_score"] > 0).mean()),
        },
    }

    df.to_csv(output_dir / "agent_comparison_scores.csv", index=False)
    with open(output_dir / "agent_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_images": int(len(df)),
                "per_agent": per_agent,
                "sources": {
                    "agreement_source": agreement_source,
                    "object_source": object_source,
                    "scene_source": scene_source,
                    "vlm_source": vlm_source,
                    "restoration_source": restoration_source,
                    "document_source": document_source,
                    "metric_source": "mixed" if "proxy" in {agreement_source, object_source, scene_source, vlm_source, restoration_source, document_source} else "measured",
                },
                "notes": "Includes monolithic baseline (single opaque score) and coordinator fusion (agreement-aware).",
            },
            f,
            indent=2,
        )

    print(f"✅ Wrote {output_dir / 'agent_comparison_scores.csv'}")
    print(f"✅ Wrote {output_dir / 'agent_comparison_summary.json'}")


if __name__ == "__main__":
    main()
