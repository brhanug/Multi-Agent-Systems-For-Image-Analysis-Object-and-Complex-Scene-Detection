#!/usr/bin/env python3
"""
Run multi-agent validation for object + complex scene realism.

This script aggregates outputs from existing agents and computes:
1) per-agent validation scores
2) an overall realism score
3) an uncertainty score + HITL review flag
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import pstdev
from typing import Any

import pandas as pd
import yaml


@dataclass
class MultiAgentConfig:
    object_weight: float
    agreement_weight: float
    scene_weight: float
    vlm_weight: float
    document_weight: float
    restoration_weight: float
    min_agents_required: int
    uncertainty_threshold: float
    uncertainty_threshold_quantile: float | None
    realism_threshold: float
    object_count_saturation: int
    document_char_saturation: int


def load_config(config_path: Path) -> tuple[Path, MultiAgentConfig]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    project_base = Path(cfg.get("project", {}).get("base_dir", config_path.parents[2])).resolve()
    mac = cfg.get("multi_agent", {})

    return project_base, MultiAgentConfig(
        object_weight=float(mac.get("object_weight", 0.25)),
        agreement_weight=float(mac.get("agreement_weight", 0.2)),
        scene_weight=float(mac.get("scene_weight", 0.15)),
        vlm_weight=float(mac.get("vlm_weight", 0.15)),
        document_weight=float(mac.get("document_weight", 0.1)),
        restoration_weight=float(mac.get("restoration_weight", 0.15)),
        min_agents_required=int(mac.get("min_agents_required", 3)),
        uncertainty_threshold=float(mac.get("uncertainty_threshold", 0.2)),
        uncertainty_threshold_quantile=(
            float(mac["uncertainty_threshold_quantile"])
            if mac.get("uncertainty_threshold_quantile") is not None
            else None
        ),
        realism_threshold=float(mac.get("realism_threshold", 0.5)),
        object_count_saturation=int(mac.get("object_count_saturation", 3)),
        document_char_saturation=int(mac.get("document_char_saturation", 80)),
    )


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


def load_agreement_scores(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = normalize_image_id(row.get("image_id", ""))
            try:
                out[img] = float(row.get("final_score", 0.0))
            except (TypeError, ValueError):
                out[img] = 0.0
    return out


def discover_agreement_scores(project_base: Path) -> dict[str, float]:
    candidates = [
        project_base / "results" / "agreement_scores_v8" / "agreement_v8_final_results.csv",
        project_base / "results" / "agreement_scores_final" / "agreement_v8_final_results.csv",
        project_base / "results" / "agreement_scores_final" / "agreement_final_results.csv",
    ]
    for c in candidates:
        scores = load_agreement_scores(c)
        if scores:
            return scores
    return {}


def load_scene_scores(path: Path) -> dict[str, float]:
    """
    Expected flexible formats:
    - {image_id: [[label, score], ...]}
    - {image_id: [{"label": "...", "score": 0.9}, ...]}
    """
    data = safe_load_json(path)
    out: dict[str, float] = {}
    if not isinstance(data, dict):
        return out

    for k, vals in data.items():
        img = normalize_image_id(k)
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
        out[img] = max(0.0, min(1.0, score))
    return out


def load_object_counts(path: Path) -> dict[str, int]:
    data = safe_load_json(path)
    out: dict[str, int] = {}
    if not isinstance(data, dict):
        return out
    for k, dets in data.items():
        img = normalize_image_id(k)
        if isinstance(dets, list):
            out[img] = len(dets)
        else:
            out[img] = 0
    return out


def load_vqa_scores(path: Path) -> dict[str, float]:
    """
    Flexible support:
    - list of {"image_id": "...", "...binary fields..."}
    - dict image_id -> dict(question->0/1 or yes/no)
    """
    data = safe_load_json(path)
    out: dict[str, float] = {}
    if data is None:
        return out

    def mean_binary(values: list[Any]) -> float:
        nums = []
        for v in values:
            if isinstance(v, bool):
                nums.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)):
                nums.append(float(v))
            elif isinstance(v, str):
                lv = v.strip().lower()
                if lv in {"yes", "true", "1"}:
                    nums.append(1.0)
                elif lv in {"no", "false", "0"}:
                    nums.append(0.0)
        if not nums:
            return 0.0
        return max(0.0, min(1.0, sum(nums) / len(nums)))

    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            img = normalize_image_id(row.get("image_id", row.get("image", "")))
            vals = [v for k, v in row.items() if k not in {"image_id", "image"}]
            out[img] = mean_binary(vals)
    elif isinstance(data, dict):
        for k, payload in data.items():
            img = normalize_image_id(k)
            if isinstance(payload, dict):
                out[img] = mean_binary(list(payload.values()))
            else:
                out[img] = mean_binary([payload])
    return out


def load_srs_scores(path: Path) -> dict[str, float]:
    data = safe_load_json(path)
    raw: dict[str, float] = {}
    if not isinstance(data, list):
        return raw
    for row in data:
        if not isinstance(row, dict):
            continue
        img = normalize_image_id(row.get("image_id", row.get("image", "")))
        try:
            raw[img] = float(row.get("srs_gain", 0.0))
        except (TypeError, ValueError):
            raw[img] = 0.0

    if not raw:
        return raw

    min_v = min(raw.values())
    max_v = max(raw.values())
    if max_v - min_v < 1e-9:
        return {k: 0.5 for k in raw}
    return {k: (v - min_v) / (max_v - min_v) for k, v in raw.items()}


def merge_missing(primary: dict[str, Any], fallback: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """
    Fill keys missing in `primary` with values from `fallback`.
    Returns merged map and number of filled keys.
    """
    merged = dict(primary)
    filled = 0
    for k, v in fallback.items():
        if k not in merged:
            merged[k] = v
            filled += 1
    return merged, filled


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run multi-agent validation scoring.")
    p.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "config.yaml"),
        help="Path to config.yaml",
    )
    p.add_argument("--base-dir", default=None, help="Override project base dir")
    p.add_argument("--dataset-root", default="final_dataset", help="Dataset root dir")
    p.add_argument("--output-dir", default="results/multi_agent", help="Output directory")
    p.add_argument(
        "--sweep-quantiles",
        default="",
        help="Comma-separated uncertainty quantiles to evaluate (e.g., 0.8,0.85,0.9).",
    )
    p.add_argument(
        "--sweep-output",
        default="",
        help="Optional output CSV path for uncertainty threshold sweep.",
    )
    return p


def build_manifest_fallbacks(manifest_df: pd.DataFrame) -> tuple[dict[str, int], dict[str, float], dict[str, float], dict[str, str]]:
    object_counts: dict[str, int] = {}
    scene_scores: dict[str, float] = {}
    srs_scores: dict[str, float] = {}
    kosmos_text: dict[str, str] = {}

    for _, row in manifest_df.iterrows():
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
        scene_scores[img] = max(0.0, min(1.0, conf))

        try:
            srs_scores[img] = float(row.get("srs_gain", 0.0) or 0.0)
        except (TypeError, ValueError):
            srs_scores[img] = 0.0

        kosmos_text[img] = str(row.get("kosmos_markdown", "") or "")

    # Normalize srs fallback to [0,1]
    if srs_scores:
        mn = min(srs_scores.values())
        mx = max(srs_scores.values())
        if mx - mn < 1e-9:
            srs_scores = {k: 0.5 for k in srs_scores}
        else:
            srs_scores = {k: (v - mn) / (mx - mn) for k, v in srs_scores.items()}

    return object_counts, scene_scores, srs_scores, kosmos_text


def compute_agreement_proxy(
    image_ids: list[str],
    object_counts: dict[str, int],
    scenes: dict[str, float],
    vqa: dict[str, float],
    srs: dict[str, float],
    kosmos: dict[str, str],
    object_count_saturation: int,
    document_char_saturation: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for img in image_ids:
        obj = min(object_counts.get(img, 0) / max(object_count_saturation, 1), 1.0)
        scn = max(0.0, min(1.0, scenes.get(img, 0.0)))
        vlm = max(0.0, min(1.0, vqa.get(img, 0.0)))
        rst = max(0.0, min(1.0, srs.get(img, 0.0)))
        doc = min(len(str(kosmos.get(img, ""))) / max(document_char_saturation, 1), 1.0)
        # proxy: how consistent object/scene/vqa/restoration/doc signals are
        vals = [obj, scn, vlm, rst, doc]
        out[img] = max(0.0, 1.0 - pstdev(vals))
    return out


def parse_quantiles(raw: str) -> list[float]:
    if not raw.strip():
        return []
    values: list[float] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            q = float(tok)
        except ValueError:
            continue
        if 0.0 < q < 1.0:
            values.append(q)
    return sorted(set(values))


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    project_base, mac = load_config(config_path)
    if args.base_dir:
        project_base = Path(args.base_dir).resolve()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = project_base / dataset_root
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_base / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_root / "metadata" / "manifest.csv"
    agreement_path = project_base / "results" / "agreement_scores_v8" / "agreement_v8_final_results.csv"
    object_path = project_base / "results" / "aligned_detections" / "student_iter_3_aligned.json"
    scene_path = project_base / "results" / "scene_labels" / "scene_labels_clip.json"
    vqa_path = dataset_root / "metadata" / "vqa_binary_classification.json"
    srs_path = dataset_root / "metadata" / "srs_scores.json"
    kosmos_path = project_base / "results" / "kosmos_grounding.jsonl"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    image_ids = [normalize_image_id(x) for x in manifest_df["image_id"].astype(str).tolist()]

    agreement = load_agreement_scores(agreement_path)
    agreement_source = "measured"
    measured_object_counts = load_object_counts(object_path)
    measured_scenes = load_scene_scores(scene_path)
    vqa = load_vqa_scores(vqa_path)
    measured_srs = load_srs_scores(srs_path)
    measured_kosmos = load_jsonl_map(kosmos_path, key="image", value="kosmos_output")

    fb_object, fb_scene, fb_srs, fb_kosmos = build_manifest_fallbacks(manifest_df)
    object_counts, object_fill = merge_missing(measured_object_counts, fb_object)
    scenes, scene_fill = merge_missing(measured_scenes, fb_scene)
    srs, restoration_fill = merge_missing(measured_srs, fb_srs)
    kosmos, document_fill = merge_missing(measured_kosmos, fb_kosmos)
    if not agreement:
        agreement = discover_agreement_scores(project_base)
        agreement_source = "measured" if agreement else agreement_source
    if not agreement:
        agreement = compute_agreement_proxy(
            image_ids=image_ids,
            object_counts=object_counts,
            scenes=scenes,
            vqa=vqa,
            srs=srs,
            kosmos=kosmos,
            object_count_saturation=mac.object_count_saturation,
            document_char_saturation=mac.document_char_saturation,
        )
        agreement_source = "proxy"

    object_source = "measured+manifest_fallback" if measured_object_counts else "manifest_fallback"
    scene_source = "measured+manifest_fallback" if measured_scenes else "manifest_fallback"
    vqa_source = "measured" if vqa else "proxy"
    restoration_source = "measured+manifest_fallback" if measured_srs else "manifest_fallback"
    document_source = "measured+manifest_fallback" if measured_kosmos else "manifest_fallback"

    row_candidates = []
    coverage = {
        "agreement": 0,
        "object": 0,
        "scene": 0,
        "vlm": 0,
        "document": 0,
        "restoration": 0,
    }

    for img in image_ids:
        has_agreement = img in agreement
        has_object = img in object_counts
        has_scene = img in scenes
        has_vlm = img in vqa
        has_document = img in kosmos
        has_restoration = img in srs

        obj_score = min(object_counts.get(img, 0) / max(mac.object_count_saturation, 1), 1.0)
        agr_score = max(0.0, min(1.0, agreement.get(img, 0.0)))
        scene_score = max(0.0, min(1.0, scenes.get(img, 0.0)))
        vlm_score = max(0.0, min(1.0, vqa.get(img, 0.0)))
        doc_len = len(str(kosmos.get(img, "")))
        doc_score = min(doc_len / max(mac.document_char_saturation, 1), 1.0)
        rest_score = max(0.0, min(1.0, srs.get(img, 0.0)))

        if has_agreement:
            coverage["agreement"] += 1
        if has_object:
            coverage["object"] += 1
        if has_scene:
            coverage["scene"] += 1
        if has_vlm:
            coverage["vlm"] += 1
        if has_document:
            coverage["document"] += 1
        if has_restoration:
            coverage["restoration"] += 1

        agent_scores = {
            "object_agent_score": obj_score,
            "agreement_agent_score": agr_score,
            "scene_agent_score": scene_score,
            "vlm_agent_score": vlm_score,
            "document_agent_score": doc_score,
            "restoration_agent_score": rest_score,
        }
        active_flags = {
            "object": has_object,
            "agreement": has_agreement,
            "scene": has_scene,
            "vlm": has_vlm,
            "document": has_document,
            "restoration": has_restoration,
        }
        active_count = sum(int(v) for v in active_flags.values())

        # Dynamic weighting based on SRS (Restoration Score)
        # If rest_score < 0.4, penalize object detection and boost VLM
        dyn_object_weight = mac.object_weight
        dyn_vlm_weight = mac.vlm_weight
        if rest_score < 0.4:
            penalty = 0.5 * (0.4 - rest_score) / 0.4 # up to 50% penalty
            dyn_object_weight *= (1.0 - penalty)
            dyn_vlm_weight *= (1.0 + penalty)

        validation_weighted_sum = (
            obj_score * dyn_object_weight
            + agr_score * mac.agreement_weight
            + scene_score * mac.scene_weight
            + vlm_score * dyn_vlm_weight
        )
        validation_weight_total = (
            dyn_object_weight
            + mac.agreement_weight
            + mac.scene_weight
            + dyn_vlm_weight
        )
        realism = validation_weighted_sum / validation_weight_total if validation_weight_total > 0 else 0.0

        uncertainty = pstdev(list(agent_scores.values())) if len(agent_scores) > 1 else 0.0
        row_candidates.append(
            {
                "image_id": img,
                **agent_scores,
                "agreement_source": agreement_source,
                "object_source": object_source,
                "scene_source": scene_source,
                "vlm_source": vqa_source,
                "document_source": document_source,
                "restoration_source": restoration_source,
                "metric_source": "mixed" if "proxy" in {agreement_source, object_source, scene_source, vqa_source, document_source, restoration_source} else "measured",
                "overall_realism_score": round(realism, 6),
                "uncertainty_score": round(uncertainty, 6),
                "active_agents": active_count,
                "active_object_agent": int(has_object),
                "active_agreement_agent": int(has_agreement),
                "active_scene_agent": int(has_scene),
                "active_vlm_agent": int(has_vlm),
                "active_document_agent": int(has_document),
                "active_restoration_agent": int(has_restoration),
            }
        )

    effective_uncertainty_threshold = mac.uncertainty_threshold
    if row_candidates and mac.uncertainty_threshold_quantile is not None:
        q = mac.uncertainty_threshold_quantile
        if 0.0 < q < 1.0:
            unc_series = pd.Series([r["uncertainty_score"] for r in row_candidates], dtype=float)
            effective_uncertainty_threshold = float(unc_series.quantile(q))

    rows = []
    for row in row_candidates:
        reason_min_agents = row["active_agents"] < mac.min_agents_required
        reason_uncertainty = row["uncertainty_score"] >= effective_uncertainty_threshold
        reason_low_realism = row["overall_realism_score"] < mac.realism_threshold
        needs_review = reason_min_agents or reason_uncertainty or reason_low_realism
        row["needs_hitl_review"] = int(needs_review)
        row["hitl_reason_min_agents"] = int(reason_min_agents)
        row["hitl_reason_uncertainty"] = int(reason_uncertainty)
        row["hitl_reason_low_realism"] = int(reason_low_realism)
        rows.append(row)

    out_csv = output_dir / "multi_agent_validation_scores.csv"
    out_json = output_dir / "multi_agent_validation_summary.json"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    rows_df = pd.DataFrame(rows)
    summary = {
        "num_images": len(rows),
        "coverage_counts": coverage,
        "coverage_ratio": {k: (v / len(rows) if rows else 0.0) for k, v in coverage.items()},
        "mean_realism_score": float(rows_df["overall_realism_score"].mean()) if rows else 0.0,
        "mean_uncertainty_score": float(rows_df["uncertainty_score"].mean()) if rows else 0.0,
        "hitl_review_ratio": float(rows_df["needs_hitl_review"].mean()) if rows else 0.0,
        "hitl_reason_ratio": {
            "min_agents": float(rows_df["hitl_reason_min_agents"].mean()) if rows else 0.0,
            "uncertainty": float(rows_df["hitl_reason_uncertainty"].mean()) if rows else 0.0,
            "low_realism": float(rows_df["hitl_reason_low_realism"].mean()) if rows else 0.0,
        },
        "sources": {
            "agreement_source": agreement_source,
            "object_source": object_source,
            "scene_source": scene_source,
            "vlm_source": vqa_source,
            "document_source": document_source,
            "restoration_source": restoration_source,
            "metric_source": "mixed" if "proxy" in {agreement_source, object_source, scene_source, vqa_source, document_source, restoration_source} else "measured",
        },
        "effective_uncertainty_threshold": effective_uncertainty_threshold,
        "fallback_fill_counts": {
            "object": object_fill,
            "scene": scene_fill,
            "document": document_fill,
            "restoration": restoration_fill,
        },
        "config": mac.__dict__,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    sweep_quantiles = parse_quantiles(args.sweep_quantiles)
    if sweep_quantiles:
        sweep_rows: list[dict[str, float]] = []
        for q in sweep_quantiles:
            threshold = float(rows_df["uncertainty_score"].quantile(q))
            reason_min_agents = rows_df["active_agents"] < mac.min_agents_required
            reason_uncertainty = rows_df["uncertainty_score"] >= threshold
            reason_low_realism = rows_df["overall_realism_score"] < mac.realism_threshold
            needs_review = reason_min_agents | reason_uncertainty | reason_low_realism
            sweep_rows.append(
                {
                    "uncertainty_quantile": q,
                    "effective_uncertainty_threshold": threshold,
                    "hitl_review_ratio": float(needs_review.mean()),
                    "hitl_reason_min_agents_ratio": float(reason_min_agents.mean()),
                    "hitl_reason_uncertainty_ratio": float(reason_uncertainty.mean()),
                    "hitl_reason_low_realism_ratio": float(reason_low_realism.mean()),
                }
            )
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_out = Path(args.sweep_output) if args.sweep_output else (output_dir / "uncertainty_threshold_sweep.csv")
        if not sweep_out.is_absolute():
            sweep_out = project_base / sweep_out
        sweep_out.parent.mkdir(parents=True, exist_ok=True)
        sweep_df.to_csv(sweep_out, index=False)
        print(f"✅ Wrote: {sweep_out}")

    print(f"✅ Wrote: {out_csv}")
    print(f"✅ Wrote: {out_json}")
    print(f"📊 Mean realism score: {summary['mean_realism_score']:.4f}")
    print(f"🧪 HITL review ratio: {summary['hitl_review_ratio']:.4f}")


if __name__ == "__main__":
    main()
