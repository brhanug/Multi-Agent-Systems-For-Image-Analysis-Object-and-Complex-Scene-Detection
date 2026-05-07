"""
Shared utility helpers for the multi-agent pipeline scripts.

Previously, normalize_image_id, safe_load_json, load_jsonl_map, and several
loader functions were copy-pasted verbatim across at least 5 scripts
(run_agent_comparison.py, run_multi_agent_validation.py, run_error_analysis.py,
etc.).  Centralising them here removes the duplication and makes bug-fixes
apply everywhere at once.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID normalisation
# ---------------------------------------------------------------------------

def normalize_image_id(raw: str) -> str:
    """Strip directory and extension from an image path/id, returning the bare stem."""
    return Path(str(raw)).stem


# ---------------------------------------------------------------------------
# Generic I/O helpers
# ---------------------------------------------------------------------------

def safe_load_json(path: Path) -> Any:
    """Load a JSON file; return None (with a warning) if it does not exist or is malformed."""
    if not path.exists():
        logger.debug("JSON file not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        logger.warning("Malformed JSON at %s: %s", path, exc)
        return None


def load_jsonl_map(path: Path, key: str, value: str) -> dict[str, Any]:
    """
    Parse a .jsonl file and return a dict mapping ``obj[key]`` to ``obj[value]``.
    Lines that cannot be decoded are skipped with a warning.
    """
    out: dict[str, Any] = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                k = normalize_image_id(obj.get(key, ""))
                out[k] = obj.get(value, "")
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line %d in %s", lineno, path)
    return out


# ---------------------------------------------------------------------------
# Domain-specific loaders (shared across pipeline scripts)
# ---------------------------------------------------------------------------

def load_object_counts(path: Path) -> dict[str, int]:
    """
    Load a JSON file of the form ``{image_id: [detection, ...]}`` and return
    a dict mapping normalised image IDs to detection counts.
    """
    data = safe_load_json(path)
    out: dict[str, int] = {}
    if not isinstance(data, dict):
        return out
    for k, dets in data.items():
        out[normalize_image_id(k)] = len(dets) if isinstance(dets, list) else 0
    return out


def load_agreement_scores(path: Path) -> dict[str, float]:
    """
    Load an agreement CSV with columns ``image_id`` and ``final_score``.
    Returns an empty dict if the file is absent or columns are missing.
    """
    out: dict[str, float] = {}
    if not path.exists():
        return out
    df = pd.read_csv(path)
    if "image_id" not in df.columns or "final_score" not in df.columns:
        logger.warning("Expected columns 'image_id' and 'final_score' in %s", path)
        return out
    for _, row in df.iterrows():
        try:
            out[normalize_image_id(row["image_id"])] = float(row["final_score"])
        except (TypeError, ValueError):
            pass
    return out


def discover_agreement_scores(project_base: Path) -> dict[str, float]:
    """Try several candidate paths for agreement score CSVs and return the first hit."""
    candidates = [
        project_base / "results" / "agreement_scores_v8" / "agreement_v8_final_results.csv",
        project_base / "results" / "agreement_scores_final" / "agreement_v8_final_results.csv",
        project_base / "results" / "agreement_scores_final" / "agreement_final_results.csv",
    ]
    for c in candidates:
        scores = load_agreement_scores(c)
        if scores:
            logger.info("Loaded agreement scores from %s", c)
            return scores
    return {}


def load_scene_scores(path: Path) -> dict[str, float]:
    """
    Load scene-classification JSON.  Supports two formats:
      - {image_id: [[label, score], ...]}
      - {image_id: [{"label": "...", "score": 0.9}, ...]}
    Returns the confidence of the top-ranked scene per image, clamped to [0, 1].
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
                    pass
            elif isinstance(top, dict):
                try:
                    score = float(top.get("score", 0.0))
                except (TypeError, ValueError):
                    pass
        out[img] = max(0.0, min(1.0, score))
    return out


def _to_binary(v: Any) -> float | None:
    """Convert a yes/no or 0/1 value to a float in {0.0, 1.0}, or None if not parseable."""
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


def load_vqa_scores(path: Path) -> dict[str, float]:
    """
    Load VQA binary-classification results (list or dict form) and return the
    mean binary answer per image, clamped to [0, 1].
    """
    data = safe_load_json(path)
    out: dict[str, float] = {}
    if data is None:
        return out

    def _mean_binary(values: list[Any]) -> float:
        nums = [b for v in values if (b := _to_binary(v)) is not None]
        return max(0.0, min(1.0, sum(nums) / len(nums))) if nums else 0.0

    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            img = normalize_image_id(row.get("image_id", row.get("image", "")))
            vals = [v for k, v in row.items() if k not in {"image_id", "image"}]
            out[img] = _mean_binary(vals)
    elif isinstance(data, dict):
        for k, payload in data.items():
            img = normalize_image_id(k)
            if isinstance(payload, dict):
                out[img] = _mean_binary(list(payload.values()))
            else:
                out[img] = _mean_binary([payload])
    return out


def load_srs_scores(path: Path) -> dict[str, float]:
    """
    Load Semantic Restoration Score list JSON and return min-max normalised
    ``srs_gain`` values per image.
    """
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
    mn, mx = min(raw.values()), max(raw.values())
    if mx - mn < 1e-9:
        return {k: 0.5 for k in raw}
    return {k: (v - mn) / (mx - mn) for k, v in raw.items()}


def merge_missing(primary: dict[str, Any], fallback: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """
    Fill keys absent from *primary* with values from *fallback*.
    Returns (merged_dict, number_of_filled_keys).
    """
    merged = dict(primary)
    filled = sum(1 for k, v in fallback.items() if k not in merged and not merged.setdefault(k, v) is None)
    # Simpler correct version:
    merged = dict(primary)
    filled = 0
    for k, v in fallback.items():
        if k not in merged:
            merged[k] = v
            filled += 1
    return merged, filled
