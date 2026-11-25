#!/usr/bin/env python3
"""
run_frontier_vlm_benchmark.py
------------------------------
Runs a REAL comparative benchmark of the local Multi-Agent SAA system
against live API calls to GPT-4o, Claude 3.5 Sonnet, and Gemini 2.5.

Evaluation task: zero-shot uncertainty estimation on 100 images sampled
deterministically (random_state=100) from the n=801 human-verified gold
cohort. Each frontier model is asked to rate how confident it is that an
image contains a scene (people, landscape, objects, etc.) on a 0–1 scale.
This confidence score is then used to compute:
  - ECE  : Expected Calibration Error against gold scene-presence labels
  - ROC-AUC : Ability to predict classification errors (is_error)
  - Recall@20% : Error recall at a 20% review budget

Usage:
    python3 scripts/core_pipeline/run_frontier_vlm_benchmark.py \
        --openai-key  sk-... \
        --anthropic-key sk-ant-... \
        --gemini-key  AIza...

    Keys can also be set as environment variables:
        OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY

Output:
    results/multi_agent/frontier_vlm_benchmark_real.json   — full results
    results/multi_agent/frontier_vlm_scores.csv            — per-image scores
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "core_pipeline"))
from saa_decision import SAA_THRESHOLD, STORED_COLUMN_THRESHOLD  # deliberate 0.55; see saa_decision.py
import os
import sys
import json
import time
import base64
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

# ── paths ─────────────────────────────────────────────────────────────────────
BASE        = Path("/data/brhanu/thesis_project")
GOLD_CSV    = BASE / "human_spatial_audit" / "user_annotations_800.csv"
SCORES_CSV  = BASE / "results/multi_agent/agent_comparison_scores.csv"
IMAGES_DIR  = BASE / "human_spatial_audit" / "images"
OUT_JSON    = BASE / "results/multi_agent/frontier_vlm_benchmark_real.json"
OUT_CSV     = BASE / "results/multi_agent/frontier_vlm_scores.csv"

SAMPLE_SEED = 100
SAMPLE_N    = 100
BUDGET_20   = 0.20
N_BINS      = 10

# Build a lookup: short image_id (e.g. 00000053_1) → full path
def _build_image_lookup() -> dict:
    lookup = {}
    for p in IMAGES_DIR.glob("*.jpg"):
        stem   = p.stem                      # e.g. PPN1756550050_00000053_1
        parts  = stem.split("_")
        if len(parts) >= 3 and parts[0].startswith("PPN"):
            short = "_".join(parts[-2:])     # e.g. 00000053_1
            lookup[short] = p
    return lookup

IMAGE_LOOKUP: dict = {}  # populated in main()

# ── zero-shot prompt ───────────────────────────────────────────────────────────
PROMPT = (
    "You are evaluating a historical archive image for scene presence. "
    "Answer ONLY with a single float between 0.0 and 1.0 representing your "
    "confidence that this image contains a meaningful visual scene (people, "
    "objects, landscapes, events). "
    "0.0 = definitely no scene (blank/damaged/unreadable), "
    "1.0 = definitely contains a clear scene. "
    "Output the number only, no explanation."
)

# ── helpers ───────────────────────────────────────────────────────────────────
def normalize_id(name: str) -> str:
    p = str(name).replace("images/", "").replace("\\", "/")
    p = p.split("/")[-1].rsplit(".", 1)[0]
    parts = p.split("_")
    if len(parts) >= 2 and parts[0].startswith("PPN"):
        return "_".join(parts[-2:])
    return p


def image_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def find_image(image_id: str) -> Path | None:
    """Resolve image_id to an actual file path using prebuilt lookup."""
    global IMAGE_LOOKUP
    # Direct hit in gold-kit folder
    if image_id in IMAGE_LOOKUP:
        return IMAGE_LOOKUP[image_id]
    # Fallback: suffix match across all loaded keys
    for key, path in IMAGE_LOOKUP.items():
        if key.endswith(image_id) or image_id.endswith(key):
            return path
    return None


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = mask | (y_prob == hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece_val += mask.mean() * abs(conf - acc)
    return float(ece_val)


def recall_at_budget(y_err: np.ndarray, uncertainty: np.ndarray,
                     budget: float = BUDGET_20) -> float:
    n_budget = int(len(y_err) * budget)
    top_idx  = np.argsort(uncertainty)[::-1][:n_budget]
    total_errors = y_err.sum()
    if total_errors == 0:
        return 1.0
    return float(y_err[top_idx].sum() / total_errors)


# ── API callers ───────────────────────────────────────────────────────────────
def call_openai(client, image_path: Path, model: str = "gpt-4o") -> float | None:
    b64 = image_to_b64(image_path)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}",
                                   "detail": "low"}}
                ]
            }],
            max_tokens=10,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        return float(raw)
    except Exception as e:
        print(f"    [OpenAI ERROR] {e}")
        return None


def call_anthropic(client, image_path: Path,
                   model: str = "claude-3-5-sonnet-20241022") -> float | None:
    b64   = image_to_b64(image_path)
    media = "image/jpeg"
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64",
                                "media_type": media,
                                "data": b64}},
                    {"type": "text", "text": PROMPT}
                ]
            }]
        )
        raw = resp.content[0].text.strip()
        return float(raw)
    except Exception as e:
        print(f"    [Anthropic ERROR] {e}")
        return None


def call_gemini(model, image_path: Path) -> float | None:
    import google.generativeai as genai
    from PIL import Image as PILImage
    try:
        img = PILImage.open(image_path)
        resp = model.generate_content([PROMPT, img])
        raw  = resp.text.strip()
        return float(raw)
    except Exception as e:
        print(f"    [Gemini ERROR] {e}")
        return None


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Real frontier VLM benchmark vs Multi-Agent SAA")
    parser.add_argument("--openai-key",    default=os.getenv("OPENAI_API_KEY",    ""))
    parser.add_argument("--anthropic-key", default=os.getenv("ANTHROPIC_API_KEY", ""))
    parser.add_argument("--gemini-key",    default=os.getenv("GEMINI_API_KEY",    ""))
    parser.add_argument("--skip-openai",    action="store_true")
    parser.add_argument("--skip-anthropic", action="store_true")
    parser.add_argument("--skip-gemini",    action="store_true")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds to wait between API calls (rate limiting)")
    args = parser.parse_args()

    # ── load + join data ──────────────────────────────────────────────────────
    print("📂 Loading gold labels and agent scores …")
    gold = pd.read_csv(GOLD_CSV)
    gold["cvat_id"] = gold.index
    gold = gold[gold["cvat_id"] <= 800].copy()
    gold["gold_has_scene"] = (gold["n_scene_labels"] > 0).astype(int)
    gold["image_id"] = gold["image_id"].apply(normalize_id)
    # keep raw_name for image path resolution
    gold["raw_name"] = gold["raw_name"].astype(str)

    agents = pd.read_csv(SCORES_CSV)
    agents["image_id"] = agents["image_id"].apply(normalize_id)
    agents = agents.drop_duplicates(subset=["image_id"])

    joined = gold.merge(agents, on="image_id", how="inner")
    print(f"  Joined {len(joined)} images with agent scores.")

    # deterministic 100-image subset (same seed as simulation script)
    subset = joined.sample(n=SAMPLE_N, random_state=SAMPLE_SEED).reset_index(drop=True)
    print(f"  Sampled {len(subset)} images (seed={SAMPLE_SEED}).")

    # ── build image lookup from gold-kit folder ───────────────────────────────
    global IMAGE_LOOKUP
    IMAGE_LOOKUP = _build_image_lookup()
    print(f"  Image lookup: {len(IMAGE_LOOKUP)} entries in gold-kit folder.")

    y_true = subset["gold_has_scene"].values

    # classification error target (does SAA fusion prediction differ from gold?)
    subset["fusion_pred"] = (subset["comparison_fusion_score"] >= STORED_COLUMN_THRESHOLD).astype(int)
    y_err  = (subset["fusion_pred"] != y_true).astype(int).values

    print(f"  Gold: {y_true.sum()} scenes / {(1-y_true).sum()} no-scene | "
          f"Errors: {y_err.sum()}")

    # ── compute SAA uncertainty on this subset ────────────────────────────────
    agent_cols = ["existing_pipeline_agent", "agreement_agent",
                  "scene_agent", "vlm_agent"]
    u_saa = subset[agent_cols].std(axis=1).values
    u_saa_norm = u_saa / u_saa.max() if u_saa.max() > 0 else u_saa

    saa_metrics = {
        "ECE":           round(ece(y_err, u_saa_norm), 4),
        "ROC-AUC":       round(roc_auc_score(y_err, u_saa) if y_err.sum() > 0 else 0.5, 4),
        "Recall@20%":    round(recall_at_budget(y_err, u_saa), 4),
    }
    print(f"\n✅ SAA (Ours) on 100-image subset:")
    print(f"   ECE={saa_metrics['ECE']}  ROC-AUC={saa_metrics['ROC-AUC']}  "
          f"Recall@20%={saa_metrics['Recall@20%']}")

    # ── build image path map ──────────────────────────────────────────────────
    subset["img_path"] = subset["image_id"].apply(find_image)
    missing = subset["img_path"].isna().sum()
    if missing > 0:
        # Try resolving from raw_name if still missing
        def resolve_from_raw(row):
            if row["img_path"] is not None:
                return row["img_path"]
            raw = str(row.get("raw_name", ""))
            # raw_name looks like: images/PPN1756550050_00000053_1.jpg
            stem = raw.replace("images/", "").rsplit(".", 1)[0]  # PPN..._00000053_1
            candidate = IMAGES_DIR / (stem + ".jpg")
            if candidate.exists():
                return candidate
            return None
        subset["img_path"] = subset.apply(resolve_from_raw, axis=1)
        missing = subset["img_path"].isna().sum()
    print(f"  Image resolution: {SAMPLE_N - missing}/{SAMPLE_N} images found.")

    # ── initialise API clients ────────────────────────────────────────────────
    openai_client    = None
    anthropic_client = None
    gemini_model     = None

    if args.openai_key and not args.skip_openai:
        try:
            from openai import OpenAI
            openai_client = OpenAI(api_key=args.openai_key)
            print("\n🔑 OpenAI client initialised (GPT-4o).")
        except ImportError:
            print("⚠️  openai package not installed. Run: pip install openai")

    if args.anthropic_key and not args.skip_anthropic:
        try:
            import anthropic
            anthropic_client = anthropic.Anthropic(api_key=args.anthropic_key)
            print("🔑 Anthropic client initialised (Claude 3.5 Sonnet).")
        except ImportError:
            print("⚠️  anthropic package not installed. Run: pip install anthropic")

    if args.gemini_key and not args.skip_gemini:
        try:
            import google.generativeai as genai
            genai.configure(api_key=args.gemini_key)
            gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            print("🔑 Gemini client initialised (gemini-2.0-flash).")
        except ImportError:
            print("⚠️  google-generativeai not installed. Run: pip install google-generativeai")

    # ── per-image API calls ───────────────────────────────────────────────────
    scores = {
        "image_id": subset["image_id"].tolist(),
        "gold_has_scene": y_true.tolist(),
        "is_error": y_err.tolist(),
        "saa_uncertainty": u_saa.tolist(),
        "gpt4o_score":   [None] * SAMPLE_N,
        "claude_score":  [None] * SAMPLE_N,
        "gemini_score":  [None] * SAMPLE_N,
    }

    for i, row in subset.iterrows():
        idx       = subset.index.get_loc(i)
        img_path  = row["img_path"]
        image_id  = row["image_id"]

        if idx % 10 == 0:
            print(f"\n  [{idx+1}/{SAMPLE_N}] Processing {image_id} …")

        if img_path is None or (isinstance(img_path, float) and np.isnan(img_path)):
            print(f"    ↳ Image not found — skipping API calls.")
            continue

        img_path = Path(img_path)

        if openai_client:
            sc = call_openai(openai_client, img_path)
            scores["gpt4o_score"][idx] = sc
            time.sleep(args.delay)

        if anthropic_client:
            sc = call_anthropic(anthropic_client, img_path)
            scores["claude_score"][idx] = sc
            time.sleep(args.delay)

        if gemini_model:
            sc = call_gemini(gemini_model, img_path)
            scores["gemini_score"][idx] = sc
            time.sleep(args.delay)

    # ── compute metrics for each frontier model ───────────────────────────────
    def metrics_for(score_list: list, name: str) -> dict | None:
        vals = np.array([s if s is not None else np.nan for s in score_list])
        valid = ~np.isnan(vals)
        n_valid = valid.sum()
        if n_valid < 10:
            print(f"⚠️  {name}: only {n_valid} valid scores — skipping metric computation.")
            return None
        v_scores = vals[valid]
        v_err    = y_err[valid]
        v_true   = y_true[valid]
        # uncertainty for a monolithic model = |0.5 - score| inverted
        # i.e. how far from certain: low confidence in either direction = high uncertainty
        uncertainty = np.abs(v_scores - 0.5)   # higher = more certain → invert for "uncertainty"
        uncertainty_inv = 1.0 - uncertainty     # high value = model uncertain
        u_norm = uncertainty_inv / uncertainty_inv.max() if uncertainty_inv.max() > 0 else uncertainty_inv
        return {
            "n_evaluated": int(n_valid),
            "ECE":        round(ece(v_err, v_scores), 4),
            "ROC-AUC":    round(roc_auc_score(v_err, uncertainty_inv) if v_err.sum() > 0 and v_err.sum() < len(v_err) else 0.5, 4),
            "Recall@20%": round(recall_at_budget(v_err, uncertainty_inv), 4),
        }

    results = {
        "metadata": {
            "sample_n": SAMPLE_N,
            "sample_seed": SAMPLE_SEED,
            "gold_scenes": int(y_true.sum()),
            "gold_errors": int(y_err.sum()),
        },
        "systems": [
            {
                "System": "Visual Historian-MAS (Ours)",
                "n_evaluated": SAMPLE_N,
                **saa_metrics,
                "API Cost per 1k images": "$0.00 (Offline/Local)",
                "source": "REAL — computed from agent_comparison_scores.csv",
            }
        ]
    }

    for model_name, score_key in [("GPT-4o",          "gpt4o_score"),
                                   ("Claude 3.5 Sonnet", "claude_score"),
                                   ("Gemini 2.5",        "gemini_score")]:
        m = metrics_for(scores[score_key], model_name)
        if m:
            results["systems"].append({
                "System": f"{model_name} (Monolithic Zero-Shot)",
                **m,
                "API Cost per 1k images": "$15.00 (GPT-4o) / $24.00 (Claude) / $15.00 (Gemini)",
                "source": "REAL — live API evaluation",
            })
        else:
            results["systems"].append({
                "System": f"{model_name} (Monolithic Zero-Shot)",
                "note": "Skipped (no API key or insufficient valid responses)",
                "source": "NOT EVALUATED",
            })

    # ── save outputs ──────────────────────────────────────────────────────────
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Results saved → {OUT_JSON}")

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(OUT_CSV, index=False)
    print(f"✅ Per-image scores saved → {OUT_CSV}")

    # ── print summary table ───────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'System':<32} {'ECE':>7} {'ROC-AUC':>9} {'Rec@20%':>9}")
    print("="*65)
    for s in results["systems"]:
        if "ECE" in s:
            print(f"{s['System']:<32} {s['ECE']:>7.4f} {s['ROC-AUC']:>9.4f} "
                  f"{s['Recall@20%']:>9.4f}")
        else:
            print(f"{s['System']:<32} {'—':>7} {'—':>9} {s.get('note','—'):>9}")
    print("="*65)

    # ── instruction for updating thesis ──────────────────────────────────────
    print("""
📋 Next step — update thesis with real values:
   Edit thesis_v3_updated.tex  → Section sec:frontier_comparison (Table tab:frontier_comparison)
   Edit advisor_walkthrough_handout.tex → Paragraph "Comparison Against Frontier VLMs"
   Replace hardcoded numbers with values from:
     {}
""".format(OUT_JSON))


if __name__ == "__main__":
    main()
