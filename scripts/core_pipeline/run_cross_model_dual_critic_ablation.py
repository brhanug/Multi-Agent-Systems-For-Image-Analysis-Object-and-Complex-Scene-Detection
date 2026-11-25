#!/usr/bin/env python3
"""
run_cross_model_dual_critic_ablation.py
------------------------------------------
Real, honest mechanical test of genuine CROSS-MODEL independence for the
local Tier 3 Dual-LLM Coordinator idea -- the test run_local_dual_critic_
ablation.py's own docstring flagged as still missing: "the genuine
cross-model independence test... is unrun, not falsified."

Background
----------
The original local ablation (run_local_dual_critic_ablation.py) re-prompted
the SAME LLaVA-OneVision instance into both a Primary and a Critic role and
found 0/25: the same weights just agree with themselves. That result alone
doesn't distinguish "self-critique doesn't work" from "this specific model
doesn't work" -- it needed a genuinely different model to be conclusive.

This script uses two real, locally-hosted vision models:
  - Qwen2.5-VL-72B (Alibaba Qwen lineage)
  - InternVL3.5-4B (Shanghai AI Lab / OpenGVLab lineage)

NOTE on what "disjoint" means here, disclosed rather than glossed over:
Llama-3.2-90B-Vision was the original, fully lineage-disjoint choice paired
against InternVL, but it could not actually be run: Ollama's current release
has a confirmed, currently-open upstream bug loading the "mllama"
architecture (github.com/ollama/ollama issue #16490/#16547, reproduced
independently by multiple users through at least Aug 2026) -- verified
directly on this machine on two separate Ollama builds (a pre-existing 0.30.9
binary and a freshly-downloaded official 0.33.2 release), both failing
identically. Qwen2.5-VL was substituted because it is confirmed working.
Its caveat: Qwen2.5-VL and Tier 1's own VLM (LLaVA-OneVision-Qwen2-7B) both
use a Qwen2 LLM backbone for language generation -- NOT fully disjoint at
that layer, though their vision encoders, vision-language connectors, and
training data/recipes differ. InternVL3.5 remains fully disjoint from both
(different LLM backbone, different vision encoder, different organization).

Second caveat disclosed up front: this is a large-vs-small pair (72B vs 4B),
not two equally-capable models. A result here answers "do two differently-
trained vision models, of whatever size/backbone overlap, ever independently
catch what the other says," not "would two frontier-scale, fully lineage-
disjoint models" -- the experiment this thesis still cannot run locally.

Protocol (identical prompts/sampling to run_local_dual_critic_ablation.py
for direct comparability against its 0/25 same-model result):
  1. Same 25 images, same random_state=100 sample from the same directory.
  2. Primary pass: Model A synthesizes a structured record from the image.
  3. Critic pass: Model B (a DIFFERENT model) is shown the image AND
     Model A's record, in an adversarial auditor stance, asked to find
     contradictions or say NONE.
  4. Run BOTH directions (A-primary/B-critic and B-primary/A-critic) for
     full coverage -- 50 critic judgments total, not 25.

Requires a running local Ollama server (localhost:11434) with
qwen2.5vl:72b and blaifa/InternVL3_5:4B pulled.

Output:
  results/multi_agent/cross_model_dual_critic_ablation.csv
  results/multi_agent/cross_model_dual_critic_ablation_summary.json
"""

import base64
import io
import json
import re
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

BASE = Path("/data/brhanu/thesis_project")
IMAGE_DIR = BASE / "final_dataset" / "images" / "diffusion_restored"
OUT_CSV = BASE / "results" / "multi_agent" / "cross_model_dual_critic_ablation_scale_matched.csv"
OUT_JSON = BASE / "results" / "multi_agent" / "cross_model_dual_critic_ablation_scale_matched_summary.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_A = "qwen2.5vl:7b"    # Alibaba Qwen lineage (shares LLM backbone w/ Tier 1's LLaVA-OneVision)
MODEL_B = "llava:7b"        # Original LLaVA: CLIP ViT + Vicuna/Llama-2 backbone -- fully disjoint lineage
# Scale-matched (7B vs 7B) specifically to isolate lineage/independence from raw capability,
# after the 72B-vs-4B run's spot-check showed the earlier result was confounded by a large
# capability gap (the 72B model was simply a better critic, not just a different one).
N_SAMPLES = 25
RANDOM_STATE = 100  # IDENTICAL to run_local_dual_critic_ablation.py -- same 25 images

# IDENTICAL prompts to run_local_dual_critic_ablation.py for direct comparability.
PRIMARY_PROMPT = (
    "You are a metadata synthesis agent for a historical image archive. "
    "Look at this image and output ONLY a JSON object with exactly these keys: "
    '"scene_type" (one short phrase), "entities_present" (a list of concrete objects, '
    'people, or animals you can see), "description" (one sentence). '
    "Be concrete and only report what you can actually see."
)

CRITIC_PROMPT_TEMPLATE = (
    "You are an adversarial auditor reviewing another agent's synthesized record of "
    "this same image, checking specifically for hallucinations -- entities or claims "
    "in the record that are NOT actually visible in the image. "
    "Here is the record to audit:\n{record}\n\n"
    "Output ONLY a JSON object with exactly these keys: "
    '"contradictions_found" (true or false), '
    '"details" (a list of specific hallucinated/incorrect items, or an empty list if none). '
    "Be strict: only flag something if you are confident it is not actually in the image."
)


def encode_image_base64(image_path: Path) -> str:
    # Resized to a max 1024px edge: full-resolution scans (some >1600px) reproducibly
    # triggered a degenerate '?????' output bug in qwen2.5vl's Ollama integration --
    # confirmed on multiple images, fixed by downsizing before encoding. Applied to
    # both models for consistency (InternVL was never observed to need it, but this
    # keeps the comparison on identical input).
    im = Image.open(image_path).convert("RGB")
    im.thumbnail((1024, 1024))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def query_ollama(model: str, image_b64: str, prompt: str) -> str | None:
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        if response.status_code != 200:
            print(f"  WARNING HTTP {response.status_code}: {response.text[:150]}")
            return None
        return response.json().get("response")
    except Exception as e:
        print(f"  WARNING Error ({model}): {e}")
        return None


def extract_json(text: str | None) -> dict | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def run_direction(sample: list[str], primary_model: str, critic_model: str, direction: str) -> list[dict]:
    rows = []
    desc = f"{direction}: {primary_model} -> {critic_model}"
    for img_path_str in tqdm(sample, desc=desc):
        img_path = Path(img_path_str)
        img_b64 = encode_image_base64(img_path)

        primary_raw = query_ollama(primary_model, img_b64, PRIMARY_PROMPT)
        primary_json = extract_json(primary_raw)

        if primary_json is None:
            rows.append({
                "direction": direction, "image": img_path.name,
                "primary_model": primary_model, "critic_model": critic_model,
                "primary_raw": primary_raw, "primary_parsed": False,
                "critic_raw": None, "critic_parsed": False,
                "contradictions_found": None, "details": None,
            })
            continue

        critic_prompt = CRITIC_PROMPT_TEMPLATE.format(record=json.dumps(primary_json))
        critic_raw = query_ollama(critic_model, img_b64, critic_prompt)
        critic_json = extract_json(critic_raw)

        rows.append({
            "direction": direction, "image": img_path.name,
            "primary_model": primary_model, "critic_model": critic_model,
            "primary_raw": primary_raw, "primary_parsed": True,
            "primary_scene_type": primary_json.get("scene_type"),
            "primary_entities": json.dumps(primary_json.get("entities_present")),
            "critic_raw": critic_raw, "critic_parsed": critic_json is not None,
            "contradictions_found": critic_json.get("contradictions_found") if critic_json else None,
            "details": json.dumps(critic_json.get("details")) if critic_json else None,
        })
    return rows


def main() -> None:
    all_images = sorted(IMAGE_DIR.glob("*.jpg"))
    print(f"Found {len(all_images)} candidate images in {IMAGE_DIR}")
    sample = pd.Series([str(p) for p in all_images]).sample(
        n=min(N_SAMPLES, len(all_images)), random_state=RANDOM_STATE
    ).tolist()
    print(f"Sampled {len(sample)} images (random_state={RANDOM_STATE}, IDENTICAL to the same-model test)")

    rows = []
    rows += run_direction(sample, MODEL_A, MODEL_B, "A_primary_B_critic")
    rows += run_direction(sample, MODEL_B, MODEL_A, "B_primary_A_critic")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    summary = {"experiment": "Cross-Model Dual-Critic Ablation (real inference, disjoint-lineage vision models)",
               "model_a": MODEL_A, "model_b": MODEL_B,
               "n_images_per_direction": len(sample), "n_directions": 2,
               "n_total_critic_judgments": len(df)}

    for direction in df["direction"].unique():
        sub = df[df["direction"] == direction]
        parsed = sub[sub["critic_parsed"] == True]
        n_flagged = int((parsed["contradictions_found"] == True).sum())
        n_parsed = len(parsed)
        summary[direction] = {
            "n_images": len(sub),
            "n_primary_parsed": int(sub["primary_parsed"].sum()),
            "n_critic_parsed": n_parsed,
            "n_flagged": n_flagged,
            "flag_rate": round(n_flagged / n_parsed, 4) if n_parsed else None,
        }

    all_parsed = df[df["critic_parsed"] == True]
    n_flagged_total = int((all_parsed["contradictions_found"] == True).sum())
    n_parsed_total = len(all_parsed)
    summary["overall_flag_rate"] = round(n_flagged_total / n_parsed_total, 4) if n_parsed_total else None
    summary["overall_n_flagged"] = n_flagged_total
    summary["overall_n_parsed"] = n_parsed_total
    summary["comparison"] = {
        "same_model_baseline_flag_rate (run_local_dual_critic_ablation.py, LLaVA-OneVision both roles)": "0/25 (0.0)",
        "note": "A materially higher flag rate here than the same-model 0/25 baseline would support the "
                "'disjoint weights catch what same weights miss' argument for Tier 3's frontier-only design. "
                "A similarly-near-zero rate would suggest the effect isn't really about model independence.",
    }
    summary["disclosed_limitation"] = (
        "Scale-matched follow-up to the 72B-vs-4B run (cross_model_dual_critic_ablation_summary.json), whose "
        "spot-check showed the larger model was simply a better critic, confounding capability with lineage. "
        "Model A (7B) and Model B (7B) are near-identical in parameter count, isolating lineage/independence "
        "as the remaining variable. Model A shares a Qwen2 LLM backbone with Tier 1's own LLaVA-OneVision "
        "(different vision encoder/training); Model B (original LLaVA, CLIP ViT + Vicuna/Llama-2) is fully "
        "disjoint from both. Both are meaningfully weaker than Tier 1's own 7B LLaVA-OneVision or the 72B/90B "
        "models tried earlier -- this trades capability for a cleaner isolation of the lineage variable."
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
