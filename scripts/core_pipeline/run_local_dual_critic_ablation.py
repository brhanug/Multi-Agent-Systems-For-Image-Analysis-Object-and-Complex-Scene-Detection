#!/usr/bin/env python3
"""
run_local_dual_critic_ablation.py
-----------------------------------
Real, honest mechanical test of whether the local Tier 3 Dual-LLM
Coordinator's Critic re-prompt step (same LLaVA-OneVision instance,
re-prompted into an adversarial auditor stance) actually changes anything
versus a single Primary-only pass.

Background: the only existing "local vs. frontier" comparison in this
codebase (compare_coordinators.py, Contradiction F1 = 0.812 vs. 0.965) was
found during a full-project audit to be a HARDCODED PLACEHOLDER printed
under a "no API keys -> documented historical data" fallback branch, not a
real measurement. No gold contradiction labels exist anywhere in this
project, so a real F1 comparison is not currently possible. This script
does NOT attempt to reproduce that F1 number. Instead it answers a
narrower, honestly-answerable question with real model inference:

    When the same model is re-prompted into a critic role, does it ever
    actually flag something different from what it just said as Primary,
    or does it just rubber-stamp itself every time?

Protocol (real inference only, no hardcoded/RNG placeholders):
  1. Primary pass: LLaVA-OneVision synthesizes a structured record
     (scene_type, entities_present, description) from the image alone.
  2. Critic pass: the SAME model instance is re-prompted, given the image
     AND its own Primary record, in an adversarial auditor stance, and
     asked to find contradictions or say NONE.
  3. Log the full transcript for every image (for manual spot-checking)
     and report the raw contradiction-flag rate.

Requires a running vLLM server on localhost:8000 hosting
llava-hf/llava-onevision-qwen2-7b-ov-hf.

Output:
  results/multi_agent/local_dual_critic_ablation.csv
  results/multi_agent/local_dual_critic_ablation_summary.json
"""

import base64
import json
import re
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BASE = Path("/data/brhanu/thesis_project")
IMAGE_DIR = BASE / "final_dataset" / "images" / "diffusion_restored"
OUT_CSV = BASE / "results" / "multi_agent" / "local_dual_critic_ablation.csv"
OUT_JSON = BASE / "results" / "multi_agent" / "local_dual_critic_ablation_summary.json"

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
N_SAMPLES = 25
RANDOM_STATE = 100  # matches exp26_hybrid_frontier_routing.py's sampling convention

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
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_llava(image_b64: str, prompt: str) -> str | None:
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }],
            "temperature": 0.0,
            "max_tokens": 400,
        }
        response = requests.post(API_URL, json=payload, timeout=120)
        if response.status_code != 200:
            print(f"  WARNING HTTP {response.status_code}: {response.text[:150]}")
            return None
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  WARNING Error: {e}")
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


def main() -> None:
    all_images = sorted(IMAGE_DIR.glob("*.jpg"))
    print(f"Found {len(all_images)} candidate images in {IMAGE_DIR}")
    sample = pd.Series([str(p) for p in all_images]).sample(
        n=min(N_SAMPLES, len(all_images)), random_state=RANDOM_STATE
    ).tolist()
    print(f"Sampled {len(sample)} images (random_state={RANDOM_STATE})")

    rows = []
    for img_path_str in tqdm(sample, desc="Local Primary+Critic ablation"):
        img_path = Path(img_path_str)
        img_b64 = encode_image_base64(img_path)

        primary_raw = query_llava(img_b64, PRIMARY_PROMPT)
        primary_json = extract_json(primary_raw)

        if primary_json is None:
            rows.append({
                "image": img_path.name,
                "primary_raw": primary_raw,
                "primary_parsed": False,
                "critic_raw": None,
                "critic_parsed": False,
                "contradictions_found": None,
                "details": None,
            })
            continue

        critic_prompt = CRITIC_PROMPT_TEMPLATE.format(record=json.dumps(primary_json))
        critic_raw = query_llava(img_b64, critic_prompt)
        critic_json = extract_json(critic_raw)

        rows.append({
            "image": img_path.name,
            "primary_raw": primary_raw,
            "primary_parsed": True,
            "primary_scene_type": primary_json.get("scene_type"),
            "primary_entities": json.dumps(primary_json.get("entities_present")),
            "critic_raw": critic_raw,
            "critic_parsed": critic_json is not None,
            "contradictions_found": critic_json.get("contradictions_found") if critic_json else None,
            "details": json.dumps(critic_json.get("details")) if critic_json else None,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    parsed = df[df["critic_parsed"] == True]
    n_total = len(df)
    n_primary_parsed = int(df["primary_parsed"].sum())
    n_critic_parsed = len(parsed)
    n_flagged = int((parsed["contradictions_found"] == True).sum())
    flag_rate = n_flagged / n_critic_parsed if n_critic_parsed > 0 else None

    summary = {
        "experiment": "Local Dual-Critic Mechanical Ablation (real inference, no gold labels available)",
        "model": MODEL_NAME,
        "n_images_sampled": n_total,
        "n_primary_json_parsed": n_primary_parsed,
        "n_critic_json_parsed": n_critic_parsed,
        "n_critic_flagged_contradiction": n_flagged,
        "critic_flag_rate": round(flag_rate, 4) if flag_rate is not None else None,
        "note": (
            "This measures how often the SAME model, re-prompted into a critic role, "
            "flags a contradiction in its own Primary output. It is NOT a substitute for "
            "the withdrawn Contradiction-Detection F1 comparison (no gold labels exist for "
            "that). A near-zero flag rate would suggest the local critic step rarely adds "
            "anything over accepting the Primary pass; a substantial, spot-checked-genuine "
            "flag rate would suggest it does surface real issues."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
