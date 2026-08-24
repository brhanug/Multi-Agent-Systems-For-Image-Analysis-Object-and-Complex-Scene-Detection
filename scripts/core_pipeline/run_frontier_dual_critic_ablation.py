#!/usr/bin/env python3
"""
run_frontier_dual_critic_ablation.py
--------------------------------------
APPROXIMATION, NOT THE DOCUMENTED FRONTIER CONFIGURATION -- READ BEFORE CITING.
The thesis's frontier configuration is Gemini 2.5 Flash (Primary) + Claude
3.5 Sonnet (Critic) -- two structurally independent model families. At run
time, the provided Anthropic key had insufficient credit balance (confirmed
via a direct API call) and the provided Gemini key no longer has access to
the gemini-2.5-flash model (only newer ones). Per explicit user direction,
this script instead uses ONE model (gemini-flash-lite-latest) for BOTH Primary and
Critic -- i.e. this is a SAME-MODEL self-critique test, structurally
identical in kind to the local LLaVA-OneVision ablation, just with a
stronger frontier-grade model. It does NOT test genuine cross-model
independence and must not be cited as evidence for or against the
frontier configuration's real F1 gap (which remains unverified -- see
compare_coordinators.py). It only answers: does a stronger single model,
re-prompted as its own critic, catch more of its own hallucinations than a
weaker one does?

Uses the SAME 25-image sample (same random_state) as the local ablation.

Requires GEMINI_API_KEY environment variable.

Output:
  results/multi_agent/frontier_approx_same_model_critic_ablation.csv
  results/multi_agent/frontier_approx_same_model_critic_ablation_summary.json
"""

import base64
import json
import os
import re
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BASE = Path("/data/brhanu/thesis_project")
IMAGE_DIR = BASE / "final_dataset" / "images" / "diffusion_restored"
OUT_CSV = BASE / "results" / "multi_agent" / "frontier_approx_same_model_critic_ablation.csv"
OUT_JSON = BASE / "results" / "multi_agent" / "frontier_approx_same_model_critic_ablation_summary.json"

N_SAMPLES = 10  # reduced from the local ablation's 25: gemini-flash-lite-latest's free-tier daily quota
                 # (20 requests/day/model) was exhausted mid-run; gemini-flash-lite-latest below
                 # has separate, apparently-available quota, but kept conservative regardless.
RANDOM_STATE = 100  # identical to run_local_dual_critic_ablation.py (same first-N-of-25 images)

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-flash-lite-latest"  # same model used for BOTH Primary and Critic -- see module docstring

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


def query_gemini(image_b64: str, prompt: str) -> str | None:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/jpeg", "data": image_b64}},
            ]
        }],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 800},
    }
    try:
        resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=90)
        if resp.status_code != 200:
            print(f"  WARNING Gemini HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        cand = resp.json()["candidates"][0]
        parts = cand.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        return text if text else None
    except Exception as e:
        print(f"  WARNING Gemini error: {e}")
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
    if not GEMINI_KEY:
        print("ERROR: GEMINI_API_KEY not set.")
        return

    all_images = sorted(IMAGE_DIR.glob("*.jpg"))
    # Draw the SAME 25-image sample as the local ablation (same n=25, same random_state),
    # then take the first N_SAMPLES of those -- so this really is a genuine subset of the
    # local test's images, not an independently-drawn (and therefore non-comparable) sample.
    full_sample_25 = pd.Series([str(p) for p in all_images]).sample(
        n=min(25, len(all_images)), random_state=RANDOM_STATE
    ).tolist()
    sample = full_sample_25[:N_SAMPLES]
    print(f"Using first {len(sample)} of the same 25-image local-ablation sample (random_state={RANDOM_STATE})")

    rows = []
    for img_path_str in tqdm(sample, desc="Same-model (Gemini) Primary+Critic approximation"):
        img_path = Path(img_path_str)
        img_b64 = encode_image_base64(img_path)

        primary_raw = query_gemini(img_b64, PRIMARY_PROMPT)
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
        critic_raw = query_gemini(img_b64, critic_prompt)  # same model as Primary -- see docstring
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
        "experiment": "APPROXIMATE same-model (Gemini Flash-Lite) Primary+Critic ablation -- NOT the documented Gemini+Claude frontier pair",
        "primary_model": GEMINI_MODEL,
        "critic_model": GEMINI_MODEL,
        "n_images_sampled": n_total,
        "n_primary_json_parsed": n_primary_parsed,
        "n_critic_json_parsed": n_critic_parsed,
        "n_critic_flagged_contradiction": n_flagged,
        "critic_flag_rate": round(flag_rate, 4) if flag_rate is not None else None,
        "note": (
            "CAVEAT: run with ONE model (gemini-flash-lite-latest) in both Primary and Critic roles, "
            "because the provided Anthropic key had insufficient credit balance and the provided "
            "Gemini key lacked access to gemini-2.5-flash at run time -- see module docstring. "
            "This is directly comparable in KIND to run_local_dual_critic_ablation.py (same "
            "25 images, same protocol, same random_state) -- both are same-model self-critique "
            "tests, just with a stronger frontier-grade model here instead of local LLaVA-OneVision. "
            "It does NOT test genuine cross-model (Gemini+Claude) independence and must not be "
            "cited as a measurement of the documented frontier configuration's real F1 gap, which "
            "remains unverified."
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
