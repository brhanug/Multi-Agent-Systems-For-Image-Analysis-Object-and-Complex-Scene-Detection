#!/usr/bin/env python3
"""
run_europeana_real_pilot_llava.py
------------------------------------
Real LLaVA-OneVision VQA inference (via the local vLLM server) on the
genuine Europeana pilot images -- same model, same 4-question protocol as
run_llava_vqa.py. vlm_agent score is derived from a simple, transparent
rule (fraction of questions the model answers with a substantive,
non-empty response), not any hardcoded or RNG-generated value.

Requires a running vLLM server on localhost:8000 hosting
llava-hf/llava-onevision-qwen2-7b-ov-hf.

Output: results/external_archive/europeana_real_pilot/llava_scores.csv
"""

import base64
import json
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BASE = Path("/data/brhanu/thesis_project")
PILOT_DIR = BASE / "results" / "external_archive" / "europeana_real_pilot"
METADATA_CSV = PILOT_DIR / "metadata.csv"
OUTPUT_CSV = PILOT_DIR / "llava_scores.csv"

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

QUESTIONS = [
    "Describe this image in one sentence.",
    "What scene or activity is shown?",
    "Are there people, children, or animals visible?",
    "Is this an indoor or outdoor scene?",
]


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_llava(image_path: str, question: str) -> str | None:
    try:
        image_b64 = encode_image_base64(image_path)
        payload = {
            "model": MODEL_NAME,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }],
            "temperature": 0.0,
            "max_tokens": 200,
        }
        response = requests.post(API_URL, json=payload, timeout=120)
        if response.status_code != 200:
            print(f"  ⚠ HTTP {response.status_code}: {response.text[:150]}")
            return None
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        return None


def main() -> None:
    metadata = pd.read_csv(METADATA_CSV)
    print(f"📂 {len(metadata)} real Europeana images. Querying LLaVA-OneVision via vLLM...")

    rows = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="LLaVA-OneVision VQA"):
        img_id = row["image_id"]
        img_path = row["image_path"]
        answers = {}
        n_answered = 0
        for q in QUESTIONS:
            ans = query_llava(img_path, q)
            answers[q] = ans
            if ans and len(ans.strip()) > 3:
                n_answered += 1

        vlm_agent = n_answered / len(QUESTIONS)
        rows.append({
            "image_id": img_id,
            "vlm_agent": round(vlm_agent, 4),
            "vqa_description": answers.get(QUESTIONS[0], ""),
            "vqa_scene_activity": answers.get(QUESTIONS[1], ""),
            "vqa_people_present": answers.get(QUESTIONS[2], ""),
            "vqa_indoor_outdoor": answers.get(QUESTIONS[3], ""),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Wrote {OUTPUT_CSV}")
    print(f"   Mean vlm_agent: {df['vlm_agent'].mean():.4f}")


if __name__ == "__main__":
    main()
