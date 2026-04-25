#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_llava_vqa_fixed.py
------------------------------------------------
Query the running vLLM LLaVA-OneVision server (local)
for image captioning + VQA using base64-encoded images.

Fixes the 400 Bad Request issue (no file:// allowed).
------------------------------------------------
"""

import argparse
import os, json, glob, requests, base64, sys, time
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

# === QUESTIONS ===
QUESTIONS = [
    "Describe this image in one sentence.",
    "What scene or activity is shown?",
    "Are there people, children, or animals visible?",
    "Is this an indoor or outdoor scene?",
]

def encode_image_base64(image_path):
    """Convert image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def query_llava(image_path, question):
    """Send image (as base64) + question to local vLLM LLaVA endpoint."""
    try:
        image_b64 = encode_image_base64(image_path)
        payload = {
            "model": MODEL_NAME,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }]
        }

        response = requests.post(API_URL, json=payload, timeout=300)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")
        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"⚠️ Error processing {os.path.basename(image_path)}: {e}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLaVA VQA over restored images.")
    parser.add_argument(
        "--dataset-root",
        default="/data/brhanu/thesis_project/final_dataset",
        help="Path to final_dataset root",
    )
    parser.add_argument(
        "--output-path",
        default="/data/brhanu/thesis_project/results/llava_vqa_responses.json",
        help="Output JSON file path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    data_dir = dataset_root / "images" / "restored"
    out_path = Path(args.output_path)

    images = sorted(glob.glob(str(data_dir / "**" / "*.jpg"), recursive=True))
    results = {}

    print(f"🔹 Found {len(images)} images. Querying LLaVA-OneVision (base64 mode)...")
    if not images:
        print(f"❌ No images found under {data_dir}")
        sys.exit(1)

    for img_path in tqdm(images):
        img_id = os.path.basename(img_path)
        responses = {}
        for q in QUESTIONS:
            ans = query_llava(img_path, q)
            responses[q] = ans
            time.sleep(0.5)
        results[img_id] = responses

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Completed LLaVA VQA (base64 fixed).")
    print(f"📁 Results saved to: {out_path}")


if __name__ == "__main__":
    main()