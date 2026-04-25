
import os
import json
import argparse
import requests
import base64
import pandas as pd
import random
from pathlib import Path
from tqdm import tqdm
import time

# === CONFIG ===
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

# Binary Questions: Expect "Yes" or "No"
QUESTIONS = [
    "Is the text in this image legible appropriate for reading?",
    "Are the facial features of the people clearly visible and sharp?",
    "Is the fine detail of the clothing or objects clear and distinct?"
]

def encode_image_base64(image_path):
    """Convert image to base64 string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None

def query_llava_binary(image_path, question, mock=False):
    """
    Query LLaVA and return 1 for Yes, 0 for No/Unsure.
    """
    if mock:
        # Simulate response
        return random.choice([0, 1])

    image_b64 = encode_image_base64(image_path)
    if not image_b64:
        return 0

    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Answer with only 'Yes' or 'No'. {question}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                }
            ]
        }],
        "max_tokens": 10,
        "temperature": 0.0 # Deterministic
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        if response.status_code != 200:
            print(f"Error API: {response.status_code}")
            return 0
        
        content = response.json()["choices"][0]["message"]["content"].strip().lower()
        if "yes" in content:
            return 1
        return 0
    except Exception as e:
        print(f"Request Error: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Calculate Semantic Restoration Score (SRS)")
    parser.add_argument("dataset_root", help="Root of final_dataset")
    parser.add_argument("--mock", action="store_true", help="Use mock responses instead of API")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_root)
    manifest_path = dataset_path / "metadata" / "manifest.csv"
    output_path = dataset_path / "metadata" / "srs_scores.json"

    if not manifest_path.exists():
        print(f"❌ Manifest not found at {manifest_path}")
        return

    print("📥 Loading Manifest...")
    df = pd.read_csv(manifest_path)
    
    # Filter valid pairs
    valid_df = df[df['original_path'].notna() & df['restored_path'].notna()]
    print(f"🔍 Found {len(valid_df)} valid original-restored pairs.")
    
    if args.limit:
        valid_df = valid_df.head(args.limit)
        print(f"⚠️ Limiting processing to {len(valid_df)} samples.")

    results = []
    
    print("🚀 Starting Evaluation loop...")
    for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df)):
        orig_full = dataset_path / row['original_path']
        rest_full = dataset_path / row['restored_path']
        
        if not orig_full.exists() or not rest_full.exists():
            continue
            
        # Score Original
        score_orig = 0
        for q in QUESTIONS:
            score_orig += query_llava_binary(orig_full, q, mock=args.mock)
            
        # Score Restored
        score_rest = 0
        for q in QUESTIONS:
            score_rest += query_llava_binary(rest_full, q, mock=args.mock)
            
        # SRS = Gain
        srs_gain = score_rest - score_orig
        
        results.append({
            "image_id": row['image_id'],
            "score_original": score_orig,
            "score_restored": score_rest,
            "srs_gain": srs_gain
        })
        
        if not args.mock:
            time.sleep(0.5) # Rate limit

    # Save
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    # Summary
    if results:
        avg_gain = sum(r['srs_gain'] for r in results) / len(results)
        print(f"\n✅ Evaluation Complete.")
        print(f"📊 Average SRS Gain: {avg_gain:.2f} (Positive means improvement)")
        print(f"💾 Results saved to {output_path}")
    else:
        print("❌ No results computed.")

if __name__ == "__main__":
    main()
