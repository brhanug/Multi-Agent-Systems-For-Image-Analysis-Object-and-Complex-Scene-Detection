#!/usr/bin/env python3
"""
run_europeana_real_pilot_kosmos.py
-------------------------------------
Real Kosmos-2.5 grounded-OCR inference on the genuine Europeana pilot images.
Same model, same prompt ("<md>" markdown generation), same document_agent
scoring formula as the deployed pipeline (min(len(text)/80, 1.0), matching
config.yaml's document_char_saturation=80).

Output: results/external_archive/europeana_real_pilot/kosmos_scores.csv
"""

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

BASE = Path("/data/brhanu/thesis_project")
PILOT_DIR = BASE / "results" / "external_archive" / "europeana_real_pilot"
METADATA_CSV = PILOT_DIR / "metadata.csv"
OUTPUT_CSV = PILOT_DIR / "kosmos_scores.csv"

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
DOCUMENT_CHAR_SATURATION = 80.0


def main() -> None:
    metadata = pd.read_csv(METADATA_CSV)
    print(f"🚀 Loading Kosmos-2.5 on {DEVICE}...")
    model = AutoModelForVision2Seq.from_pretrained(
        "microsoft/kosmos-2.5", trust_remote_code=True, device_map=DEVICE, dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2.5", trust_remote_code=True)

    rows = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Kosmos-2.5"):
        img_id = row["image_id"]
        img_path = row["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(text="<md>", images=image, return_tensors="pt").to(device=DEVICE, dtype=torch.float16)
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception as e:
            print(f"  ⚠ Kosmos-2.5 failed on {img_id}: {e}")
            generated_text = ""

        document_agent = min(len(generated_text) / DOCUMENT_CHAR_SATURATION, 1.0)
        rows.append({
            "image_id": img_id,
            "kosmos_markdown": generated_text,
            "kosmos_text_len": len(generated_text),
            "document_agent": round(document_agent, 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Wrote {OUTPUT_CSV}")
    print(f"   Mean document_agent: {df['document_agent'].mean():.4f}")


if __name__ == "__main__":
    main()
