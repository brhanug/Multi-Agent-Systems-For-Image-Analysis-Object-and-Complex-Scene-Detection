#!/usr/bin/env python3
"""
benchmark_agent4_visual_embedding_linking.py
-----------------------------------------------
A real benchmark of Agent 4's core mechanism, run to close a previously-
documented gap ("no ground truth exists for this task"). Rather than the
earlier TF-IDF-on-VLM-text proxy (which wasn't a visual-feature method at
all), this extracts genuine SigLIP image embeddings -- the same model
family already deployed as Agent 3's scene classifier -- for a sample of
accessible images, then finds the highest-similarity cross-PPN (different
publication series) pairs. This is much closer to Agent 4's actual
described mechanism ("embeds each image's ... visual features, then finds
nearest neighbors across different publication series").

Ground truth for "is this a genuine recurring composition" still doesn't
exist as a labeled dataset -- but the top candidate pairs this script
finds are small enough in number to be visually checked by a single rater
directly, which is what this script enables. Two real findings from doing
that (see the thesis, Agent 4 section, for the full visual write-up):
  1. At very high similarity (~0.94), a genuine match: the literal same
     illustration reprinted across two different digitized volumes.
  2. At moderately-high similarity (~0.90), a false-positive-for-recurrence
     case: two DIFFERENT compositions in the same illustration genre/style
     (same engraving technique, similar period costume), not the same
     picture. The raw embedding captures style, not composition, at this
     similarity band -- a real, useful limit on the mechanism as deployed
     with a single global-similarity threshold.

Outputs:
  results/multi_agent/agent4_visual_embedding_top_pairs.csv
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

BASE = Path("/data/brhanu/thesis_project")
IMAGE_DIR = BASE / "data" / "yolo11_v1_refresh" / "images" / "train"
OUT_CSV = BASE / "results" / "multi_agent" / "agent4_visual_embedding_top_pairs.csv"

SAMPLE_N = 400
RANDOM_STATE = 7
MODEL_NAME = "google/siglip-so400m-patch14-384"
BATCH_SIZE = 16


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    files = sorted(IMAGE_DIR.glob("*.jpg"))
    random.seed(RANDOM_STATE)
    random.shuffle(files)
    files = files[:SAMPLE_N]
    print("sample size:", len(files))

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float16).to(device).eval()

    embeds, stems, ppns = [], [], []
    with torch.no_grad():
        for i in range(0, len(files), BATCH_SIZE):
            batch_files = files[i:i + BATCH_SIZE]
            imgs = [Image.open(f).convert("RGB") for f in batch_files]
            inputs = processor(images=imgs, return_tensors="pt").to(device, torch.float16)
            feats = F.normalize(model.get_image_features(**inputs), dim=-1)
            embeds.append(feats.cpu().float())
            stems.extend(f.stem for f in batch_files)
            ppns.extend(f.stem.split("_")[0] for f in batch_files)
            if i % 80 == 0:
                print(f"  {i}/{len(files)}")

    X = torch.cat(embeds, dim=0)
    sim = X @ X.T

    n = len(stems)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if ppns[i] == ppns[j]:
                continue  # only cross-archive (different PPN) pairs
            pairs.append((sim[i, j].item(), stems[i], stems[j]))
    pairs.sort(reverse=True)

    print("\nTop 10 highest-similarity cross-PPN pairs (real SigLIP visual embeddings):")
    for s, a, b in pairs[:10]:
        print(f"sim={s:.4f}  {a}  <->  {b}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w") as f:
        f.write("similarity,image_a,image_b\n")
        for s, a, b in pairs[:20]:
            f.write(f"{s:.4f},{a},{b}\n")
    print(f"\n✅ Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
