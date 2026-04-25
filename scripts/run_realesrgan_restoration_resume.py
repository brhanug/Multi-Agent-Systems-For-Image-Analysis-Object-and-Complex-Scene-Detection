#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_realesrgan_restoration_resume.py
------------------------------------
Memory-safe, resumable Real-ESRGAN enhancement for large datasets.

✅ Automatically skips already restored images
✅ Retries failed tiles with smaller tile sizes (memory-safe)
✅ Logs progress to `esrgan_resume.log`

Usage:
  python ~/thesis_project/scripts/run_realesrgan_restoration_resume.py \
    --input /data/brhanu/thesis_project/data/colibri/images/data \
    --output /data/brhanu/thesis_project/results_full_parallel/esrgan_full_restored
"""

import os
import cv2
import torch
import argparse
import traceback
from tqdm import tqdm
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# ---------------------------
# Parse CLI arguments
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Input folder containing images")
parser.add_argument("--output", required=True, help="Output folder for restored images")
parser.add_argument("--log", default="~/thesis_project/logs/esrgan_resume.log", help="Path to log file")
args = parser.parse_args()

input_dir = args.input
output_dir = args.output
log_path = os.path.expanduser(args.log)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.dirname(log_path), exist_ok=True)

# ---------------------------
# Model setup
# ---------------------------
model_path = "/data/brhanu/thesis_project/Real-ESRGAN/weights/RealESRGAN_x4plus.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model weights not found: {model_path}")

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

def load_upscaler(tile_size):
    return RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=tile_size,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available()
    )

upscaler = load_upscaler(512)

# ---------------------------
# Helper function
# ---------------------------
def safe_restore(img_path, out_path):
    global upscaler
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Unreadable image")

        output, _ = upscaler.enhance(img, outscale=2)
        cv2.imwrite(out_path, output)
        return True

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            # retry with smaller tile
            for smaller_tile in [256, 128, 64]:
                try:
                    upscaler = load_upscaler(smaller_tile)
                    output, _ = upscaler.enhance(img, outscale=2)
                    cv2.imwrite(out_path, output)
                    return True
                except RuntimeError:
                    continue
            print(f"[OOM] Skipped (even smallest tile failed): {img_path}")
            return False
        else:
            print(f"[!] Failed {img_path}: {e}")
            return False

    except Exception as e:
        print(f"[!] Error {img_path}: {e}")
        traceback.print_exc()
        return False

# ---------------------------
# Resume-safe loop
# ---------------------------
processed = 0
failed = 0

all_images = [
    os.path.join(root, f)
    for root, _, files in os.walk(input_dir)
    for f in files
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

print(f"🖼 Found {len(all_images)} total images to process.")

with open(log_path, "a") as log:
    for img_path in tqdm(all_images, desc="Enhancing images", unit="img"):
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path):
            continue  # skip already done

        ok = safe_restore(img_path, out_path)
        if ok:
            processed += 1
            log.write(f"✅ {img_path}\n")
        else:
            failed += 1
            log.write(f"❌ {img_path}\n")
        log.flush()

print(f"\n✅ Restoration complete.")
print(f"Processed successfully: {processed}")
print(f"Failed: {failed}")
print(f"📁 Output: {output_dir}")
print(f"🧾 Log: {log_path}")