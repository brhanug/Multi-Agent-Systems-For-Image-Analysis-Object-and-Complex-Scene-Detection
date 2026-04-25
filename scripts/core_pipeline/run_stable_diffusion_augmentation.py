#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_stable_diffusion_augmentation.py
------------------------------------
Generates synthetic archival images for minority classes to balance the dataset.
Targets: 'weapon', 'clothing', 'vehicle'.
"""

import os
import torch
import argparse
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageOps, ImageFilter
import random

# ---------------------------
# Configuration
# ---------------------------
MINORITY_CLASSES = {
    "weapon": [
        "historical archival photograph of a sword on a table",
        "early 20th century documentary photo of a soldier holding a rifle",
        "antique museum display of historical weapons, grainy sepia",
        "blurred historical scan of an old pistol"
    ],
    "clothing": [
        "historical archival photo of traditional 19th century dress",
        "early 20th century documentary photo portrait showing detailed lace clothing",
        "vintage photograph of workers in uniform, industrial era",
        "sepia toned scan of historical military uniform"
    ],
    "vehicle": [
        "historical photo of an early automobile on a cobblestone street",
        "early 20th century archival scan of a horse-drawn carriage",
        "vintage documentary photograph of an old truck, grainy, black and white",
        "blurred historical archive image of an old bicycle"
    ]
}

def apply_archival_filter(image):
    """
    Applies noise, grain, and color shifts to match archival quality.
    """
    # Grayscale/Sepia
    if random.random() > 0.5:
        image = ImageOps.grayscale(image)
        image = ImageOps.colorize(image, "#402000", "#ffcc80") # Sepia shift
    else:
        image = ImageOps.grayscale(image)
        image = ImageOps.colorize(image, "black", "white") # B&W

    # Add slight blur to mimic scan quality
    if random.random() > 0.3:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))
    
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10, help="Images per class")
    parser.add_argument("--output", default="data/augmented_images", help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"🚀 Loading SDXL Pipeline on {args.device}...")
    # Using SDXL Base for high quality
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to(args.device)
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    for cls_name, prompts in MINORITY_CLASSES.items():
        print(f"✨ Generating {args.count} images for class: {cls_name}")
        cls_dir = os.path.join(args.output, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        for i in tqdm(range(args.count)):
            prompt = random.choice(prompts)
            # Add static archival quality modifiers
            full_prompt = f"{prompt}, archival quality, high grain, documentary photography, non-professional scan, 1920s style"
            
            with torch.no_grad():
                image = pipe(
                    prompt=full_prompt, 
                    num_inference_steps=30, 
                    guidance_scale=7.5
                ).images[0]
            
            # Post-processing to match archival look
            image = apply_archival_filter(image)
            
            # Save
            image.save(os.path.join(cls_dir, f"aug_{cls_name}_{i:04d}.jpg"), quality=85)

    print(f"✅ Generation complete. Images saved to {args.output}")

if __name__ == "__main__":
    main()
