import os
import argparse
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
import yaml

# ==============================
# CONFIGURATION
# ==============================
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CFG = load_config()

BASE_DIR = CFG['project']['base_dir']
MODEL_ID = CFG['models'].get('kosmos', {}).get('model_id', "microsoft/kosmos-2.5")
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# Standard input/output directories
IMAGE_DIR = os.path.join(BASE_DIR, "final_dataset/images/diffusion_restored") # Using diffusion restored images
OUTPUT_JSONL = os.path.join(BASE_DIR, "results/kosmos_grounding.jsonl")

def main():
    parser = argparse.ArgumentParser(description="Kosmos-2.5 Grounding & Text Extraction")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images for testing")
    args = parser.parse_args()

    print(f"🚀 Loading Kosmos-2.5 model ({MODEL_ID}) on {DEVICE}...")
    
    # Load Model & Processor
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True, 
        device_map=DEVICE,
        torch_dtype=torch.float16 if "cuda" in DEVICE else torch.float32
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Prepare Image List
    all_images = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if args.limit:
        all_images = all_images[:args.limit]
        print(f"⚠️ Limiting to {args.limit} images for testing.")

    print(f"🔥 Processing {len(all_images)} images...")
    
    # Inference Loop
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    
    # Open for appending or writing new? Let's write new for now.
    with open(OUTPUT_JSONL, "w") as f_out:
        for img_name in tqdm(all_images):
            img_path = os.path.join(IMAGE_DIR, img_name)
            try:
                # Load Image
                image = Image.open(img_path).convert("RGB")
                
                # Kosmos-2.5 Prompting
                # Task: <ocr> for text extraction, <grounding> doesn't work out of the box for generic descriptions like Florence.
                # Kosmos-2.5 is primarily for "Multimodal Literate" tasks (Markdown generation).
                # We will specificially use it to generate the MARKDOWN representation of the image.
                
                prompt = "<md>" # Trigger Markdown generation
                
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(device=DEVICE, dtype=torch.float16)
                
                print(f"DEBUG: Input keys: {inputs.keys()}")
                # Pass all processor outputs to generate
                # This handles 'flattened_patches' vs 'pixel_values' automatically
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024
                )
                
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Cleanup output
                # The model often repeats the prompt or has special tokens.
                # However, batch_decode(skip_special_tokens=True) should handle most.
                
                entry = {
                    "image": img_name,
                    "kosmos_output": generated_text,
                    "mode": "markdown_generation"
                }
                
                f_out.write(json.dumps(entry) + "\n")
                f_out.flush() # Ensure write
                
            except Exception as e:
                print(f"❌ Error processing {img_name}: {e}")

    print(f"✅ Kosmos-2.5 processing complete. Results saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
