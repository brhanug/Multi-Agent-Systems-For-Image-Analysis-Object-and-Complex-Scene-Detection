#!/usr/bin/env python3
import os
import json
from tqdm import tqdm

BASE = "/data/brhanu/thesis_project"
KOSMOS_DIR = os.path.join(BASE, "results_v1/kosmos")
OUTPUT_JSONL = os.path.join(BASE, "results/kosmos_grounding.jsonl")

def main():
    print(f"Reading from {KOSMOS_DIR}...")
    if not os.path.exists(KOSMOS_DIR):
        print("Error: Kosmos source directory does not exist.")
        return
        
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    
    files = sorted([f for f in os.listdir(KOSMOS_DIR) if f.endswith(".json")])
    print(f"Found {len(files)} JSON files to compile.")
    
    count = 0
    with open(OUTPUT_JSONL, "w") as f_out:
        for fname in tqdm(files):
            fpath = os.path.join(KOSMOS_DIR, fname)
            try:
                with open(fpath, "r") as f_in:
                    data = json.load(f_in)
                
                # Format to match run_kosmos2_5_grounding.py output:
                # {"image": img_name, "kosmos_output": generated_text, "mode": "markdown_generation"}
                img_name = data.get("image", fname.replace(".json", ".jpg"))
                generated_text = data.get("kosmos_md", "")
                
                entry = {
                    "image": img_name,
                    "kosmos_output": generated_text,
                    "mode": "markdown_generation"
                }
                
                f_out.write(json.dumps(entry) + "\n")
                count += 1
            except Exception as e:
                print(f"Error compiling {fname}: {e}")
                
    print(f"Successfully compiled {count} files into {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
