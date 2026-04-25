#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import shutil

# Configuration
MANIFEST_PATH = "/data/brhanu/thesis_project/final_dataset/metadata/manifest.csv"
OUTPUT_DIR = "/data/brhanu/thesis_project/human_baseline_gold_v2"
TARGET_COUNT = 2000
STRATA_RATIO = {"high": 0.4, "low": 0.4, "random": 0.2}

# Thresholds (based on analysis of vqa_total_objects)
HIGH_THRESHOLD = 4
LOW_THRESHOLD = 1

def generate_stratified_baseline():
    print(f"🚀 Loading manifest from: {MANIFEST_PATH}")
    df = pd.read_csv(MANIFEST_PATH)
    
    # Ensure OUTPUT_DIR exists
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    
    # Define strata
    high_pool = df[df['vqa_total_objects'] >= HIGH_THRESHOLD]
    low_pool = df[df['vqa_total_objects'] <= LOW_THRESHOLD]
    
    # Filter remaining for random pool
    used_ids = set(high_pool['image_id']).union(set(low_pool['image_id']))
    random_pool = df[~df['image_id'].isin(used_ids)]
    
    print(f"Pool sizes - High: {len(high_pool)}, Low: {len(low_pool)}, Random: {len(random_pool)}")
    
    # Calculate target samples
    n_high = int(TARGET_COUNT * STRATA_RATIO["high"])
    n_low = int(TARGET_COUNT * STRATA_RATIO["low"])
    n_random = TARGET_COUNT - n_high - n_low
    
    # Sample
    high_sample = high_pool.sample(n=min(n_high, len(high_pool)), random_state=42)
    low_sample = low_pool.sample(n=min(n_low, len(low_pool)), random_state=42)
    random_sample = random_pool.sample(n=min(n_random, len(random_pool)), random_state=42)
    
    # Combine
    baseline_df = pd.concat([high_sample, low_sample, random_sample])
    baseline_df['strata'] = (['high'] * len(high_sample) + 
                             ['low'] * len(low_sample) + 
                             ['random'] * len(random_sample))
    
    # Save manifest
    baseline_csv = os.path.join(OUTPUT_DIR, "baseline_audit_manifest.csv")
    baseline_df.to_csv(baseline_csv, index=False)
    print(f"✅ Baseline manifest saved to: {baseline_csv} (Total: {len(baseline_df)} images)")
    
    # Copy images (simulated or actual if needed for audit kit)
    # For now, we just list the paths to be prepared
    print(f"📦 Audit Kit prepared in: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_stratified_baseline()
