#!/usr/bin/env python3
"""
Select 9,000 high-quality images from the remaining Colibri collection
for Tier 2 of the Zenodo package.

Selection criteria:
1. Not already in processed set (final_dataset/images/original)
2. Diverse PPN distribution
3. Good file size (indicates quality scan)
4. Temporal/collection diversity
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import random

# === CONFIGURATION ===
COLIBRI_ROOT = "/data/brhanu/datasets/colibri"
PROCESSED_DIR = "/data/brhanu/thesis_project/final_dataset/images/original"
OUTPUT_FILE = "/data/brhanu/thesis_project/tier2_selection.txt"
TARGET_COUNT = 9000

def get_processed_images():
    """Get set of already processed image IDs"""
    processed = set()
    
    if not os.path.exists(PROCESSED_DIR):
        print(f"⚠️  Processed directory not found: {PROCESSED_DIR}")
        return processed
    
    for root, dirs, files in os.walk(PROCESSED_DIR):
        for file in files:
            if file.endswith('.jpg'):
                # Extract PPN and image ID
                ppn = os.path.basename(root)
                image_id = f"{ppn}/{file}"
                processed.add(image_id)
    
    print(f"📊 Found {len(processed)} already processed images")
    return processed

def scan_colibri_collection():
    """Scan full Colibri collection and gather metadata"""
    print(f"\n🔍 Scanning Colibri collection: {COLIBRI_ROOT}")
    
    images = []
    ppn_counts = defaultdict(int)
    
    for root, dirs, files in os.walk(COLIBRI_ROOT):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, file)
                
                # Get PPN from directory structure
                rel_path = os.path.relpath(root, COLIBRI_ROOT)
                ppn = rel_path.split('/')[0] if '/' in rel_path else rel_path
                
                # Get file size
                try:
                    size = os.path.getsize(full_path)
                except:
                    continue
                
                image_id = f"{ppn}/{file}"
                
                images.append({
                    'path': full_path,
                    'id': image_id,
                    'ppn': ppn,
                    'size': size,
                    'filename': file
                })
                
                ppn_counts[ppn] += 1
    
    print(f"✅ Found {len(images)} total images across {len(ppn_counts)} PPNs")
    return images, ppn_counts

def select_tier2_images(all_images, processed_set, ppn_counts, target=9000):
    """Select diverse, high-quality subset for Tier 2"""
    print(f"\n🎯 Selecting {target} images for Tier 2...")
    
    # Filter out already processed
    candidates = [img for img in all_images if img['id'] not in processed_set]
    print(f"   Candidates (unprocessed): {len(candidates)}")
    
    # Filter by size (keep images > 100KB, likely good quality)
    MIN_SIZE = 100 * 1024  # 100 KB
    candidates = [img for img in candidates if img['size'] > MIN_SIZE]
    print(f"   After size filter (>100KB): {len(candidates)}")
    
    if len(candidates) <= target:
        print(f"   ⚠️  Only {len(candidates)} candidates available, using all")
        return candidates
    
    # Strategy: Balanced sampling across PPNs
    # Calculate target per PPN
    ppn_targets = {}
    total_unprocessed = len(candidates)
    
    for ppn in ppn_counts.keys():
        ppn_candidates = [c for c in candidates if c['ppn'] == ppn]
        if ppn_candidates:
            # Proportional allocation
            proportion = len(ppn_candidates) / total_unprocessed
            ppn_targets[ppn] = int(target * proportion)
    
    # Select from each PPN
    selected = []
    for ppn, target_count in ppn_targets.items():
        ppn_images = [c for c in candidates if c['ppn'] == ppn]
        
        # Sort by size (prefer larger = better quality)
        ppn_images.sort(key=lambda x: x['size'], reverse=True)
        
        # Take top N, but add some randomness for diversity
        if len(ppn_images) > target_count:
            # Take top 80% by quality, then random sample
            quality_cutoff = int(len(ppn_images) * 0.8)
            quality_subset = ppn_images[:quality_cutoff]
            selected_from_ppn = random.sample(quality_subset, 
                                             min(target_count, len(quality_subset)))
        else:
            selected_from_ppn = ppn_images
        
        selected.extend(selected_from_ppn)
    
    # If we're short, add more from largest PPNs
    if len(selected) < target:
        remaining_candidates = [c for c in candidates if c not in selected]
        remaining_candidates.sort(key=lambda x: x['size'], reverse=True)
        needed = target - len(selected)
        selected.extend(remaining_candidates[:needed])
    
    print(f"✅ Selected {len(selected)} images")
    
    # Print distribution
    ppn_distribution = defaultdict(int)
    for img in selected:
        ppn_distribution[img['ppn']] += 1
    
    print(f"\n📊 Distribution across {len(ppn_distribution)} PPNs:")
    top_ppns = sorted(ppn_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
    for ppn, count in top_ppns:
        print(f"   {ppn}: {count} images")
    if len(ppn_distribution) > 10:
        print(f"   ... and {len(ppn_distribution) - 10} more PPNs")
    
    return selected

def save_selection(selected_images, output_file):
    """Save selected image paths to file"""
    print(f"\n💾 Saving selection to: {output_file}")
    
    with open(output_file, 'w') as f:
        for img in selected_images:
            f.write(f"{img['path']}\n")
    
    # Also save metadata as JSON
    metadata_file = output_file.replace('.txt', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump({
            'total_selected': len(selected_images),
            'selection_criteria': {
                'min_size_kb': 100,
                'strategy': 'balanced_ppn_sampling',
                'quality_preference': 'size_based'
            },
            'images': [
                {
                    'path': img['path'],
                    'id': img['id'],
                    'ppn': img['ppn'],
                    'size_mb': round(img['size'] / (1024**2), 2)
                }
                for img in selected_images
            ]
        }, f, indent=2)
    
    print(f"✅ Saved {len(selected_images)} paths")
    print(f"✅ Saved metadata to: {metadata_file}")
    
    # Calculate total size
    total_size_gb = sum(img['size'] for img in selected_images) / (1024**3)
    print(f"\n📦 Estimated Tier 2 size: {total_size_gb:.2f} GB")

def main():
    print("=" * 70)
    print("TIER 2 IMAGE SELECTION FOR ZENODO PACKAGE")
    print("=" * 70)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Step 1: Get already processed images
    processed = get_processed_images()
    
    # Step 2: Scan full collection
    all_images, ppn_counts = scan_colibri_collection()
    
    # Step 3: Select Tier 2 subset
    selected = select_tier2_images(all_images, processed, ppn_counts, TARGET_COUNT)
    
    # Step 4: Save selection
    save_selection(selected, OUTPUT_FILE)
    
    print("\n" + "=" * 70)
    print("SELECTION COMPLETE")
    print("=" * 70)
    print(f"Selected: {len(selected)} images")
    print(f"Output: {OUTPUT_FILE}")
    print("\nNext step: Run copy_tier2_images.py to add to package")
    print("=" * 70)

if __name__ == "__main__":
    main()
