#!/usr/bin/env python3
"""
Copy selected Tier 2 images to the Zenodo package.
Preserves directory structure (PPN-based organization).
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
BASE_DIR = "/data/brhanu/thesis_project"
SELECTION_FILE = os.path.join(BASE_DIR, "tier2_selection.txt")
DEST_TIER2 = os.path.join(BASE_DIR, "final_dataset_v1_refresh/tier2_raw_archive/original")

def load_selection(selection_file):
    """Load list of selected image paths"""
    if not os.path.exists(selection_file):
        print(f"❌ Selection file not found: {selection_file}")
        print(f"   Run select_tier2_images.py first!")
        return []
    
    with open(selection_file, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    
    print(f"📋 Loaded {len(paths)} image paths from selection")
    return paths

def copy_tier2_images(image_paths, dest_root):
    """Copy images to Tier 2 directory with PPN structure"""
    print(f"\n📁 Copying images to: {dest_root}")
    os.makedirs(dest_root, exist_ok=True)
    
    copied_count = 0
    skipped_count = 0
    total_size = 0
    
    with tqdm(total=len(image_paths), desc="Copying", unit="images") as pbar:
        for src_path in image_paths:
            if not os.path.exists(src_path):
                skipped_count += 1
                pbar.update(1)
                continue
            
            # Extract PPN and filename from path
            # Assuming structure: /path/to/colibri/PPNXXXXX/image.jpg
            path_parts = Path(src_path).parts
            
            # Find PPN (starts with 'PPN')
            ppn = None
            for part in reversed(path_parts):
                if part.startswith('PPN'):
                    ppn = part
                    break
            
            if not ppn:
                # Fallback: use parent directory name
                ppn = Path(src_path).parent.name
            
            filename = Path(src_path).name
            
            # Create destination path
            dest_ppn_dir = os.path.join(dest_root, ppn)
            os.makedirs(dest_ppn_dir, exist_ok=True)
            
            dest_path = os.path.join(dest_ppn_dir, filename)
            
            # Copy file
            if not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)
                copied_count += 1
                total_size += os.path.getsize(dest_path)
            else:
                skipped_count += 1
            
            pbar.update(1)
    
    total_size_gb = total_size / (1024**3)
    
    print(f"\n✅ Copy complete:")
    print(f"   Copied: {copied_count} images")
    print(f"   Skipped: {skipped_count} (already exist or not found)")
    print(f"   Total size: {total_size_gb:.2f} GB")
    
    return copied_count, total_size_gb

def verify_tier2_structure(dest_root):
    """Verify the copied Tier 2 structure"""
    print(f"\n🔍 Verifying Tier 2 structure...")
    
    ppn_dirs = [d for d in os.listdir(dest_root) 
                if os.path.isdir(os.path.join(dest_root, d))]
    
    total_images = 0
    for ppn in ppn_dirs:
        ppn_path = os.path.join(dest_root, ppn)
        images = [f for f in os.listdir(ppn_path) 
                 if f.endswith(('.jpg', '.jpeg', '.png'))]
        total_images += len(images)
    
    print(f"✅ Verification:")
    print(f"   PPNs: {len(ppn_dirs)}")
    print(f"   Total images: {total_images}")
    
    return total_images

def main():
    print("=" * 70)
    print("TIER 2 IMAGE COPY FOR ZENODO PACKAGE")
    print("=" * 70)
    
    # Step 1: Load selection
    image_paths = load_selection(SELECTION_FILE)
    
    if not image_paths:
        return
    
    # Step 2: Copy images
    copied, size_gb = copy_tier2_images(image_paths, DEST_TIER2)
    
    # Step 3: Verify
    total_images = verify_tier2_structure(DEST_TIER2)
    
    print("\n" + "=" * 70)
    print("TIER 2 COPY COMPLETE")
    print("=" * 70)
    print(f"Location: {DEST_TIER2}")
    print(f"Images: {total_images}")
    print(f"Size: {size_gb:.2f} GB")
    print("\nNext step: Run expand_zenodo_package.py to finalize Tier 1")
    print("=" * 70)

if __name__ == "__main__":
    main()
