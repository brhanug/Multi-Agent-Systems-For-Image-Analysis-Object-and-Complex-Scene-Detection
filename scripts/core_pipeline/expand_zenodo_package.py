#!/usr/bin/env python3
"""
Expand the Zenodo dataset package from 23GB to ~42GB by adding:
1. Original historical scans (6.2 GB)
2. Diffusion-refined gallery subset (0.5 GB)
3. VLM metadata exports (1.5 GB)
4. Training checkpoints (0.5 GB)

This brings the total to ~32 GB as a first step.
Additional images can be processed separately if desired.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
BASE_DIR = "/data/brhanu/thesis_project"
SOURCE_ORIGINAL = os.path.join(BASE_DIR, "final_dataset/images/original")
SOURCE_DIFFUSION = os.path.join(BASE_DIR, "final_dataset/images/diffusion_restored")
SOURCE_VLM_RESULTS = os.path.join(BASE_DIR, "results_v1")
SOURCE_CHECKPOINTS = os.path.join(BASE_DIR, "results/yolo11_final_v1_refresh/exp_final2/weights")

DEST_PACKAGE = os.path.join(BASE_DIR, "final_dataset_v1_refresh")
DEST_ORIGINAL = os.path.join(DEST_PACKAGE, "images/original")
DEST_DIFFUSION = os.path.join(DEST_PACKAGE, "images/diffusion_refined")
DEST_VLM = os.path.join(DEST_PACKAGE, "metadata/vlm_outputs")
DEST_CHECKPOINTS = os.path.join(DEST_PACKAGE, "models/checkpoints")

def get_dir_size(path):
    """Calculate total size of directory in GB"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024**3)  # Convert to GB

def copy_directory_structure(src, dst, description):
    """Copy directory with progress bar"""
    print(f"\n📁 {description}")
    print(f"   Source: {src}")
    print(f"   Destination: {dst}")
    
    if not os.path.exists(src):
        print(f"   ⚠️  Source not found, skipping...")
        return False
    
    # Get total files for progress bar
    total_files = sum([len(files) for r, d, files in os.walk(src)])
    
    with tqdm(total=total_files, desc=f"   Copying", unit="files") as pbar:
        for root, dirs, files in os.walk(src):
            # Create corresponding directory structure
            rel_path = os.path.relpath(root, src)
            dest_dir = os.path.join(dst, rel_path) if rel_path != '.' else dst
            os.makedirs(dest_dir, exist_ok=True)
            
            # Copy files
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dest_dir, file)
                
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                pbar.update(1)
    
    size_gb = get_dir_size(dst)
    print(f"   ✅ Complete ({size_gb:.2f} GB)")
    return True

def export_vlm_metadata():
    """Export VLM results to package"""
    print(f"\n📊 Exporting VLM Metadata")
    os.makedirs(DEST_VLM, exist_ok=True)
    
    # Copy key VLM result directories
    vlm_dirs = {
        "florence2": "Florence-2 detections",
        "groundingdino": "GroundingDINO detections",
        "llava": "LLaVA scene descriptions",
        "kosmos": "Kosmos-2.5 grounding"
    }
    
    total_size = 0
    for dirname, description in vlm_dirs.items():
        src = os.path.join(SOURCE_VLM_RESULTS, dirname)
        dst = os.path.join(DEST_VLM, dirname)
        
        if os.path.exists(src):
            print(f"   Copying {description}...")
            shutil.copytree(src, dst, dirs_exist_ok=True)
            size = get_dir_size(dst)
            total_size += size
            print(f"   ✅ {description}: {size:.2f} GB")
        else:
            print(f"   ⚠️  {description} not found, skipping...")
    
    print(f"   Total VLM metadata: {total_size:.2f} GB")
    return total_size

def copy_training_checkpoints():
    """Copy select training checkpoints"""
    print(f"\n🎯 Copying Training Checkpoints")
    os.makedirs(DEST_CHECKPOINTS, exist_ok=True)
    
    # Copy best and last weights
    checkpoints = ["best.pt", "last.pt"]
    
    total_size = 0
    for ckpt in checkpoints:
        src = os.path.join(SOURCE_CHECKPOINTS, ckpt)
        dst = os.path.join(DEST_CHECKPOINTS, ckpt)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            size = os.path.getsize(dst) / (1024**3)
            total_size += size
            print(f"   ✅ {ckpt}: {size:.3f} GB")
        else:
            print(f"   ⚠️  {ckpt} not found")
    
    # Copy results.csv if exists
    results_csv = os.path.join(SOURCE_CHECKPOINTS.replace("/weights", ""), "results.csv")
    if os.path.exists(results_csv):
        shutil.copy2(results_csv, os.path.join(DEST_CHECKPOINTS, "training_results.csv"))
        print(f"   ✅ Training metrics CSV")
    
    print(f"   Total checkpoints: {total_size:.2f} GB")
    return total_size

def update_readme():
    """Update README with new package structure"""
    readme_path = os.path.join(DEST_PACKAGE, "README.md")
    
    # Read existing README
    if os.path.exists(readme_path):
        with open(readme_path, 'r') as f:
            content = f.read()
    else:
        content = "# Visual Historian Dataset v1.0\n\n"
    
    # Add package structure section
    structure_section = """
## Package Structure (Updated)

### Images
- **original/** (6.2 GB): Raw historical scans from Colibri archive
- **restored/** (23 GB): CycleGAN + Real-ESRGAN enhanced (12,110 images)
- **diffusion_refined/** (0.5 GB): Stable Diffusion gallery subset (231 images)

### Labels
- **yolo_labels/**: 6,480 consensus-based bounding box annotations (YOLO format)

### Metadata
- **manifest_v1.csv**: Complete image metadata with VLM outputs
- **vlm_outputs/**: Raw JSON results from Florence-2, GroundingDINO, LLaVA, Kosmos-2.5

### Models
- **yolo11_best.pt**: Final trained YOLOv11m weights (mAP50: 0.93, mAP50-95: 0.886)
- **checkpoints/**: Training artifacts and metrics

## Dataset Statistics
- **Total Images**: 12,341 (12,110 restored + 231 refined)
- **Original Scans**: 12,110
- **Labeled Images**: 6,480
- **Total Package Size**: ~32 GB

## Reproducibility
This package includes both raw scans and processed outputs, enabling full pipeline reproduction:
1. Original → CycleGAN → Real-ESRGAN → Restored
2. Restored → VLM Ensemble → Pseudo-labels
3. Pseudo-labels → YOLOv11 Training → Final Model
"""
    
    # Append or replace structure section
    if "## Package Structure" in content:
        # Replace existing section
        import re
        content = re.sub(
            r'## Package Structure.*?(?=##|\Z)',
            structure_section,
            content,
            flags=re.DOTALL
        )
    else:
        content += structure_section
    
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"\n📝 Updated README.md")

def main():
    print("=" * 70)
    print("ZENODO DATASET EXPANSION: 23 GB → ~32 GB")
    print("=" * 70)
    
    # Check current package size
    current_size = get_dir_size(DEST_PACKAGE)
    print(f"\n📦 Current package size: {current_size:.2f} GB")
    
    # Step 1: Add original images
    if copy_directory_structure(SOURCE_ORIGINAL, DEST_ORIGINAL, 
                                "Adding Original Historical Scans"):
        pass
    
    # Step 2: Add diffusion-refined gallery
    if copy_directory_structure(SOURCE_DIFFUSION, DEST_DIFFUSION,
                                "Adding Diffusion-Refined Gallery"):
        pass
    
    # Step 3: Export VLM metadata
    vlm_size = export_vlm_metadata()
    
    # Step 4: Copy training checkpoints
    ckpt_size = copy_training_checkpoints()
    
    # Step 5: Update README
    update_readme()
    
    # Final size check
    final_size = get_dir_size(DEST_PACKAGE)
    added_size = final_size - current_size
    
    print("\n" + "=" * 70)
    print("EXPANSION COMPLETE")
    print("=" * 70)
    print(f"Initial size:  {current_size:.2f} GB")
    print(f"Final size:    {final_size:.2f} GB")
    print(f"Added:         {added_size:.2f} GB")
    print(f"Zenodo usage:  {(final_size/50)*100:.1f}% of 50 GB limit")
    print(f"Remaining:     {50-final_size:.2f} GB")
    print("=" * 70)
    
    print("\n✅ Package ready for Zenodo upload!")
    print(f"📂 Location: {DEST_PACKAGE}")

if __name__ == "__main__":
    main()
