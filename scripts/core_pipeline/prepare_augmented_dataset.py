#!/usr/bin/env python3
import os
from pathlib import Path

# Paths
BASE_DIR = Path("/data/brhanu/thesis_project/data")
V1_DIR = BASE_DIR / "yolo11_v1_refresh"
AUG_DIR = BASE_DIR / "augmented_dataset_v1"
V2_DIR = BASE_DIR / "yolo11_v2_augmented"

def create_symlinks(src_dir, dest_dir, pattern="*"):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src_path in src_dir.glob(pattern):
        if src_path.is_file() or src_path.is_symlink():
            dest_link = dest_dir / src_path.name
            if dest_link.exists():
                dest_link.unlink()
            # Use absolute path for reliability
            os.symlink(src_path.absolute(), dest_link)

def main():
    print(f"📂 Creating augmented dataset at: {V2_DIR}")
    
    # 1. Copy structure and symlink original training data
    print("🔗 Symlinking original training data...")
    create_symlinks(V1_DIR / "images/train", V2_DIR / "images/train")
    create_symlinks(V1_DIR / "labels/train", V2_DIR / "labels/train")
    
    # 2. Symlink original validation data (keep it consistent)
    print("🔗 Symlinking original validation data...")
    create_symlinks(V1_DIR / "images/val", V2_DIR / "images/val")
    create_symlinks(V1_DIR / "labels/val", V2_DIR / "labels/val")
    
    # 3. Add augmented images and labels
    print("✨ Adding augmented data...")
    for cls in ["weapon", "vehicle", "hat", "furniture"]:
        cls_img_dir = AUG_DIR / cls
        if cls_img_dir.exists():
            print(f"  Adding {cls} images...")
            create_symlinks(cls_img_dir, V2_DIR / "images/train", "*.jpg")
            
    # Add augmented labels
    aug_label_dir = AUG_DIR / "labels/train"
    if aug_label_dir.exists():
        print("  Adding augmented labels...")
        create_symlinks(aug_label_dir, V2_DIR / "labels/train", "*.txt")
        
    print(f"✅ Dataset preparation complete. {V2_DIR}")

if __name__ == "__main__":
    main()
