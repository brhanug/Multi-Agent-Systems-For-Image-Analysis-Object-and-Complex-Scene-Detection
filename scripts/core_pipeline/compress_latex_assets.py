#!/usr/bin/env python3
from PIL import Image
from pathlib import Path

def compress_image(source_path, target_path, format="JPEG", quality=85, max_size=(1024, 1024)):
    print(f"Compressing {source_path.name}...")
    img = Image.open(source_path)
    # Convert to RGB if saving as JPEG
    if format == "JPEG" and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    # Resize if extremely large to save compile time/memory
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    img.save(target_path, format=format, quality=quality, optimize=True)
    print(f"  Saved to {target_path.name} (Size: {target_path.stat().st_size / 1024:.1f} KB)")

def main():
    assets_dir = Path("/data/brhanu/thesis_project/latex_assets")
    
    # 1. Compress restoration_output_1.png (17.4 MB) -> restoration_output_1.jpg
    src_restoration = assets_dir / "restoration_output_1.png"
    tar_restoration = assets_dir / "restoration_output_1.jpg"
    if src_restoration.exists():
        compress_image(src_restoration, tar_restoration, format="JPEG", quality=80, max_size=(1200, 1200))
        
    # 2. Compress triptych_example_new.jpg (1.6 MB) -> overwrite
    src_triptych = assets_dir / "triptych_example_new.jpg"
    if src_triptych.exists():
        # Compress in-place to avoid breaking LaTeX filename
        temp_tar = assets_dir / "triptych_example_new_compressed.jpg"
        compress_image(src_triptych, temp_tar, format="JPEG", quality=80, max_size=(1200, 1200))
        temp_tar.replace(src_triptych)
        print("  Overwrote triptych_example_new.jpg in-place with compressed version.")

if __name__ == "__main__":
    main()
