import os
from PIL import Image

assets_dir = "/data/brhanu/thesis_project/latex_assets"

total_saved = 0
for fname in os.listdir(assets_dir):
    fpath = os.path.join(assets_dir, fname)
    if not os.path.isfile(fpath):
        continue
    ext = os.path.splitext(fname)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        orig_size = os.path.getsize(fpath)
        if orig_size > 100 * 1024:  # > 100 KB
            try:
                with Image.open(fpath) as img:
                    # Convert RGBA to RGB if saving as JPEG
                    max_dim = 1600
                    w, h = img.size
                    if w > max_dim or h > max_dim:
                        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                    
                    if ext in ['.jpg', '.jpeg']:
                        img.save(fpath, format='JPEG', quality=82, optimize=True)
                    elif ext == '.png':
                        # If RGB/RGBA, save with optimization or convert to palette if appropriate
                        img.save(fpath, format='PNG', optimize=True)
                new_size = os.path.getsize(fpath)
                saved = orig_size - new_size
                if saved > 0:
                    total_saved += saved
                    print(f"Compressed {fname}: {orig_size/1024:.1f}KB -> {new_size/1024:.1f}KB (Saved {saved/1024:.1f}KB)")
            except Exception as e:
                print(f"Error compressing {fname}: {e}")

print(f"\nTotal size saved across images: {total_saved / (1024*1024):.2f} MB")
