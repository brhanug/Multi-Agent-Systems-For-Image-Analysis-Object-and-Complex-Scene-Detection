import pandas as pd
import os
import shutil
from pathlib import Path

# Config
MANIFEST_PATH = "/data/brhanu/thesis_project/final_dataset/metadata/manifest_v2.csv"
PROJECT_ROOT = Path("/data/brhanu/thesis_project")
KIT_DIR = PROJECT_ROOT / "human_baseline_gold_kit"
IMAGES_OUT_DIR = KIT_DIR / "images"
WORKSHEET_PATH = KIT_DIR / "labeling_worksheet.csv"

# Create directories
os.makedirs(IMAGES_OUT_DIR, exist_ok=True)

def prepare_kit():
    print("📂 Loading manifest...")
    df = pd.read_csv(MANIFEST_PATH)
    diff_dir = PROJECT_ROOT / "final_dataset/images/diffusion_restored"
    diff_files = set(os.listdir(diff_dir))
    
    # 1. Pool A: Gold Tier (Basename exists in diffusion_restored)
    df['basename_key'] = df['image_id'].apply(lambda x: Path(x).name + ".jpg")
    gold_candidates = df[df['basename_key'].isin(diff_files)]
    
    # Handle potential duplicates (basename matches multiple rows)
    # Pick the one with highest vqa_scene_confidence or first
    gold_pool = gold_candidates.sort_values('vqa_scene_confidence', ascending=False).drop_duplicates('basename_key')
    print(f"✨ Found {len(gold_pool)} unique Gold Tier images.")
    
    # 2. Pool B: Silver Tier (High confidence, not in gold)
    needed = 500 - len(gold_pool)
    silver_candidates = df[~df['image_id'].isin(gold_pool['image_id'])]
    silver_pool = silver_candidates.sort_values(by='vqa_scene_confidence', ascending=False).head(needed)
    print(f"🥈 Selected {len(silver_pool)} high-confidence Silver Tier images.")
    
    # Combined selection
    final_selection = pd.concat([gold_pool, silver_pool])
    
    # Prepare worksheet
    worksheet_df = pd.DataFrame()
    worksheet_df['Image_ID'] = final_selection['image_id']
    worksheet_df['Tier'] = ['Gold' if x in gold_pool['image_id'].values else 'Silver' for x in final_selection['image_id']]
    
    # Add annotation columns
    classes = ["Person", "Child", "Horse", "Building", "Weapon", "Vehicle", "Tree", "Clothing", "Text", "Animal"]
    for cls in classes:
        worksheet_df[cls] = ""
    
    worksheet_df['Primary_Scene'] = ""
    worksheet_df['Ambiguity_Notes'] = ""
    
    # Copy/Symlink images
    print(f"🚀 Preparing {len(final_selection)} images in {IMAGES_OUT_DIR}...")
    for idx, row in final_selection.iterrows():
        # Check Gold dir first
        gold_path = diff_dir / row['basename_key']
        if gold_path.exists() and row['image_id'] in gold_pool['image_id'].values:
            src_path = gold_path
        else:
            src_path = PROJECT_ROOT / "final_dataset" / row['restored_path']
            
        dst_name = f"{row['image_id'].replace('/', '_')}.jpg"
        dst_path = IMAGES_OUT_DIR / dst_name
        
        if src_path.exists():
            if os.path.exists(dst_path):
                os.remove(dst_path)
            os.symlink(src_path, dst_path)
        else:
            print(f"⚠️ Warning: File not found {src_path}")
            
    # Save worksheet
    worksheet_df.to_csv(WORKSHEET_PATH, index=False)
    print(f"✅ Worksheet saved to {WORKSHEET_PATH}")
    print(f"🎉 Kit preparation complete! Total images: {len(final_selection)}")

if __name__ == "__main__":
    prepare_kit()
