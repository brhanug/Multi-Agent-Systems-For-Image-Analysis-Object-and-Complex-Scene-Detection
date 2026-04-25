#!/usr/bin/env python3
import pandas as pd
import re
import os

def clean_kosmos_text(text):
    if pd.isna(text) or text == "N/A":
        return text
    
    # Remove excessive repetitions of "**Coco**" or similar hallucinatory patterns
    # If the pattern repeats more than 3 times, truncate it
    text = re.sub(r'(\*\*Coco\*\*\s*){3,}', '**Coco** [Truncated Hallucination]', text)
    
    # Also clean up "A dog is a good example of a dog" loops
    text = re.sub(r'(A dog is a good example of a dog being a good example of a dog\.\s*){3,}', 'A dog... [Truncated Hallucination]', text)
    
    # General cleanup: if the text is just repeating the same sentence infinitely
    lines = str(text).split('\n')
    if len(lines) > 20:
        unique_lines = []
        for line in lines:
            if line not in unique_lines:
                unique_lines.append(line)
            elif len(unique_lines) > 5 and line == unique_lines[-1]: # consecutive repeat
                continue
        text = '\n'.join(unique_lines[:50]) # Keep at most 50 unique lines for safety
        
    return text

def main():
    path = "/data/brhanu/thesis_project/final_dataset/metadata/manifest.csv"
    out_path = "/data/brhanu/thesis_project/final_dataset/metadata/manifest_cleaned.csv"
    
    if not os.path.exists(path):
        print(f"❌ Error: {path} not found.")
        return

    print(f"🧹 Loading manifest from {path}...")
    # Using low_memory=False to handle the large fields
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    original_len = len(df)
    print(f"📊 Original row count: {original_len}")
    
    # Truncate image_id duplicates (if any)
    df = df.drop_duplicates(subset=['image_id'])
    print(f"📊 Unique image count: {len(df)}")
    
    # Clean the Kosmos Markdown column
    for col in ['kosmos_markdown', 'kosmos_md']:
        if col in df.columns:
            print(f"🧼 Cleaning '{col}' hallucinations...")
            df[col] = df[col].apply(clean_kosmos_text)
        
    # Save the cleaned manifest
    df.to_csv(out_path, index=False)
    print(f"✅ Cleaned manifest saved to {out_path}")
    
    # Replace old one
    os.rename(path, path + ".bak") # Keep backup
    os.rename(out_path, path)
    print(f"🚀 Replaced original with cleaned version. (Backup at {path}.bak)")

if __name__ == "__main__":
    main()
