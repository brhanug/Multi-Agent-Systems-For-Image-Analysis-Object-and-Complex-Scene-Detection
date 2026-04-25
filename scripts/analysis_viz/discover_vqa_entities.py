#!/usr/bin/env python3
import pandas as pd
from collections import Counter
import re

MANIFEST_PATH = "/data/brhanu/thesis_project/final_dataset/metadata/manifest.csv"
OUTPUT_MD = "/data/brhanu/thesis_project/results/open_vocab_discovery.md"

def discover_from_captions():
    print(f"🔍 Loading full archive manifest: {MANIFEST_PATH}")
    try:
        df = pd.read_csv(MANIFEST_PATH)
    except Exception as e:
        print(f"❌ Failed to load manifest: {e}")
        return

    # Using BLIP-2 captions as the open-vocabulary source
    target_col = 'blip2_caption'
    if target_col not in df.columns:
        print(f"❌ Column '{target_col}' not found in manifest.")
        return

    # Basic stopwords and non-object words to filter out
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'to', 'in', 'on', 'with', 'is', 'are', 'of', 'for', 'by', 'at',
        'from', 'this', 'that', 'there', 'it', 'its', 'as', 'an', 'some', 'many', 'few', 'two', 'three',
        'picture', 'photo', 'image', 'drawing', 'showing', 'shows', 'black', 'white', 'man', 'woman', 
        'people', 'old', 'young', 'large', 'small', 'group', 'standing', 'sitting', 'front', 'back',
        'next', 'near', 'side', 'top', 'bottom', 'very', 'which', 'who', 'where', 'when', 'how', 'what',
        'their', 'they', 'he', 'she', 'his', 'her', 'we', 'us', 'our', 'my', 'mine', 'your', 'yours'
    }
    
    object_counts = Counter()
    total_valid = 0

    print("🧠 Extracting visual entities from open-vocabulary captions...")
    for idx, row in df.iterrows():
        text = str(row[target_col]).lower()
        if text == 'n/a' or text == 'nan' or not text:
            continue
            
        # Clean text: keep only letters, split into words representing potential objects
        words = re.findall(r'\b[a-z]{3,}\b', text)
        entities = [w for w in words if w not in stopwords]
        
        # We can also do simple bi-grams if needed, but unigrams are good enough for a rough taxonomy
        for entity in entities:
            object_counts[entity] += 1
            
        total_valid += 1

    print(f"\n📊 Processed {total_valid} images with open-vocabulary captions.")
    print(f"🎯 Unique semantic entities discovered: {len(object_counts)}")

    print("\n🏆 Top 40 Most Frequent Semantic Entities:")
    print("-" * 50)
    for label, count in object_counts.most_common(40):
        print(f"{label:<30} {count:>10}")

    with open(OUTPUT_MD, "w") as f:
        f.write("# Open-Vocabulary Taxonomy Discovery\n\n")
        f.write("This analysis extracts the most frequently naturally occurring visual objects across the 12,110 image archive, based on unconstrained BLIP-2 descriptions.\n\n")
        f.write(f"- **Images Analyzed**: {total_valid}\n")
        f.write(f"- **Unique Entities Found**: {len(object_counts)}\n\n")
        f.write("## Top 50 Most Frequent Open-Vocabulary Objects\n")
        f.write("| Rank | Unconstrained Entity | Frequency |\n")
        f.write("|------|----------------------|-----------|\n")
        for i, (label, count) in enumerate(object_counts.most_common(50), 1):
            f.write(f"| {i} | `{label}` | {count} |\n")

    print(f"\n✅ Full report saved to: {OUTPUT_MD}")

if __name__ == "__main__":
    discover_from_captions()
