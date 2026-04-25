#!/usr/bin/env python3
import json
import os
from collections import Counter

# Configuration
FLORENCE_JSON = "/data/brhanu/thesis_project/results/florence2_detections_v1.json"

def discover_classes():
    print(f"🔍 Analyzing raw open-vocabulary detections from: {FLORENCE_JSON}")
    
    if not os.path.exists(FLORENCE_JSON):
        print("❌ File not found.")
        return

    with open(FLORENCE_JSON, 'r') as f:
        data = json.load(f)

    # Counter for raw labels
    label_counts = Counter()
    
    total_images_analyzed = len(data)
    total_detections = 0
    
    for item in data:
        for det in item.get('detections', []):
            # Normalize label (lowercase, strip whitespace)
            label = det.get('label', '').lower().strip()
            if label:
                label_counts[label] += 1
                total_detections += 1

    print(f"\n📊 Processed {total_images_analyzed} images.")
    print(f"🎯 Total raw open-vocabulary detections: {total_detections}")
    print(f"🧠 Unique object classes discovered: {len(label_counts)}")

    print("\n🏆 Top 30 Most Frequent Raw Classes:")
    print("-" * 50)
    for label, count in label_counts.most_common(30):
        print(f"{label:<30} {count:>10}")

    # Write results to a markdown file for review
    out_path = "/data/brhanu/thesis_project/results/open_vocab_discovery.md"
    with open(out_path, "w") as f:
        f.write("# Open-Vocabulary Class Discovery\n\n")
        f.write(f"- **Images Analyzed**: {total_images_analyzed}\n")
        f.write(f"- **Total Native Detections**: {total_detections}\n")
        f.write(f"- **Unique Classes Found**: {len(label_counts)}\n\n")
        f.write("## Top 50 Most Frequent Classes\n")
        f.write("| Rank | Class Label | Detection Count |\n")
        f.write("|------|-------------|-----------------|\n")
        for i, (label, count) in enumerate(label_counts.most_common(50), 1):
            f.write(f"| {i} | `{label}` | {count} |\n")

    print(f"\n✅ Full report saved to: {out_path}")

if __name__ == "__main__":
    discover_classes()
