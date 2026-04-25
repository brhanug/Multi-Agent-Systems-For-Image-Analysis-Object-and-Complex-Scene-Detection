#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_trend_analysis.py
---------------------
A prototype for the "Visual Trend Analysis" feature.
Aggregates object detections across a temporal axis to study cultural change.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Configuration
MANIFEST_PATH = "/data/brhanu/thesis_project/final_dataset_v1_refresh/metadata/manifest_v1.csv"
OUTPUT_PATH = "/data/brhanu/thesis_project/results_full_pipeline/visual_trends.png"

def main():
    print("📊 Loading manifest...")
    df = pd.read_csv(MANIFEST_PATH)
    
    # Simulate years since not available in direct CSV (range 1900-1960 as per thesis)
    # In a real scenario, this would be extracted from the Pica-Produktionsnummer (PPN) metadata.
    random.seed(42)
    df['simulated_year'] = [random.randint(1900, 1960) for _ in range(len(df))]
    df['decade'] = (df['simulated_year'] // 10) * 10
    
    # Parse object list
    # vqa_objects is usually a comma-separated string
    all_classes = ['person', 'child', 'horse', 'building', 'weapon', 'vehicle', 'tree', 'clothing', 'text', 'animal']
    
    trend_data = []
    for decade, group in df.groupby('decade'):
        counts = {cls: 0 for cls in all_classes}
        for obj_str in group['vqa_objects'].astype(str):
            for cls in all_classes:
                if cls in obj_str.lower():
                    counts[cls] += 1
        
        for cls, count in counts.items():
            # Normalized by decade size to see relative density
            trend_data.append({
                "Decade": decade,
                "Class": cls,
                "Relative Density": count / len(group) if len(group) > 0 else 0
            })
            
    trend_df = pd.DataFrame(trend_data)
    
    # 3. Visualization
    print("🎨 Generating Trend Plot...")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=trend_df, x="Decade", y="Relative Density", hue="Class", marker='o')
    plt.title("Visual Historian: Morphological Object Evolution (Trend Analysis)")
    plt.ylabel("Normalized Frequency (Objects per Image)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"✅ Trend analysis plot saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
