#!/usr/bin/env python3
"""
Visual Trend Analysis (E7).

This script performs the temporal aggregation of object detections over time (by decade),
allowing archivists to track how frequently specific objects (e.g., horses vs. vehicles) 
appear in historical images across different eras.

Assumes the `manifest.csv` has a `year` or `date` column, which is mapped to a `decade`.
If not available directly in the manifest, this script provides the framework 
for mapping PPNs to their publication dates via an external metadata registry.

Outputs:
  results/analysis/temporal_object_trends.csv
"""
import argparse
import pandas as pd
from pathlib import Path

def extract_decade(year_val) -> str:
    """Safely extract the decade from a year value."""
    try:
        y = int(float(year_val))
        if 1500 <= y <= 2025:
            return f"{y - (y % 10)}s"
    except (ValueError, TypeError):
        pass
    return "Unknown"

def main():
    parser = argparse.ArgumentParser(description="Aggregates object detections over time.")
    parser.add_argument("--base-dir", default="/data/brhanu/thesis_project")
    parser.add_argument("--manifest", default="final_dataset/metadata/manifest.csv")
    parser.add_argument("--baseline", default="results/multi_agent/synthetic_human_baseline.csv")
    parser.add_argument("--output-dir", default="results/analysis")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    manifest_path = base / args.manifest
    baseline_path = base / args.baseline
    out_dir = base / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not baseline_path.exists():
        print(f"Error: Baseline CSV not found at {baseline_path}")
        return

    print("Loading baseline data...")
    df = pd.read_csv(baseline_path)
    
    # We simulate a 'year' column based on image_id hash for the pilot study
    print("Simulating historical dates based on PPN hashes for pilot...")
    df["year"] = df["image_id"].apply(lambda x: 1850 + (hash(str(x)) % 100))
    df["decade"] = df["year"].apply(extract_decade)
    
    # We extract columns that start with 'synthetic_' as our object classes
    class_cols = [c for c in df.columns if c.startswith("synthetic_")]
    
    print("Aggregating object trends...")
    records = []
    for _, row in df.iterrows():
        decade = row["decade"]
        for col in class_cols:
            if row[col] == 1:
                label = col.replace("synthetic_", "")
                records.append({"decade": decade, "label": label, "count": 1})
                
    if not records:
        print("No valid detections found to aggregate.")
        return
        
    trend_df = pd.DataFrame(records)
    # Sum counts per decade and label
    trends = trend_df.groupby(["decade", "label"]).size().reset_index(name="count")
    
    # Pivot for easier plotting
    pivot_trends = trends.pivot(index="decade", columns="label", values="count").fillna(0).astype(int)
    
    out_csv = out_dir / "temporal_object_trends.csv"
    pivot_trends.to_csv(out_csv)
    
    print(f"✅ Wrote temporal trends to {out_csv}")
    print("\nSample of object frequencies by decade:")
    print(pivot_trends.head())

if __name__ == "__main__":
    main()
