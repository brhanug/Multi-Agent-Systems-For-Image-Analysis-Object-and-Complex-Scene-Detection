#!/usr/bin/env python3
"""Feasibility Test: Temporal Visual Evolution Analysis"""

import os
import re
from pathlib import Path
from collections import defaultdict

def check_temporal_feasibility():
    print("=" * 60)
    print("TEMPORAL VISUAL EVOLUTION - FEASIBILITY TEST")
    print("=" * 60)
    
    # Check results directory
    results_dir = Path("results/yolo11_final_open_vocab")
    if not results_dir.exists():
        print("❌ No YOLO results found")
        return False
    
    print(f"✓ Found results directory: {results_dir}")
    
    # List experiment runs
    exp_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    print(f"✓ Found {len(exp_dirs)} experiment runs")
    
    # Check for detection files
    detection_files = list(results_dir.rglob("*.txt"))
    print(f"✓ Found {len(detection_files)} detection label files")
    
    # Sample format
    if detection_files:
        sample = detection_files[0]
        print(f"\nSample: {sample.name}")
        with open(sample) as f:
            line = f.readline().strip()
            print(f"Format: {line}")
    
    # Check data directory for images
    data_dir = Path("data/colibri")
    if data_dir.exists():
        images = list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.png"))
        print(f"✓ Found {len(images)} images")
        
        # Check for year patterns
        years = set()
        for img in images[:100]:  # Sample first 100
            year_match = re.search(r'\b(19|20)\d{2}\b', img.name)
            if year_match:
                years.add(year_match.group())
        
        if years:
            print(f"✓ Found year patterns: {sorted(years)}")
        else:
            print("⚠ No year patterns in filenames - metadata needed")
    
    print("\n" + "=" * 60)
    print("FEASIBILITY: ✅ POSSIBLE with metadata enhancement")
    print("=" * 60)
    print("Next steps:")
    print("1. Extract/create image date metadata")
    print("2. Aggregate detections by time period")
    print("3. Build temporal trend visualizations")
    
    return True

if __name__ == "__main__":
    check_temporal_feasibility()
