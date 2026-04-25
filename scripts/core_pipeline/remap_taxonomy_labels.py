#!/usr/bin/env python3
import os
import glob

# Configuration
LABELS_DIR = "/data/brhanu/thesis_project/final_dataset/labels/yolo_labels/"
# New Mapping: {old_idx: new_idx}
# 0: person -> 0
# 1: child -> 1
# 2: horse -> 2 (Merged into animal)
# 3: building -> 3
# 4: weapon -> 4
# 5: vehicle -> 5
# 6: tree -> 6
# 7: clothing -> None (Drop)
# 8: text -> 7
# 9: animal -> 2
REMAP_DICT = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 8: 7, 9: 2}

def remap_labels():
    label_files = glob.glob(os.path.join(LABELS_DIR, "*.txt"))
    print(f"🔄 Remapping {len(label_files)} label files...")
    
    count = 0
    for file_path in label_files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.split()
            if not parts: continue
            
            old_idx = int(parts[0])
            if old_idx in REMAP_DICT:
                new_idx = REMAP_DICT[old_idx]
                new_lines.append(f"{new_idx} {' '.join(parts[1:])}\n")
        
        # Only rewrite if changed or always for consistency? 
        # Always rewrite to ensure clean indices
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        
        count += 1
        if count % 1000 == 0:
            print(f"  Processed {count} files...")

    print(f"✅ Label remapping complete. {count} files updated.")

if __name__ == "__main__":
    remap_labels()
