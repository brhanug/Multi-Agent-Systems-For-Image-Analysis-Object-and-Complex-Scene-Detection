#!/usr/bin/env python3
import os
import re

NEW_FUNC = """def normalize_id(raw: str) -> str:
    from pathlib import Path
    parts = Path(str(raw)).parts
    if len(parts) >= 2:
        if parts[-2] not in ["original", "restored", "images", "metadata", "results", "CycleGAN", "pix2pix"]:
            return f"{parts[-2]}/{Path(parts[-1]).stem}"
    return Path(str(raw)).stem
"""

# Variations to catch
def patch_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    in_func = False
    skip_next = 0
    modified = False
    
    for i, line in enumerate(lines):
        if skip_next > 0:
            skip_next -= 1
            continue
            
        # Match "def normalize_id" or "def normalize("
        if re.match(r'^\s*def (normalize(?:_id)?)\(', line):
            # Look ahead for "return ...stem"
            found_return = False
            for j in range(i, min(i+10, len(lines))):
                if "return" in lines[j] and ".stem" in lines[j]:
                    found_return = True
                    # Find where function ends (next def or end of file)
                    end_idx = j
                    for k in range(j+1, min(j+5, len(lines))):
                         if "return" in lines[k] and ".stem" in lines[k]:
                             end_idx = k
                    skip_next = end_idx - i
                    break
            
            if found_return:
                new_lines.append(NEW_FUNC)
                modified = True
                continue
        
        new_lines.append(line)

    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return True
    return False

TARGET_DIR = "/data/brhanu/thesis_project/scripts/core_pipeline"
for filename in os.listdir(TARGET_DIR):
    if filename.endswith(".py") and filename != "patch_normalization.py":
        if patch_file(os.path.join(TARGET_DIR, filename)):
            print(f"✅ Patched {filename}")
