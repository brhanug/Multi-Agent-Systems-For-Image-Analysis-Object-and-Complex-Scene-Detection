#!/usr/bin/env python3
import os
import json
import csv
from tqdm import tqdm

# === CONFIG ===
BASE_DIR = "/data/brhanu/thesis_project"
RESULTS_V1 = os.path.join(BASE_DIR, "results_v1")
FLORENCE_DIR = os.path.join(RESULTS_V1, "florence2")
DINO_DIR = os.path.join(RESULTS_V1, "groundingdino")
LLAVA_DIR = os.path.join(RESULTS_V1, "llava")
KOSMOS_DIR = os.path.join(RESULTS_V1, "kosmos")
DEST_DIR = os.path.join(BASE_DIR, "final_dataset_v1_refresh")
YOLO_LABELS_PATH = os.path.join(DEST_DIR, "labels/yolo_labels")
MANIFEST_PATH = os.path.join(DEST_DIR, "metadata/manifest_v1.csv")

TAXONOMY = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text", "hat", "furniture"]
class_to_id = {cls: idx for idx, cls in enumerate(TAXONOMY)}

def box_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def extract_from_vqa(vqa_data, kosmos_md, final_detections):
    return {
        "vqa_objects": vqa_data.get("Identify objects present: person, child, animal, building, weapon, vehicle, tree, text, hat, furniture. List only items found.", "N/A"),
        "vqa_scene": vqa_data.get("Which scene fits best: teaching, family, playing, landscape, drawing?", "N/A"),
        "scene_triplets": vqa_data.get("List relational triplets in (Subject, Predicate, Object) format for the primary interactions in this image.", "N/A"),
        "kosmos_md": kosmos_md,
        "has_label": "Yes" if final_detections else "No"
    }

def main():
    print("🔄 Consolidating v1.0 Metadata and Consensus Labels...")
    os.makedirs(YOLO_LABELS_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    
    # 1. Gather all image IDs from Florence (since it's 100% done)
    florence_files = sorted(os.listdir(FLORENCE_DIR))
    
    manifest_data = []
    
    print(f"📦 Merging {len(florence_files)} entries...")
    for f in tqdm(florence_files):
        img_id = os.path.splitext(f)[0]
        
        # Load Florence
        with open(os.path.join(FLORENCE_DIR, f), 'r') as j:
            flo_data = json.load(j)
        
        # Load DINO (if exists)
        dino_data = {}
        dino_path = os.path.join(DINO_DIR, f)
        if os.path.exists(dino_path):
            with open(dino_path, 'r') as j:
                dino_data = json.load(j)
        
        # Load LLaVA
        vqa_data = {}
        llava_path = os.path.join(LLAVA_DIR, f)
        if os.path.exists(llava_path):
            with open(llava_path, 'r') as j:
                llava_data = json.load(j)
                vqa_data = llava_data.get("responses", {})
        
        # Load Kosmos
        kosmos_md = "N/A"
        kosmos_path = os.path.join(KOSMOS_DIR, f)
        if os.path.exists(kosmos_path):
            with open(kosmos_path, 'r') as j:
                kosmos_data = json.load(j)
                kosmos_md = kosmos_data.get("kosmos_md", "N/A")

        # --- Consensus Logic ---
        final_detections = []
        if dino_data:
            # Match Florence detections against DINO for consensus
            for f_det in flo_data.get("detections", []):
                # box: [x1, y1, x2, y2]
                f_box = f_det["box"]
                f_cls = f_det["label"]
                
                # Look for support in DINO
                supported = False
                for d_det in dino_data.get("detections", []):
                    # GroundingDINO bbox: [cx, cy, w, h]
                    cx, cy, w, h = d_det["bbox"]
                    d_box = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                    
                    if d_det["label"] == f_cls and box_iou(f_box, d_box) > 0.4:
                        supported = True
                        break
                
                if supported:
                    # Convert f_box [x1, y1, x2, y2] to yolo [cx, cy, w, h]
                    x1, y1, x2, y2 = f_box
                    cx, cy, bw, bh = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
                    final_detections.append(f"{class_to_id[f_cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # Save Label File
        if final_detections:
            with open(os.path.join(YOLO_LABELS_PATH, f"{img_id}.txt"), 'w') as lf:
                lf.write("\n".join(final_detections))

        # Update Manifest Row
        row = {
            "image_id": flo_data["image"],
            **extract_from_vqa(vqa_data, kosmos_md, final_detections)
        }
        manifest_data.append(row)

    # Save Manifest
    keys = manifest_data[0].keys()
    with open(MANIFEST_PATH, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(manifest_data)

    print(f"✅ v1.0 Metadata Consolidation Complete. Manifest saved to {MANIFEST_PATH}")

if __name__ == "__main__":
    main()
