import os
import json
import numpy as np

TAXONOMY = [
    "person", "child", "animal", "building", "weapon", "vehicle", 
    "tree", "text", "hat", "furniture"
]
class_to_id = {cls: idx for idx, cls in enumerate(TAXONOMY)}

def box_iou(box1, box2):
    """
    Standard IoU calculation. Boxes are [x1, y1, x2, y2].
    """
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def convert_xywh_to_xyxy(box):
    """[cx, cy, w, h] -> [x1, y1, x2, y2]"""
    cx, cy, w, h = box
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

class PseudoLabelRefiner:
    def __init__(self, florence_json_path, dino_dir):
        print(f"📦 Initializing Refiner with Florence-2 weights from {florence_json_path}")
        with open(florence_json_path, "r") as f:
            florence_data = json.load(f)
        self.florence_map = {item["image"]: item["detections"] for item in florence_data}
        self.dino_dir = dino_dir

    def get_dino_detections(self, img_name):
        json_path = os.path.join(self.dino_dir, os.path.splitext(img_name)[0] + ".json")
        if not os.path.exists(json_path):
            return []
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                return data.get("detections", [])
            except:
                return []

    def refine(self, img_name, yolo_detections):
        """
        yolo_detections: list of dicts {'box': [cx, cy, w, h], 'conf': float, 'cls': int}
        Returns: list of dicts {'box': [cx, cy, w, h], 'cls': int}
        """
        refined = []
        
        # 1. Get other model detections
        florence_dets = self.florence_map.get(img_name, [])
        dino_dets = self.get_dino_detections(img_name)
        
        # Normalize other model detections to a common format: {box_xyxy, cls_id, conf}
        florence_norm = []
        for d in florence_dets:
            if d["label"] in class_to_id:
                florence_norm.append({
                    "box": d["box"], # already xyxy
                    "cls": class_to_id[d["label"]],
                    "conf": d["confidence"]
                })
        
        dino_norm = []
        for d in dino_dets:
            matched_cls = None
            dino_label = d["label"].lower()
            for tax_cls in TAXONOMY:
                if tax_cls in dino_label or (tax_cls == "animal" and "horse" in dino_label):
                    matched_cls = class_to_id[tax_cls]
                    break
            if matched_cls is not None:
                dino_norm.append({
                    "box": convert_xywh_to_xyxy(d["bbox"]),
                    "cls": matched_cls,
                    "conf": d["confidence"]
                })

        yolo_norm = []
        for d in yolo_detections:
            yolo_norm.append({
                "box": convert_xywh_to_xyxy(d["box"]),
                "cls": d["cls"],
                "conf": d["conf"],
                "orig_box": d["box"]
            })

        # --- Consensus Logic ---
        
        # We'll use YOLO detections as the primary proposals and search for support
        for y_det in yolo_norm:
            has_consensus = False
            
            # Check Florence-2 support
            for f_det in florence_norm:
                if y_det["cls"] == f_det["cls"] and box_iou(y_det["box"], f_det["box"]) > 0.5:
                    has_consensus = True
                    break
            
            # Check GroundingDINO support
            if not has_consensus:
                for d_det in dino_norm:
                    if y_det["cls"] == d_det["cls"] and box_iou(y_det["box"], d_det["box"]) > 0.5:
                        has_consensus = True
                        break
            
            # Decision
            if has_consensus:
                # Keep if supported by another model
                refined.append({"box": y_det["orig_box"], "cls": y_det["cls"]})
            elif y_det["conf"] > 0.7:
                # Keep if very high confidence even without consensus
                refined.append({"box": y_det["orig_box"], "cls": y_det["cls"]})

        # Add detections from other models if they are very confident and not already covered by YOLO
        # (This helps catch objects YOLO missed)
        for other_nets in [florence_norm, dino_norm]:
            for o_det in other_nets:
                if o_det["conf"] > 0.5:
                    # Check if YOLO already has this
                    covered = False
                    for y_det in yolo_norm:
                        if o_det["cls"] == y_det["cls"] and box_iou(o_det["box"], y_det["box"]) > 0.5:
                            covered = True
                            break
                    if not covered:
                        # Convert xyxy [x1,y1,x2,y2] back to xywh [cx,cy,w,h]
                        x1, y1, x2, y2 = o_det["box"]
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        w, h = x2 - x1, y2 - y1
                        refined.append({"box": [cx, cy, w, h], "cls": o_det["cls"]})

        return refined
