import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

# ==============================
# CONFIGURATION
# ==============================
BASE_DIR = "/data/brhanu/thesis_project"
MODEL_ID = "microsoft/Florence-2-large"
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"

# Source: Using the 231-image subset for benchmark comparison
IMAGE_DIR = os.path.join(BASE_DIR, "results/dataset_export/yolo11_dataset_v3/images")
OUTPUT_JSON = os.path.join(BASE_DIR, "results/florence2_detections_v1.json")

# Taxonomy v2.0
TAXONOMY = [
    "person", "child", "animal", "building", "weapon", "vehicle", 
    "tree", "text", "hat", "furniture"
]

# Mapping from Florence-2 generic labels to our taxonomy
LABEL_MAPPING = {
    "man": "person", "woman": "person", "person": "person", "people": "person", "human face": "person",
    "child": "child", "boy": "child", "girl": "child",
    "horse": "animal", "cow": "animal", "dog": "animal", "bird": "animal", "cat": "animal", "animal": "animal",
    "building": "building", "house": "building", "tower": "building", "castle": "building", "window": "building", "door": "building",
    "weapon": "weapon", "gun": "weapon", "sword": "weapon", "cannon": "weapon",
    "vehicle": "vehicle", "car": "vehicle", "carriage": "vehicle", "boat": "vehicle", "bicycle": "vehicle", "train": "vehicle",
    "tree": "tree", "plant": "tree", "forest": "tree", "flower": "tree",
    "text": "text", "writing": "text", "inscription": "text", "book": "text", "paper": "text",
    "hat": "hat", "cap": "hat", "bonnet": "hat", "helmet": "hat", "turban": "hat",
    "furniture": "furniture", "table": "furniture", "chair": "furniture", "desk": "furniture", "bed": "furniture", "bench": "furniture", "shelf": "furniture"
}

# ==============================
# LOAD MODEL
# ==============================
print(f"🚀 Loading Florence-2 model ({MODEL_ID}) on {DEVICE}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True, 
    attn_implementation="eager"
).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# ==============================
# INFERENCE LOOP
# ==============================
all_results = []
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"🔥 Running Florence-2 inference on {len(image_files)} images...")

for img_name in tqdm(image_files):
    img_path = os.path.join(IMAGE_DIR, img_name)
    try:
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        
        # We use <OD> for general object detection
        task_prompt = '<OD>'
        inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
                use_cache=False # Crucial for current transformers/remote code compatibility
            )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(w, h))
        
        detections = []
        # Florence-2 output format for <OD>: {'<OD>': {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['label1', ...]}}
        data = parsed_answer.get('<OD>', {})
        bboxes = data.get('bboxes', [])
        labels = data.get('labels', [])
        
        for box, label in zip(bboxes, labels):
            label_clean = label.lower().strip()
            # Map to our taxonomy
            mapped_label = None
            if label_clean in LABEL_MAPPING:
                mapped_label = LABEL_MAPPING[label_clean]
            else:
                # Partial match check
                for key in LABEL_MAPPING:
                    if key in label_clean:
                        mapped_label = LABEL_MAPPING[key]
                        break
            
            if mapped_label:
                # Normalize box to [0, 1]
                x1, y1, x2, y2 = box
                norm_box = [x1/w, y1/h, x2/w, y2/h]
                detections.append({
                    "label": mapped_label,
                    "box": norm_box,
                    "confidence": 0.9,  # Florence-2 doesn't provide per-box confidence in text output easily
                    "original_label": label_clean
                })
        
        all_results.append({
            "image": img_name,
            "detections": detections
        })

    except Exception as e:
        print(f"⚠️ Error processing {img_name}: {e}")

# ==============================
# SAVE RESULTS
# ==============================
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"✅ Florence-2 detections saved to {OUTPUT_JSON}")
