import gradio as gr
import os
import json
import shutil
from PIL import Image, ImageDraw
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================
BASE_DIR = "/data/brhanu/thesis_project/results/self_training/student_iter_3"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
LABELS_DIR = os.path.join(BASE_DIR, "labels")
VERIFIED_DIR = os.path.join(BASE_DIR, "verified")
KOSMOS_JSONL = "/data/brhanu/thesis_project/results/kosmos_grounding.jsonl"

os.makedirs(VERIFIED_DIR, exist_ok=True)
os.makedirs(os.path.join(VERIFIED_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(VERIFIED_DIR, "labels"), exist_ok=True)

# Taxonomy
TAXONOMY = [
    "person", "child", "horse", "building", "weapon", "vehicle", 
    "tree", "clothing", "text", "animal"
]

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
]

# Load Kosmos Metadata
kosmos_data = {}
if os.path.exists(KOSMOS_JSONL):
    with open(KOSMOS_JSONL, "r") as f:
        for line in f:
            obj = json.loads(line)
            kosmos_data[obj["image"]] = obj.get("kosmos_output", "No output available.")

# Get list of images with labels (using the ones that have refined labels)
all_labels = sorted([f for f in os.listdir(LABELS_DIR) if f.endswith(".txt")])
all_images = [f.replace(".txt", ".jpg") for f in all_labels] # Adjusted for student results structure

# ==============================
# DRAWING HELPERS
# ==============================
def draw_boxes(image_path, label_path):
    try:
        img = Image.open(image_path).convert("RGB")
    except:
        return Image.new("RGB", (512, 512), color="gray")
        
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    
                    # YOLO -> Pixel coords
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    
                    color = COLORS[cls_id % len(COLORS)]
                    label = TAXONOMY[cls_id] if cls_id < len(TAXONOMY) else f"class_{cls_id}"
                    
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    draw.text((x1, y1 - 10), label, fill=color)
    
    return img

# ==============================
# GRADIO LOGIC
# ==============================
def load_image_at(idx):
    if 0 <= idx < len(all_images):
        img_name = all_images[idx]
        # Images are symlinked in student_iter_3/images
        img_path = os.path.join(IMAGES_DIR, img_name)
        label_path = os.path.join(LABELS_DIR, os.path.splitext(img_name)[0] + ".txt")
        
        status = f"Image {idx + 1} of {len(all_images)}: {img_name}"
        if os.path.exists(os.path.join(VERIFIED_DIR, "images", img_name)):
            status += " (ALREADY VERIFIED)"
            
        kosmos_text = kosmos_data.get(img_name, "No Kosmos metadata found for this image.")
            
        return draw_boxes(img_path, label_path), status, kosmos_text
    return None, "Out of bounds", ""

def on_confirm(idx):
    if 0 <= idx < len(all_images):
        img_name = all_images[idx]
        shutil.copy2(os.path.join(IMAGES_DIR, img_name), os.path.join(VERIFIED_DIR, "images", img_name))
        
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_src = os.path.join(LABELS_DIR, label_name)
        if os.path.exists(label_src):
            shutil.copy2(label_src, os.path.join(VERIFIED_DIR, "labels", label_name))
            
        return idx + 1
    return idx

def on_reject(idx):
    return idx + 1

# ==============================
# UI DESIGN
# ==============================
with gr.Blocks(title="Final Student Results HITL Tool") as demo:
    gr.Markdown("# 🎨 Final Student Results Validator (Iteration 3)")
    gr.Markdown("Review the self-trained labels and Kosmos-2.5 metadata. **Confirm** only high-quality labels for the Gold Standard.")
    
    with gr.Row():
        with gr.Column(scale=2):
            viewer = gr.Image(label="Student Iter 3 Detections", type="pil")
        with gr.Column(scale=1):
            kosmos_box = gr.Textbox(label="Kosmos-2.5 Metadata (OCR/Layout)", lines=15, interactive=False)
    
    with gr.Row():
        status_box = gr.Textbox(label="Status", value="", interactive=False)
    
    with gr.Row():
        btn_prev = gr.Button("⬅️ Previous")
        btn_reject = gr.Button("❌ Reject / Skip", variant="secondary")
        btn_confirm = gr.Button("✅ Confirm & Next", variant="primary")
        btn_next = gr.Button("➡️ Next")

    state_idx = gr.State(0)

    def update_view(idx):
        img, status, kosmos = load_image_at(idx)
        return img, status, kosmos, idx

    def handle_prev(idx):
        new_idx = max(0, idx - 1)
        return update_view(new_idx)

    def handle_next(idx):
        new_idx = min(len(all_images) - 1, idx + 1)
        return update_view(new_idx)

    def handle_confirm(idx):
        after_idx = on_confirm(idx)
        new_idx = min(len(all_images) - 1, after_idx)
        return update_view(new_idx)

    def handle_reject(idx):
        after_idx = on_reject(idx)
        new_idx = min(len(all_images) - 1, after_idx)
        return update_view(new_idx)

    # Init
    demo.load(fn=update_view, inputs=state_idx, outputs=[viewer, status_box, kosmos_box, state_idx])

    btn_prev.click(fn=handle_prev, inputs=state_idx, outputs=[viewer, status_box, kosmos_box, state_idx])
    btn_next.click(fn=handle_next, inputs=state_idx, outputs=[viewer, status_box, kosmos_box, state_idx])
    btn_confirm.click(fn=handle_confirm, inputs=state_idx, outputs=[viewer, status_box, kosmos_box, state_idx])
    btn_reject.click(fn=handle_reject, inputs=state_idx, outputs=[viewer, status_box, kosmos_box, state_idx])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
