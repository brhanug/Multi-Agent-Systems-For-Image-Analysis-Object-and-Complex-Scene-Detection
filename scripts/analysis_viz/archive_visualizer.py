
import gradio as gr
import pandas as pd
import os
from PIL import Image

# Config
MANIFEST_PATH = "/data/brhanu/thesis_project/final_dataset_v1_refresh/metadata/manifest_v1.csv"
IMAGES_DIR = "/data/brhanu/thesis_project/final_dataset_v1_refresh/images/restored"

def load_data():
    if not os.path.exists(MANIFEST_PATH):
        return None
    df = pd.read_csv(MANIFEST_PATH)
    return df

df = load_data()

def get_image_data(idx):
    if df is None: return None, "Manifest not found", "", ""
    row = df.iloc[idx]
    img_path = os.path.join(IMAGES_DIR, row['image_id'])
    
    # Extract object list and scene type
    vqa_objs = row['vqa_objects']
    vqa_scene = row['vqa_scene']
    triplets = row['scene_triplets']
    
    try:
        img = Image.open(img_path)
    except:
        img = None
        
    status = f"Displaying {idx+1}/{len(df)}: {row['image_id']}"
    return img, status, vqa_objs, vqa_scene, triplets

with gr.Blocks(title="Visual Historian Archive Explorer") as demo:
    gr.Markdown("# 🏛️ Visual Historian: Archive Metadata Explorer")
    gr.Markdown("Use this tool to verify the **Object-Level Detection** and **Scene-Level Taxonomy** assignments for the full collection.")
    
    with gr.Row():
        with gr.Column(scale=2):
            img_disp = gr.Image(label="Restored Archival Image")
        with gr.Column(scale=1):
            status = gr.Textbox(label="Status", interactive=False)
            obj_box = gr.Textbox(label="Detected Objects (VQA-O)", lines=3)
            scene_box = gr.Textbox(label="Scene Taxonomy (VQA-S)", lines=2)
            triplet_box = gr.Textbox(label="Relational Triplets (Scene Graph)", lines=5)

    with gr.Row():
        prev_btn = gr.Button("⬅️ Previous")
        next_btn = gr.Button("Next ➡️")
        slider = gr.Slider(0, len(df)-1 if df is not None else 0, step=1, label="Jump to Image Index")

    curr_idx = gr.State(0)

    def update(idx):
        img, stat, objs, scn, trp = get_image_data(idx)
        return img, stat, objs, scn, trp, idx

    next_btn.click(lambda i: min(len(df)-1, i+1), inputs=curr_idx, outputs=curr_idx).then(
        update, inputs=curr_idx, outputs=[img_disp, status, obj_box, scene_box, triplet_box, curr_idx]
    )
    prev_btn.click(lambda i: max(0, i-1), inputs=curr_idx, outputs=curr_idx).then(
        update, inputs=curr_idx, outputs=[img_disp, status, obj_box, scene_box, triplet_box, curr_idx]
    )
    slider.change(update, inputs=slider, outputs=[img_disp, status, obj_box, scene_box, triplet_box, curr_idx])

    # Initial load
    demo.load(update, inputs=curr_idx, outputs=[img_disp, status, obj_box, scene_box, triplet_box, curr_idx])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862, share=True)

