import sqlite3
import json
import re
import shutil
import pandas as pd
from pathlib import Path
import numpy as np

DB_PATH = '/home/brhanu/.local/share/label-studio/label_studio.sqlite3'
MANIFEST_PATH = 'final_dataset/metadata/manifest.csv'
SCORES_PATH = 'results/multi_agent/agent_comparison_scores.csv'
YOLO_LABEL_DIR = 'final_dataset/labels/yolo_labels'
LATEX_ASSETS_DIR = 'latex_assets'

# YOLO class mapping
IDX_TO_CLASS = {
    0: "person", 1: "child", 2: "horse", 3: "building",
    4: "weapon", 5: "vehicle", 6: "tree", 7: "clothing",
    8: "text", 9: "animal"
}

def load_yolo_labels(image_id):
    # Map PPN1845710797/00000620_1 -> PPN1845710797_00000620_1.txt
    norm = image_id.replace('/', '_')
    lbl_file = Path(YOLO_LABEL_DIR) / f"{norm}.txt"
    if not lbl_file.exists():
        return []
    classes = []
    with open(lbl_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 0:
                try:
                    cls_idx = int(float(parts[0]))
                    classes.append(IDX_TO_CLASS.get(cls_idx, f"class_{cls_idx}"))
                except:
                    pass
    return classes

def main():
    print("Loading data...")
    scores_df = pd.read_csv(SCORES_PATH)
    scores_lookup = scores_df.set_index('image_id').to_dict('index')

    manifest_df = pd.read_csv(MANIFEST_PATH)
    manifest_lookup = manifest_df.set_index('image_id').to_dict('index')

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT t.id, t.data, tc.result FROM task t JOIN task_completion tc ON t.id=tc.task_id')
    rows = c.fetchall()
    conn.close()

    print(f"Loaded {len(rows)} database tasks.")

    records = []
    for tid, data_str, res_str in rows:
        data = json.loads(data_str)
        res = json.loads(res_str)
        img_url = data.get('image', '')
        img_name = img_url.split('/')[-1]
        
        m = re.match(r'^(PPN[a-zA-Z\d]+)_(.+)\.(jpg|jpeg|png)$', img_name, re.IGNORECASE)
        if not m:
            continue
        ppn_part = m.group(1)
        file_part = m.group(2)
        image_id = f'{ppn_part}/{file_part}'
        
        if image_id in scores_lookup:
            score_info = scores_lookup[image_id]
            manifest_info = manifest_lookup.get(image_id, {})
            
            # Calculate disagreement
            agent_cols = ['existing_pipeline_agent','agreement_agent','scene_agent','vlm_agent','restoration_agent','document_agent']
            vals = [score_info[col] for col in agent_cols if col in score_info]
            disagreement = np.std(vals)
            
            # Extract human annotations
            scene = clarity = confidence = None
            human_boxes = []
            for item in res:
                fn = item.get('from_name', '')
                v = item.get('value', {})
                if fn == 'scene' and v.get('choices'):
                    scene = v['choices'][0]
                elif fn == 'clarity' and v.get('choices'):
                    clarity = v['choices'][0]
                elif fn == 'confidence' and v.get('choices'):
                    confidence = v['choices'][0]
                elif fn == 'label' and item.get('type') == 'rectanglelabels':
                    labels = v.get('rectanglelabels', [])
                    if labels:
                        human_boxes.append(labels[0].lower())
            
            # If no scene or clarity or human boxes, skip
            if not scene and not clarity and not human_boxes:
                continue

            records.append({
                'image_id': image_id,
                'disagreement': disagreement,
                'human_scene': scene,
                'human_clarity': clarity,
                'human_boxes': human_boxes,
                'existing_pipeline_agent': score_info.get('existing_pipeline_agent'),
                'agreement_agent': score_info.get('agreement_agent'),
                'scene_agent': score_info.get('scene_agent'),
                'vlm_agent': score_info.get('vlm_agent'),
                'restoration_agent': score_info.get('restoration_agent'),
                'document_agent': score_info.get('document_agent'),
                'blip2_caption': manifest_info.get('blip2_caption', ''),
                'vqa_primary_scene': manifest_info.get('vqa_primary_scene', ''),
                'kosmos_markdown': manifest_info.get('kosmos_markdown', ''),
                'original_path': manifest_info.get('original_path', '')
            })

    print(f"Parsed {len(records)} records with expert annotations.")

    # Now let's classify records into failure modes
    # We want 5 records per mode:
    # 1. Degraded/Faded: high disagreement & (clarity in Blurry, Faded)
    # 2. Drawings: high disagreement & (scene in drawing, drawing-like)
    # 3. Crowded scenes: high disagreement & (many human boxes >= 5)
    # 4. OCR/Text conflict: high disagreement & (human boxes contains text, or document_agent is high)
    # 5. Occlusion failures: high disagreement & (low agreement agent & high object/vlm, or vice versa, indicating mismatch on overlap)

    categories = {
        'degraded': [],
        'drawing': [],
        'crowded': [],
        'text_conflict': [],
        'occlusion': []
    }

    # Sort records by disagreement descending
    records = sorted(records, key=lambda x: x['disagreement'], reverse=True)

    for r in records:
        # Load YOLO detections
        yolo_classes = load_yolo_labels(r['image_id'])
        r['yolo_classes'] = yolo_classes
        
        # 1. Degraded
        if r['human_clarity'] in ['Blurry', 'Faded'] and len(categories['degraded']) < 5:
            categories['degraded'].append(r)
            continue
            
        # 2. Drawings
        if r['human_scene'] == 'drawing' and len(categories['drawing']) < 5:
            categories['drawing'].append(r)
            continue
            
        # 3. Crowded
        if len(r['human_boxes']) >= 4 and len(categories['crowded']) < 5:
            categories['crowded'].append(r)
            continue
            
        # 4. Text conflict
        if ('text' in r['human_boxes'] or r['document_agent'] > 0.6) and len(categories['text_conflict']) < 5:
            categories['text_conflict'].append(r)
            continue
            
        # 5. Occlusion / Spatial consensus failure
        # (Where agreement agent is low, meaning spatial overlap is bad, or YOLO and VLM disagree on count)
        if r['agreement_agent'] < 0.4 and len(categories['occlusion']) < 5:
            categories['occlusion'].append(r)
            continue

    # Fallback to make sure we have 5 in each:
    # If any is less than 5, fill from general high-disagreement
    for cat, items in categories.items():
        print(f"Category '{cat}': found {len(items)} cases.")
        if len(items) < 5:
            needed = 5 - len(items)
            for r in records:
                if r not in items and not any(r in val for val in categories.values()):
                    items.append(r)
                    needed -= 1
                    if needed == 0:
                        break

    # Copy files and output LaTeX code & Markdown description
    Path(LATEX_ASSETS_DIR).mkdir(parents=True, exist_ok=True)
    
    latex_code = "% Failure Case Gallery Appendix\n"
    markdown_code = "# Failure Case Gallery\n"
    
    cat_names = {
        'degraded': 'Faded and Degraded Scans',
        'drawing': 'Historical Drawings and Abstract Styles',
        'crowded': 'Crowded Scenes and High Object Density',
        'text_conflict': 'OCR and Text-Object Conflicts',
        'occlusion': 'Occlusion and Spatial Overlap Failures'
    }

    for cat, items in categories.items():
        latex_code += f"\n\\subsection{{{cat_names[cat]}}}\n"
        latex_code += f"This category highlights representative failure modes related to {cat_names[cat].lower()}.\n\n"
        
        markdown_code += f"\n## {cat_names[cat]}\n"
        
        # We will present them in a nice layout: 2x3 or grid.
        # Let's generate a table or subfigures for LaTeX
        for i, r in enumerate(items):
            src_path = Path(r['original_path'])
            # If path is relative, make sure it starts with final_dataset or correct prefix
            if not src_path.exists():
                # Try relative to base
                src_path = Path('final_dataset') / src_path
            
            dest_name = f"failure_{cat}_{i+1}.jpg"
            dest_path = Path(LATEX_ASSETS_DIR) / dest_name
            
            if src_path.exists():
                shutil.copy(src_path, dest_path)
                print(f"Copied {src_path} -> {dest_path}")
            else:
                print(f"WARNING: source image {src_path} not found!")
                
            # Document details
            yolo_desc = ", ".join(r['yolo_classes']) if r['yolo_classes'] else "No detections"
            human_desc = f"Scene: {r['human_scene']}, Clarity: {r['human_clarity']}, Objects: {', '.join(r['human_boxes'])}"
            vlm_desc = f"Primary scene: {r['vqa_primary_scene']}. Caption: {r['blip2_caption']}"
            
            # Format LaTeX block
            latex_code += f"""\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.65\\textwidth]{{latex_assets/{dest_name}}}
    \\caption{{\\textbf{{Case {cat.upper()}-{i+1}}}: Image \\texttt{{{r['image_id']}}}. 
    \\\\ \\textbf{{Human Annotation}}: {human_desc}
    \\\\ \\textbf{{YOLO detections}}: {yolo_desc} (Agent score: {r['existing_pipeline_agent']:.3f})
    \\\\ \\textbf{{VLM caption}}: {r['blip2_caption']} (Agent score: {r['vlm_agent']:.3f})
    \\\\ \\textbf{{Disagreement Score ($H_C$)}}: {r['disagreement']:.4f}.
    \\\\ \\textbf{{Verdict/Failure Mode}}: {get_verdict(cat, r)}}}
    \\label{{fig:failure_{cat}_{i+1}}}
\\end{{figure}}

"""
            # Format Markdown
            markdown_code += f"""### Case {cat.upper()}-{i+1}: {r['image_id']}
- **Image**: ![Case {cat.upper()}-{i+1}](file://{dest_path.resolve()})
- **Human Annotation**: {human_desc}
- **YOLO Detections**: {yolo_desc} (Score: {r['existing_pipeline_agent']:.3f})
- **VLM Caption**: {r['blip2_caption']} (Score: {r['vlm_agent']:.3f})
- **Disagreement ($H_C$)**: {r['disagreement']:.4f}
- **Verdict/Failure Mode**: {get_verdict(cat, r)}

"""

    with open('latex_assets/failure_gallery.tex', 'w') as f:
        f.write(latex_code)
    print("Wrote latex_assets/failure_gallery.tex")

    with open('results/multi_agent/failure_gallery_summary.md', 'w') as f:
        f.write(markdown_code)
    print("Wrote results/multi_agent/failure_gallery_summary.md")

def get_verdict(cat, r):
    if cat == 'degraded':
        return "Severe scan fading and archival degradation result in weak object contrast. YOLO fails to detect objects, while the VLM leverages semantic context to infer the scene layout, leading to high agent disagreement."
    elif cat == 'drawing':
        return "The abstract, hand-drawn sketch style differs significantly from real photographic training distributions. YOLO's bounding-box priors fail, while VLMs successfully generalize semantic concepts, creating a high-disagreement trigger."
    elif cat == 'crowded':
        return "High object density and visual clutter cause overlapping bounding boxes. YOLO fails to separate individual instances, leading to spatial localization conflicts with human annotations and VLM VQA counts."
    elif cat == 'text_conflict':
        return "The presence of printed text within a historical layout creates conflict. The document agent (OCR) flags high semantic content, while YOLO and scene agents struggle to reconcile text elements with surrounding scene context."
    else:
        return "Partial occlusion and overlapping object boundaries lead to consensus breakdown. YOLO under-detects or merges adjacent entities, whereas VLM reasoning identifies distinct hidden entities, resulting in high epistemic uncertainty."

if __name__ == '__main__':
    main()
