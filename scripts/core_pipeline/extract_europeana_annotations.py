#!/usr/bin/env python3
"""
extract_europeana_annotations.py
----------------------------------
Extracts completed human annotations for the GENUINE Europeana pilot project
(see setup_europeana_labelstudio.py) from the Label Studio SQLite database.

REWRITTEN during a full-project reproducibility audit: the previous version
hardcoded project_id=2 and parsed filenames as Colibri-style
"PPN.../shortid.jpg" -- appropriate for the fabricated pilot it was built
for (ordinary Colibri images mislabeled "Europeana"), not for genuine
Europeana images, whose filenames are Europeana record IDs. Looks up the
project by title now, and parses the real filename format
("europeana_real_<image_id>.jpg").

Outputs:
  results/external_archive/europeana_real_pilot/europeana_human_annotations_REAL.csv
"""

import sqlite3
import json
import pandas as pd
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")
DB_PATH = Path("/home/brhanu/.local/share/label-studio/label_studio.sqlite3")
OUTPUT_CSV = BASE / "results/external_archive/europeana_real_pilot/europeana_human_annotations_REAL.csv"
PROJECT_TITLE = "Europeana REAL Pilot (genuine API images)"

# Object classes exactly as defined in this project's Label Studio label_config
# (verified against the stored config, not assumed). Order fixes the class index.
CLASS_MAPPING = {
    "person": 0, "child": 1, "animal": 2, "building": 3, "weapon": 4,
    "vehicle": 5, "tree": 6, "text": 7, "hat": 8, "furniture": 9,
}

# The `scene` control offers teaching/family/playing/landscape/drawing and is
# optional with no explicit "none" choice -- so an unset scene means the rater
# saw no indexable scene, which is what has_scene=0 records below.

def main():
    print("=" * 70)
    print("EXTRACTING GENUINE EUROPEANA HUMAN ANNOTATIONS FROM LABEL STUDIO")
    print("=" * 70)

    if not DB_PATH.exists():
        print(f"❌ SQLite database not found at: {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM project WHERE title = ?", (PROJECT_TITLE,))
    proj_row = cursor.fetchone()
    if not proj_row:
        print(f"❌ Project '{PROJECT_TITLE}' not found. Run setup_europeana_labelstudio.py first.")
        conn.close()
        return
    project_id = proj_row[0]

    cursor.execute("""
        SELECT t.id, t.data, tc.result, tc.created_at
        FROM task t
        JOIN task_completion tc ON t.id = tc.task_id
        WHERE t.project_id = ?
        ORDER BY t.id ASC
    """, (project_id,))

    rows = cursor.fetchall()
    print(f"Loaded {len(rows)} annotated tasks from project {project_id} ('{PROJECT_TITLE}').")

    if len(rows) == 0:
        print("⚠️ No annotations found yet. Please make sure to complete and submit labeling in Label Studio!")
        conn.close()
        return

    parsed_records = []
    
    for task_id, data_str, result_str, annotated_at in rows:
        data = json.loads(data_str)
        result = json.loads(result_str)
        
        # Get filename and convert back to image_id
        img_url = data.get('image', '')
        filename = img_url.split('/')[-1]

        # Format "europeana_real_<image_id>.jpg" -> "<image_id>"
        stem = filename.rsplit('.', 1)[0]
        image_id = stem[len("europeana_real_"):] if stem.startswith("europeana_real_") else stem
        
        bboxes = []
        scenes = []
        confidence = "5 - Certain"
        clarity = "Clear"
        
        for item in result:
            from_name = item.get('from_name')
            item_type = item.get('type')
            value = item.get('value', {})
            
            if item_type == 'rectanglelabels':
                labels = value.get('rectanglelabels', [])
                if labels:
                    label = labels[0].lower()
                    if label in CLASS_MAPPING:
                        class_idx = CLASS_MAPPING[label]
                        w_norm = value.get('width', 0) / 100.0
                        h_norm = value.get('height', 0) / 100.0
                        x_center = (value.get('x', 0) + value.get('width', 0) / 2.0) / 100.0
                        y_center = (value.get('y', 0) + value.get('height', 0) / 2.0) / 100.0
                        bboxes.append((class_idx, x_center, y_center, w_norm, h_norm))
            elif item_type == 'choices':
                choices = value.get('choices', [])
                if choices:
                    choice_val = choices[0]
                    if from_name == 'scene':
                        scenes.append(choice_val.lower())
                    elif from_name == 'confidence':
                        confidence = choice_val
                    elif from_name == 'clarity':
                        clarity = choice_val
        
        parsed_records.append({
            'task_id': task_id,
            'image_id': image_id,
            'filename': filename,
            'scene': scenes[0] if scenes else '',
            'has_scene': 1 if scenes and scenes[0] != 'none' else 0,
            'confidence': confidence,
            'clarity': clarity,
            'num_boxes': len(bboxes),
            'annotated_at': annotated_at
        })
        
    df = pd.DataFrame(parsed_records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"🎉 Successfully exported {len(df)} Europeana annotations to {OUTPUT_CSV}")
    print(df[['image_id', 'scene', 'has_scene', 'num_boxes']].head(10).to_string())
    
    conn.close()

if __name__ == '__main__':
    main()
