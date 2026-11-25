#!/usr/bin/env python3
"""
setup_europeana_labelstudio.py
--------------------------------
Initializes a Label Studio project for the GENUINE Europeana pilot images
(see run_europeana_real_pilot_download.py) so a human can provide real
ground-truth (has_scene) labels.

REWRITTEN during a full-project reproducibility audit: the previous version
of this script did not use any real Europeana data -- it sampled ordinary
Colibri images (any image outside the first reviewed batch) and copied
those into a project titled "Europeana 30-Image Evaluation," which is how
the fabricated n=30 "Europeana" pilot entered the thesis. This version
copies the actual images downloaded live from the real Europeana API.
"""

import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
DB_PATH = Path("/home/brhanu/.local/share/label-studio/label_studio.sqlite3")
PILOT_DIR = BASE / "results" / "external_archive" / "europeana_real_pilot"
METADATA_CSV = PILOT_DIR / "metadata.csv"
IMAGE_DIR = PILOT_DIR / "images"
CORS_DIR = BASE / "cvat_upload_temp" / "obj_train_data"
PROJECT_TITLE = "Europeana REAL Pilot (genuine API images)"


def main():
    print("=" * 70)
    print("SETTING UP GENUINE EUROPEANA PILOT PROJECT IN LABEL STUDIO")
    print("=" * 70)

    if not METADATA_CSV.exists():
        print(f"❌ {METADATA_CSV} not found -- run run_europeana_real_pilot_download.py first.")
        return
    matched = pd.read_csv(METADATA_CSV)
    print(f"  {len(matched)} genuine Europeana images to annotate.")

    # ── Copy real images to the CORS serving directory ──
    CORS_DIR.mkdir(parents=True, exist_ok=True)
    for _, row in matched.iterrows():
        src_path = IMAGE_DIR / f"{row['image_id']}.jpg"
        if not src_path.exists():
            print(f"    ⚠ Missing downloaded image: {src_path}")
            continue
        dest_filename = f"europeana_real_{row['image_id']}.jpg"
        dest_path = CORS_DIR / dest_filename
        if not dest_path.exists():
            shutil.copy(src_path, dest_path)

    # ── DB setup ──
    if not DB_PATH.exists():
        print(f"❌ Label Studio DB not found at {DB_PATH}. Start Label Studio once first.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM project WHERE id = 1")
    p1 = c.fetchone()
    if not p1:
        print("  ❌ Error: Project 1 not found in Label Studio database (used as a config template).")
        return

    c.execute("SELECT id FROM project WHERE title = ?", (PROJECT_TITLE,))
    proj_row = c.fetchone()

    if proj_row:
        project_id = proj_row["id"]
        print(f"  Project '{PROJECT_TITLE}' already exists with ID: {project_id}")
    else:
        print("  Creating new Label Studio project...")
        p_dict = dict(p1)
        p_dict.pop("id")
        p_dict["title"] = PROJECT_TITLE
        p_dict["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        p_dict["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        columns = ", ".join(p_dict.keys())
        placeholders = ", ".join(["?"] * len(p_dict))
        c.execute(f"INSERT INTO project ({columns}) VALUES ({placeholders})", list(p_dict.values()))
        project_id = c.lastrowid
        conn.commit()
        print(f"    Created project with ID: {project_id}")

    c.execute("SELECT COUNT(*) FROM task WHERE project_id = ?", (project_id,))
    existing_tasks = c.fetchone()[0]
    print(f"  Existing tasks in project: {existing_tasks}")

    if existing_tasks >= len(matched):
        print("  All tasks already present. Setup is complete.")
    else:
        print("  Inserting genuine Europeana tasks...")
        for t_idx, (_, row) in enumerate(matched.iterrows(), 1):
            dest_filename = f"europeana_real_{row['image_id']}.jpg"
            image_url = f"http://localhost:9000/{dest_filename}"
            task_data = json.dumps({
                "image": image_url,
                "title": row.get("title", ""),
                "data_provider": row.get("data_provider", ""),
            })
            c.execute("SELECT id FROM task WHERE project_id = ? AND data LIKE ?", (project_id, f"%{dest_filename}%"))
            if not c.fetchone():
                c.execute(
                    "INSERT INTO task (data, meta, project_id, created_at, updated_at, is_labeled, overlap, total_annotations, cancelled_annotations, total_predictions, comment_count, unresolved_comment_count, allow_skip, inner_id) "
                    "VALUES (?, '{}', ?, datetime('now'), datetime('now'), 0, 1, 0, 0, 0, 0, 0, 1, ?)",
                    (task_data, project_id, t_idx),
                )
        conn.commit()
        c.execute("SELECT COUNT(*) FROM task WHERE project_id = ?", (project_id,))
        print(f"  Setup complete! Total tasks in project: {c.fetchone()[0]}")

    conn.close()
    print(f"\n➡️  Open Label Studio and annotate project '{PROJECT_TITLE}' to complete the real Europeana pilot.")


if __name__ == "__main__":
    main()
