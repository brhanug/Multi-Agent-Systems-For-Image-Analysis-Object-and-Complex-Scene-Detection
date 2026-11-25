#!/usr/bin/env python3
"""
extract_self_annotation_labelstudio.py
------------------------------------------
Extracts completed annotations from the four self-annotation Label Studio
projects (see setup_self_annotation_labelstudio.py) and writes them back
into each task_*/ folder's labeling_sheet.csv format, so the existing
"when you're done, send me this CSV" pipeline in each task's README works
unchanged whether the CSV was filled in by hand or via Label Studio.

Safe to re-run at any point -- only tasks with a submitted annotation are
included; unfinished tasks are silently skipped (partial progress is fine).

Output:
  self_annotation_package/task_a_negative_controls/labeling_sheet_LS_export.csv
  self_annotation_package/task_b_contradiction_ground_truth/labeling_sheet_LS_export.csv
  self_annotation_package/task_c_cross_archive_duplicates/labeling_sheet_LS_export.csv
  self_annotation_package/task_d_ocr_ground_truth/labeling_sheet_LS_export.csv
"""
import csv
import json
import sqlite3
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")
PKG = BASE / "self_annotation_package"
DB_PATH = Path("/home/brhanu/.local/share/label-studio/label_studio.sqlite3")

PROJECTS = {
    "Self-Annotation A: Negative Control Blind Spot": ("task_a_negative_controls", "a"),
    "Self-Annotation B: Agent 2 Contradiction Ground Truth": ("task_b_contradiction_ground_truth", "b"),
    "Self-Annotation C: Agent 4 Cross-Archive Duplicate Pairs": ("task_c_cross_archive_duplicates", "c"),
    "Self-Annotation D: Agent 5 OCR Ground Truth": ("task_d_ocr_ground_truth", "d"),
}


def get_choice(result, from_name):
    for item in result:
        if item.get("from_name") == from_name and item.get("type") == "choices":
            choices = item.get("value", {}).get("choices", [])
            if choices:
                return choices[0]
    return ""


def get_text(result, from_name):
    for item in result:
        if item.get("from_name") == from_name and item.get("type") == "textarea":
            texts = item.get("value", {}).get("text", [])
            if texts:
                return texts[0]
    return ""


def main():
    if not DB_PATH.exists():
        print(f"Label Studio DB not found at {DB_PATH}")
        return
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    for title, (folder, letter) in PROJECTS.items():
        c.execute("SELECT id FROM project WHERE title = ?", (title,))
        row = c.fetchone()
        if not row:
            print(f"Project '{title}' not found -- run setup_self_annotation_labelstudio.py first.")
            continue
        project_id = row[0]

        c.execute("""
            SELECT t.data, tc.result
            FROM task t JOIN task_completion tc ON t.id = tc.task_id
            WHERE t.project_id = ?
            ORDER BY t.id ASC
        """, (project_id,))
        rows = c.fetchall()
        print(f"{title}: {len(rows)} completed annotations")
        if not rows:
            continue

        out_rows = []
        for data_str, result_str in rows:
            data = json.loads(data_str)
            result = json.loads(result_str)
            if letter == "a":
                out_rows.append({"cvat_id": data.get("cvat_id"), "your_label": get_choice(result, "scene")})
            elif letter == "b":
                out_rows.append({"id": data.get("row_id"), "claimed_noun": data.get("claimed_noun"),
                                  "your_judgment": get_choice(result, "judgment")})
            elif letter == "c":
                out_rows.append({"pair_id": data.get("pair_id"), "similarity": data.get("similarity"),
                                  "your_judgment": get_choice(result, "judgment")})
            elif letter == "d":
                out_rows.append({"id": data.get("row_id"), "ground_truth_transcription": get_text(result, "transcription")})

        out_csv = PKG / folder / "labeling_sheet_LS_export.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"  Wrote {out_csv}")

    conn.close()


if __name__ == "__main__":
    main()
