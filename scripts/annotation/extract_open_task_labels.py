#!/usr/bin/env python3
"""Pull finished labels out of Label Studio for the open tasks (A2, E, F, G, H).

Reads Label Studio's own database, so it works whether or not the server is running, and
writes one CSV per task next to that task's other files, in the same shape the chat-recorded
labels use, so both paths merge by picture id.

Also reports how long each answer took (Label Studio's lead time), which is what the
rare-class mining comparison needs: pictures found per minute, by search tactic.

    python extract_open_task_labels.py [--verify]
"""
import argparse, json, sqlite3
from pathlib import Path

import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
DB = Path.home() / ".local/share/label-studio/label_studio.sqlite3"
PKG = BASE / "self_annotation_package"
PROJECTS = {
    "Task A2 remaining: negative controls still open (465 pictures)": ("task_a2_remaining_negative_controls", "labelstudio_labels.csv"),
    "Task E: Audit of what the pipeline auto-approves (420 pictures)": ("task_e_auto_approved_audit", "labelstudio_labels.csv"),
    "Task F: What the pseudo-label class ids mean (20 boxes)": ("task_f_pseudo_label_classes", "labelstudio_labels.csv"),
    "Task G: Rare-class mining (548 pictures, no repeats)": ("task_g_rare_class_mining", "labelstudio_labels.csv"),
    "Task G v3: Rare-class mining (306 pictures: vehicle, building, weapon, teaching, family, landscape)":
        ("task_g_rare_class_mining", "labelstudio_labels_v3.csv"),
    "Task H: Second scene rating (150 pictures)": ("task_h_second_rater_scenes", "labelstudio_labels.csv"),
}


def answers(result):
    out = {}
    for item in result if isinstance(result, list) else []:
        v = item.get("value", {})
        if "choices" in v:
            out[item.get("from_name", "choice")] = "|".join(v["choices"])
        elif "text" in v:
            out[item.get("from_name", "text")] = " ".join(v["text"]) if isinstance(v["text"], list) else v["text"]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    con = sqlite3.connect(DB)
    total = {}
    for title, (folder, name) in PROJECTS.items():
        pid = con.execute("SELECT id FROM project WHERE title = ?", (title,)).fetchone()
        if not pid:
            print(f"no project: {title}"); continue
        rows = []
        for data, result, lead, created in con.execute(
                "SELECT t.data, tc.result, tc.lead_time, tc.created_at FROM task t "
                "JOIN task_completion tc ON t.id = tc.task_id WHERE t.project_id = ?", (pid[0],)):
            d = json.loads(data)
            rows.append({"image_id": d.get("image_id"), **answers(json.loads(result)),
                         "seconds": round(float(lead), 1) if lead else None, "answered_at": created,
                         **{k: v for k, v in d.items() if k not in ("image", "image_id")}})
        total[folder] = len(rows)
        if rows and not a.verify:
            pd.DataFrame(rows).to_csv(PKG / folder / name, index=False)
    for folder, n in total.items():
        print(f"  {folder}: {n} answered")
    if not a.verify and any(total.values()):
        print("wrote labelstudio_labels.csv in each task folder that has answers")


if __name__ == "__main__":
    raise SystemExit(main())
