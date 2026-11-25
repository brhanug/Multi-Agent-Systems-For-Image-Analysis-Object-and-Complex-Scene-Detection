#!/usr/bin/env python3
"""Put every open task's pictures where the CORS image server can serve them.

The server (scripts/core_pipeline/serve_images_cors.py, port 9000) serves one directory,
and Label Studio tasks point at http://localhost:9000/<prefixed name>. Pictures are
symlinked under a per-task prefix, except Task F, whose pictures are drawn with the red
box the rater has to judge. Each task also gets a manifest the project setup reads.

Already-labelled pictures are left out (Task E), so a project only holds open work.

    python prepare_labelstudio_images.py
"""
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

BASE = Path("/data/brhanu/thesis_project")
SERVE = BASE / "cvat_upload_temp/obj_train_data"
PKG = BASE / "self_annotation_package"
RESTORED = BASE / "final_dataset/images/restored"
MAXSIDE = 1600


def link(src: Path, name: str):
    dst = SERVE / name
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())
    return name


def picture(image_id: str) -> Path:
    p = RESTORED / f"{image_id}.jpg"
    assert p.exists(), f"no picture for {image_id}"
    return p


def task_f():
    rows = json.loads((PKG / "task_f_pseudo_label_classes/contact_sheets/sheet.json").read_text())
    out = []
    for r in rows:
        im = Image.open(picture(r["image_id"])).convert("RGB")
        w0, h0 = r["picture_px"]
        sx, sy = im.width / w0, im.height / h0
        x1, y1, x2, y2 = r["box_px"]
        d = ImageDraw.Draw(im)
        d.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline="#ff2d2d", width=max(4, im.width // 250))
        if max(im.size) > MAXSIDE:
            f = MAXSIDE / max(im.size); im = im.resize((int(im.width * f), int(im.height * f)))
        name = f"taskf_{r['n']:02d}.jpg"
        im.save(SERVE / name, quality=88)
        out.append({"file": name, "n": r["n"], "image_id": r["image_id"], "class_id": r["class_id"],
                    "name_at_training": r["name_at_training_2026_01_28"],
                    "name_in_yaml_now": r["name_in_yaml_since_2026_02_19"],
                    "boxes_in_picture": r["boxes_in_picture"]})
    return out


def task_g():
    s = pd.read_csv(PKG / "task_g_rare_class_mining/labeling_sheet.csv")
    out = []
    for _, r in s.iterrows():
        iid = r["image_id"]
        name = link(picture(iid), "taskg_" + iid.replace("/", "_") + ".jpg")
        out.append({"file": name, "image_id": iid, "source": r["source"], "key": str(r["key"]), "sheet": r["sheet"]})
    return out


def task_h():
    s = pd.read_csv(PKG / "task_h_second_rater_scenes/labeling_sheet.csv")
    out = []
    for _, r in s.iterrows():
        iid = r["image_id"]
        name = link(BASE / f"human_spatial_audit/images/{iid}.jpg", f"taskh_{iid}.jpg")
        out.append({"file": name, "image_id": iid, "sheet": r["sheet"]})
    return out


def task_e():
    s = pd.read_csv(PKG / "task_e_auto_approved_audit/labeling_sheet.csv")
    done = set(pd.read_csv(PKG / "task_e_auto_approved_audit/chat_labels.csv").ppn_id)
    imgs = {p.name.split("_", 1)[1].rsplit(".", 1)[0].replace("_", "/", 1): p
            for p in (PKG / "task_e_auto_approved_audit/images").glob("*.jpg")}
    out = []
    for _, r in s.iterrows():
        iid = r["image_id"]
        if iid in done or iid not in imgs: continue
        name = link(imgs[iid], "taske_" + iid.replace("/", "_") + ".jpg")
        out.append({"file": name, "image_id": iid, "u_triage": float(r["U_triage"]), "pipeline_says": r["pipeline_says"]})
    return out


def task_a2():
    """Project 8 holds all 750; these are the 465 still open, whose pictures are already served."""
    s = pd.read_csv(PKG / "task_a2_remaining_negative_controls/labeling_sheet.csv")
    done = set(pd.read_csv(PKG / "task_a2_remaining_negative_controls/chat_labels.csv").ppn_id)
    out = []
    for _, r in s.iterrows():
        if r["ppn_id"] in done: continue
        name = f"taska2_{r['image_file']}"
        assert (SERVE / name).exists(), f"missing served picture {name}"
        out.append({"file": name, "image_id": r["ppn_id"], "cvat_id": int(r["cvat_id"])})
    return out


def main():
    SERVE.mkdir(parents=True, exist_ok=True)
    man = {"task_f": task_f(), "task_g": task_g(), "task_h": task_h(), "task_e": task_e(), "task_a2": task_a2()}
    (PKG / "labelstudio_manifest.json").write_text(json.dumps(man, indent=1))
    for k, v in man.items():
        print(f"{k}: {len(v)} pictures ready")
    print("wrote", PKG / "labelstudio_manifest.json")


if __name__ == "__main__":
    main()
