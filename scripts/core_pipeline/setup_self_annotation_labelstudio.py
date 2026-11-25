#!/usr/bin/env python3
"""
setup_self_annotation_labelstudio.py
---------------------------------------
Creates four Label Studio projects for the self-annotation package
(self_annotation_package/task_{a,b,c,d}_*), using the same real images and
CSVs already built there, instead of spreadsheet editing.

Run via Django shell so Label Studio's own model layer computes
parsed_label_config/control_weights correctly (not hand-crafted JSON):

  DJANGO_SETTINGS_MODULE=core.settings.label_studio \
  /data/brhanu/miniconda/bin/python3.13 \
  /data/brhanu/miniconda/lib/python3.13/site-packages/label_studio/manage.py \
  shell -c "exec(open('/data/brhanu/thesis_project/scripts/core_pipeline/setup_self_annotation_labelstudio.py').read())"

Images are copied into cvat_upload_temp/obj_train_data/ (the directory
serve_images_cors.py serves on port 9000, CORS-enabled) with a per-task
prefix to avoid collisions, then referenced by that URL in each task's data.
"""
import csv
import shutil
from pathlib import Path

from projects.models import Project
from tasks.models import Task
from organizations.models import Organization
from users.models import User

BASE = Path("/data/brhanu/thesis_project")
PKG = BASE / "self_annotation_package"
CORS_DIR = BASE / "cvat_upload_temp" / "obj_train_data"
CORS_DIR.mkdir(parents=True, exist_ok=True)
IMG_BASE_URL = "http://localhost:9000"

org = Organization.objects.first()
user = User.objects.first()


def get_or_create_project(title, label_config):
    p = Project.objects.filter(title=title).first()
    if p:
        print(f"  Project '{title}' already exists (id={p.id}), skipping creation.")
        return p, False
    p = Project.objects.create(
        title=title,
        label_config=label_config,
        organization=org,
        created_by=user,
    )
    print(f"  Created project '{title}' (id={p.id})")
    return p, True


def copy_image(src: Path, prefix: str) -> str:
    dst_name = f"{prefix}_{src.name}"
    dst = CORS_DIR / dst_name
    if not dst.exists():
        shutil.copy(src, dst)
    return f"{IMG_BASE_URL}/{dst_name}"


# ── Task A: negative-control scene/no_scene ─────────────────────────────────
def setup_task_a():
    title = "Self-Annotation A: Negative Control Blind Spot"
    config = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <Header value="Does this image show a describable scene (person, animal, landscape, drawing/narrative content)?"/>
  <Choices name="scene" toName="image" choice="single" required="true">
    <Choice value="scene"/>
    <Choice value="no_scene"/>
  </Choices>
</View>"""
    project, created = get_or_create_project(title, config)
    if not created and Task.objects.filter(project=project).exists():
        return
    task_dir = PKG / "task_a_negative_controls"
    rows = list(csv.DictReader((task_dir / "labeling_sheet.csv").open()))
    n = 0
    for r in rows:
        src = task_dir / "images" / r["image_file"]
        if not src.exists():
            continue
        url = copy_image(src, "taska")
        Task.objects.create(project=project, data={"image": url, "cvat_id": r["cvat_id"]})
        n += 1
    print(f"  Task A: inserted {n} tasks")


# ── Task B: contradiction ground truth ──────────────────────────────────────
def setup_task_b():
    title = "Self-Annotation B: Agent 2 Contradiction Ground Truth"
    config = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <Header value="VLM claimed this, YOLO did NOT detect it:"/>
  <Text name="claimed" value="$claimed_noun"/>
  <Header value="YOLO detected instead:"/>
  <Text name="detected" value="$yolo_detected_instead"/>
  <Choices name="judgment" toName="image" choice="single" required="true">
    <Choice value="present"/>
    <Choice value="absent"/>
    <Choice value="ambiguous"/>
  </Choices>
</View>"""
    project, created = get_or_create_project(title, config)
    if not created and Task.objects.filter(project=project).exists():
        return
    task_dir = PKG / "task_b_contradiction_ground_truth"
    rows = list(csv.DictReader((task_dir / "labeling_sheet.csv").open()))
    n = 0
    for r in rows:
        src = task_dir / "images" / r["image_file"]
        if not src.exists():
            continue
        url = copy_image(src, "taskb")
        Task.objects.create(project=project, data={
            "image": url,
            "claimed_noun": r["claimed_noun"],
            "yolo_detected_instead": r["yolo_detected_instead"],
            "row_id": r["id"],
        })
        n += 1
    print(f"  Task B: inserted {n} tasks")


# ── Task C: cross-archive duplicate pairs ───────────────────────────────────
def setup_task_c():
    title = "Self-Annotation C: Agent 4 Cross-Archive Duplicate Pairs"
    config = """<View>
  <View style="display: flex;">
    <View style="width: 50%; padding: 5px;">
      <Header value="Image A"/>
      <Image name="image_a" value="$image_a" zoom="true" zoomControl="true"/>
    </View>
    <View style="width: 50%; padding: 5px;">
      <Header value="Image B"/>
      <Image name="image_b" value="$image_b" zoom="true" zoomControl="true"/>
    </View>
  </View>
  <Text name="sim" value="Embedding similarity score: $similarity"/>
  <Choices name="judgment" toName="image_a" choice="single" required="true">
    <Choice value="same_composition"/>
    <Choice value="same_style_only"/>
    <Choice value="unrelated"/>
  </Choices>
</View>"""
    project, created = get_or_create_project(title, config)
    if not created and Task.objects.filter(project=project).exists():
        return
    task_dir = PKG / "task_c_cross_archive_duplicates"
    rows = list(csv.DictReader((task_dir / "labeling_sheet.csv").open()))
    n = 0
    for r in rows:
        src_a = task_dir / "images" / r["image_a"]
        src_b = task_dir / "images" / r["image_b"]
        if not (src_a.exists() and src_b.exists()):
            continue
        url_a = copy_image(src_a, "taskc")
        url_b = copy_image(src_b, "taskc")
        Task.objects.create(project=project, data={
            "image_a": url_a, "image_b": url_b,
            "similarity": r["similarity"], "pair_id": r["pair_id"],
        })
        n += 1
    print(f"  Task C: inserted {n} tasks")


# ── Task D: OCR ground-truth transcription ──────────────────────────────────
def setup_task_d():
    title = "Self-Annotation D: Agent 5 OCR Ground Truth"
    config = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <Header value="Kosmos-2.5 said (reference only -- read the image yourself first):"/>
  <Text name="kosmos" value="$kosmos_output"/>
  <TextArea name="transcription" toName="image" placeholder="Type what the image actually says, or NO_TEXT" rows="4" required="true" maxSubmissions="1"/>
</View>"""
    project, created = get_or_create_project(title, config)
    if not created and Task.objects.filter(project=project).exists():
        return
    task_dir = PKG / "task_d_ocr_ground_truth"
    rows = list(csv.DictReader((task_dir / "labeling_sheet.csv").open()))
    n = 0
    for r in rows:
        src = task_dir / "images" / r["image_file"]
        if not src.exists():
            continue
        url = copy_image(src, "taskd")
        Task.objects.create(project=project, data={
            "image": url,
            "kosmos_output": r["kosmos_output"],
            "kosmos_bucket": r["kosmos_bucket"],
            "row_id": r["id"],
        })
        n += 1
    print(f"  Task D: inserted {n} tasks")


print("Setting up self-annotation Label Studio projects...")
setup_task_a()
setup_task_b()
setup_task_c()
setup_task_d()
print("Done. Open Label Studio and look for the 4 new 'Self-Annotation' projects.")
