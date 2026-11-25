"""
Create the Label Studio project for the 750 unreviewed negative controls.

Run through Label Studio's own Django shell so the label config is parsed by its
model layer rather than hand-built:

  DJANGO_SETTINGS_MODULE=core.settings.label_studio \
  /data/brhanu/miniconda/bin/python3.13 \
  /data/brhanu/miniconda/lib/python3.13/site-packages/label_studio/manage.py \
  shell -c "exec(open('/data/brhanu/thesis_project/scripts/annotation/setup_a2_labelstudio.py').read())"
"""
import csv
from pathlib import Path

from projects.models import Project
from tasks.models import Task
from organizations.models import Organization
from users.models import User

BASE = Path("/data/brhanu/thesis_project")
SHEET = BASE / "self_annotation_package/task_a2_remaining_negative_controls/labeling_sheet.csv"
IMG_BASE_URL = "http://localhost:9000"
TITLE = "Self-Annotation A2: Remaining Negative Controls (750)"

CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <Header value="Does this image show a describable scene (person, animal, landscape, drawing/narrative content)?"/>
  <Text name="note" value="The catalogue default for every image here is 'no scene', but that default was never checked by a person and earlier passes found 57-67% of them wrong. Judge the image, not the default."/>
  <Choices name="scene" toName="image" choice="single" required="true">
    <Choice value="scene"/>
    <Choice value="no_scene"/>
  </Choices>
  <Choices name="confidence" toName="image" choice="single">
    <Choice value="clear"/>
    <Choice value="borderline"/>
  </Choices>
</View>"""

org = Organization.objects.first()
user = User.objects.first()

existing = Project.objects.filter(title=TITLE).first()
if existing:
    n = Task.objects.filter(project=existing).count()
    print(f"Project '{TITLE}' already exists (id={existing.id}) with {n} tasks. Not modifying.")
else:
    p = Project.objects.create(title=TITLE, label_config=CONFIG,
                               organization=org, created_by=user)
    rows = list(csv.DictReader(SHEET.open()))
    made = 0
    for r in rows:
        Task.objects.create(project=p, data={
            "image": f"{IMG_BASE_URL}/taska2_{r['image_file']}",
            "cvat_id": int(r["cvat_id"]),
            "ppn_id": r["ppn_id"],
        })
        made += 1
    print(f"Created project '{TITLE}' (id={p.id}) with {made} tasks")
    print(f"Open: http://localhost:8080/projects/{p.id}/data")
