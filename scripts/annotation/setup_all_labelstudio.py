"""Create the Label Studio projects for every open annotation task.

Like setup_a2_labelstudio.py, this runs inside Label Studio's own Django shell, so the
label configs are parsed by its model layer. Existing projects are left untouched, so it
is safe to re-run; pictures come from the CORS image server on port 9000, prepared by
prepare_labelstudio_images.py.

  DJANGO_SETTINGS_MODULE=core.settings.label_studio \
  /data/brhanu/miniconda/bin/python3.13 \
  /data/brhanu/miniconda/lib/python3.13/site-packages/label_studio/manage.py \
  shell -c "exec(open('/data/brhanu/thesis_project/scripts/annotation/setup_all_labelstudio.py').read())"
"""
import json
from pathlib import Path

from projects.models import Project
from tasks.models import Task
from organizations.models import Organization
from users.models import User

BASE = Path("/data/brhanu/thesis_project")
MANIFEST = json.loads((BASE / "self_annotation_package/labelstudio_manifest.json").read_text())
URL = "http://localhost:9000"
CLASSES = ["person", "child", "animal", "horse", "building", "weapon", "vehicle", "tree",
           "text", "clothing", "furniture", "hat"]
FROZEN = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]
SCENES = ["teaching", "family", "playing", "landscape", "drawing"]


def choices(name, values, mode="single", required=True):
    inner = "".join(f'<Choice value="{v}"/>' for v in values)
    return (f'<Choices name="{name}" toName="image" choice="{mode}" '
            f'required="{"true" if required else "false"}">{inner}</Choices>')


def view(header, body, note=None):
    return ("<View>\n"
            '  <Image name="image" value="$image" zoom="true" zoomControl="true"/>\n'
            f'  <Header value="{header}"/>\n'
            + (f'  <Text name="note" value="{note}"/>\n' if note else "")
            + "  " + body + "\n</View>")


PROJECTS = {
    "task_f": ("Task F: What the pseudo-label class ids mean (20 boxes)",
               view("What is inside the red box?",
                    choices("cls", CLASSES + ["something else", "nothing there"])
                    + choices("box", ["good", "loose", "wrong"])
                    + '<TextArea name="comment" toName="image" rows="2" editable="true" required="false"/>',
                    note="One box per picture is outlined. Judge only that box."),
               ["n", "image_id", "class_id", "name_at_training", "name_in_yaml_now", "boxes_in_picture"]),
    "task_g": ("Task G v3: Rare-class mining (306 pictures: vehicle, building, weapon, teaching, family, landscape)",
               view("Which objects appear, and which scene fits best?",
                    choices("classes", FROZEN, mode="multiple", required=False)
                    + choices("scene", SCENES + ["none"], required=False),
                    note="Mark every object you see, and the one scene type that fits best."),
               ["image_id", "source", "key", "sheet"]),
    "task_h": ("Task H: Second scene rating (150 pictures)",
               view("Which scene category fits best?",
                    choices("scene", SCENES + ["none"])
                    + choices("confidence", ["clear", "borderline"], required=False),
                    note="Judge the picture on its own; no earlier label is shown."),
               ["image_id", "sheet"]),
    "task_e": ("Task E: Audit of what the pipeline auto-approves (420 pictures)",
               view("Does this picture show a describable scene?",
                    choices("scene", ["scene", "no_scene"])
                    + choices("confidence", ["clear", "borderline"], required=False),
                    note="The pipeline already approved every picture here without review. A disagreement is a false positive that would have reached the catalogue."),
               ["image_id", "u_triage", "pipeline_says"]),
    "task_a2": ("Task A2 remaining: negative controls still open (465 pictures)",
                view("Does this picture show a describable scene?",
                     choices("scene", ["scene", "no_scene"])
                     + choices("confidence", ["clear", "borderline"], required=False),
                     note="The catalogue default here is 'no scene' and was never checked; earlier passes found 57-67% of those defaults wrong. Judge the picture, not the default."),
                ["image_id", "cvat_id"]),
}

org = Organization.objects.first()
user = User.objects.first()
for key, (title, config, fields) in PROJECTS.items():
    rows = MANIFEST[key]
    existing = Project.objects.filter(title=title).first()
    if existing:
        print(f"exists: '{title}' (id={existing.id}, {Task.objects.filter(project=existing).count()} tasks) -- not modified")
        continue
    p = Project.objects.create(title=title, label_config=config, organization=org, created_by=user)
    for r in rows:
        Task.objects.create(project=p, data={"image": f"{URL}/{r['file']}", **{f: r[f] for f in fields if f in r}})
    print(f"created '{title}' (id={p.id}) with {len(rows)} tasks -> http://localhost:8080/projects/{p.id}/data")
