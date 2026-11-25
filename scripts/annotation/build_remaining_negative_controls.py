"""
Build the annotation package for the negative-control images still unreviewed.

The 1,199-row default-labelled block (rows 801..1999 of the annotation table)
carried catalogue-default "no scene" labels that were never checked by a person.
449 have since been reviewed in two passes; the rest have not, and the thesis's
blind-spot finding currently rests on 5 errors among 38 confirmed negatives.

This selects exactly the unreviewed remainder, excluding every row already
covered by either pass, and writes a manifest plus copies of the images.
"""
import csv, json, shutil
from pathlib import Path
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
C1_CSV = BASE / "human_spatial_audit/user_annotations_800.csv"
AUDIT_JSON = BASE / "results/multi_agent/negative_control_reverification_audit.json"
LS_EXPORT = BASE / "self_annotation_package/task_a_negative_controls/labeling_sheet_LS_export.csv"
IMG_ROOT = BASE / "final_dataset/images/restored"
OUT = BASE / "self_annotation_package/task_a2_remaining_negative_controls"
N_EXPERT_ROWS = 801


def main():
    raw = pd.read_csv(C1_CSV)
    block = raw.iloc[N_EXPERT_ROWS:].copy()          # the 1,199 default-labelled rows
    block["cvat_id"] = block.index

    def stem_of(raw_name):
        return Path(str(raw_name)).stem              # PPN1816495190_00000281_0

    # --- everything already reviewed, by stem ---
    done = set()
    audit = json.load(open(AUDIT_JSON))
    for rec in audit["records"]:
        done.add(rec["full_id"])
    ls = pd.read_csv(LS_EXPORT)
    for cid in ls["cvat_id"].astype(int):
        done.add(stem_of(raw.iloc[cid]["raw_name"]))

    block["stem"] = block["raw_name"].map(stem_of)
    remaining = block[~block["stem"].isin(done)].copy()
    print(f"default-labelled block : {len(block)}")
    print(f"already reviewed       : {len(done)}")
    print(f"remaining to annotate  : {len(remaining)}")

    # --- resolve each to a real image on disk ---
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    rows, missing = [], 0
    for _, r in remaining.iterrows():
        ppn, rest = r["stem"].split("_", 1)
        src = IMG_ROOT / ppn / f"{rest}.jpg"
        if not src.exists():
            missing += 1
            continue
        fname = f"{int(r['cvat_id']):04d}_{r['stem']}.jpg"
        dst = OUT / "images" / fname
        if not dst.exists():
            shutil.copy(src, dst)
        rows.append({"cvat_id": int(r["cvat_id"]), "image_file": fname,
                     "ppn_id": f"{ppn}/{rest}", "ppn": ppn,
                     "current_label": "no_scene", "your_label": "", "notes": ""})
    print(f"images resolved        : {len(rows)}   (missing on disk: {missing})")

    with (OUT / "labeling_sheet.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cvat_id","image_file","ppn_id","ppn",
                                          "current_label","your_label","notes"])
        w.writeheader(); w.writerows(rows)
    (OUT / "README.md").write_text(
        "# Task A2 - remaining negative controls\n\n"
        f"{len(rows)} images from the 1,199-row default-labelled block that no one has\n"
        "reviewed. Every row already covered by the 2026 single-rater audit or the\n"
        "Label Studio second-rater pass is excluded, so this is disjoint from both.\n\n"
        "Question: does the image show a describable scene (person, animal, landscape,\n"
        "drawing or narrative content)?\n\n"
        "`current_label` is the catalogue default, not a human judgement. It is shown\n"
        "for reference only and is wrong often enough that it should not anchor you:\n"
        "the two earlier passes found 66.7% and 57.5% of these defaults incorrect.\n\n"
        "`cvat_id` indexes the original 2,000-row annotation table, which is how the\n"
        "labels merge back into the corrected pool.\n")
    print(f"wrote {OUT}/labeling_sheet.csv and README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
