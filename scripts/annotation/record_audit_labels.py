"""
Record contact-sheet labels supplied by the rater in chat.

The rater names the images in one class; every other image on the sheet takes the
other class. Provenance is written per row so these labels are never confused
with the Label Studio UI pass or the 2026 single-rater audit.

  python record_chat_labels.py --batch 1 --no-scene 2,10
  python record_chat_labels.py --batch 2 --scene 1,4,9 --borderline 7
"""
import argparse, csv, json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")
PKG = BASE / "self_annotation_package/task_e_auto_approved_audit"
LABELS = PKG / "chat_labels.csv"
FIELDS = ["cvat_id", "ppn_id", "image_file", "label", "borderline",
          "batch", "recorded_utc", "method"]


def parse(s):
    return {int(x) for x in str(s).replace(" ", "").split(",") if x} if s else set()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--batch", type=int, required=True)
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--scene", help="numbers that DO show a scene; the rest are no_scene")
    g.add_argument("--no-scene", help="numbers that do NOT show a scene; the rest are scene")
    a.add_argument("--borderline", default="")
    args = a.parse_args()

    man = json.loads((PKG / "contact_sheets" / f"batch_{args.batch:03d}.json").read_text())
    nums = {r["n"] for r in man}
    named = parse(args.scene or args.no_scene)
    bad = named - nums
    if bad:
        raise SystemExit(f"numbers not on sheet {args.batch}: {sorted(bad)} (sheet has 1..{max(nums)})")
    bl = parse(args.borderline)

    if args.scene is not None:
        scene = named
    else:
        scene = nums - named

    existing = []
    seen = set()
    if LABELS.exists():
        existing = list(csv.DictReader(LABELS.open()))
        seen = {int(r["cvat_id"]) for r in existing}

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    added, dup = [], 0
    for r in man:
        if r["cvat_id"] in seen:
            dup += 1
            continue
        added.append({"cvat_id": r["cvat_id"], "ppn_id": r["ppn_id"],
                      "image_file": r["image_file"],
                      "label": "scene" if r["n"] in scene else "no_scene",
                      "borderline": int(r["n"] in bl),
                      "batch": args.batch, "recorded_utc": now,
                      "method": "rater_via_chat_auto_approved_audit"})

    with LABELS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(existing + added)

    ns = sum(1 for r in added if r["label"] == "no_scene")
    print(f"batch {args.batch}: recorded {len(added)} ({len(added)-ns} scene, {ns} no_scene)"
          + (f", skipped {dup} already present" if dup else ""))
    allrows = existing + added
    ns = sum(1 for r in allrows if r["label"] == "no_scene")
    tot = len(allrows)
    print(f"cumulative: {tot} audited | false positives (auto-approved but NO scene) "
          f"{ns} = {100*ns/tot:.1f}%  [certified risk target 5%]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
