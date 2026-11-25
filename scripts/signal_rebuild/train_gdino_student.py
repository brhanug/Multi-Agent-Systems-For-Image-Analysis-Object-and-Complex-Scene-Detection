#!/usr/bin/env python3
"""Student rebuild, step 4: train a new YOLO11m student on the consensus pseudo-labels.

Same architecture and starting point as the deployed student (YOLO11m pretrained on COCO), so the
comparison isolates the teacher. Validation during training uses the held-out books of the pseudo-label
set; the real test is afterwards, on the human boxes (evaluate with run_teacher_candidates_on_300.py).

    python train_gdino_student.py --device 1
"""
import argparse
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="1"); ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    from ultralytics import YOLO
    YOLO(str(BASE / "yolo11m.pt")).train(
        data=str(BASE / "data/yolo11_gdino_consensus/data.yaml"), epochs=a.epochs, imgsz=640, batch=32,
        device=a.device, project=str(BASE / "runs/detect"), name="yolo11_gdino_consensus", exist_ok=True,
        patience=15, seed=20260919, deterministic=True, plots=False, workers=8)


if __name__ == "__main__":
    main()
