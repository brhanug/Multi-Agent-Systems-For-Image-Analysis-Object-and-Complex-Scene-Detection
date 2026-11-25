import json
from pathlib import Path

base = Path("/data/brhanu/thesis_project")
fusion_path = base / "results/multi_agent/upgraded_agent0_fusion.json"

with open(fusion_path) as f:
    data = json.load(f)

print(f"Total records in upgraded_agent0_fusion.json: {len(data)}")

# Let's count contradictions
contra_records = [r for r in data if r.get("critic_audit", {}).get("contradictions_found") is True]
print(f"Total records with contradictions: {len(contra_records)}")

# Let's inspect some of these
for r in contra_records[:10]:
    print(f"Image ID: {r['image_id']} | Image Name: {r['image_name']}")
    print(f"  YOLO Detections: {[o['name'] for o in r['synthesized_metadata'].get('objects', [])]}")
    print(f"  OCR: {r['synthesized_metadata'].get('ocr_transcription')}")
    print(f"  Critic Justification: {r['critic_audit'].get('justification')}")
    print("-" * 50)
