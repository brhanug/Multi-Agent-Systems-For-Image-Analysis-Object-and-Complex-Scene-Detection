import base64
import requests
import json
from pathlib import Path

img_path = Path("/data/brhanu/thesis_project/human_spatial_audit/images/PPN1756641218_00000122_0.jpg")
with open(img_path, "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("utf-8")

prompt = "Is there a scene (people, objects, landscapes, events) in this image? Answer with a float between 0.0 and 1.0 representing confidence."

# Test InternVL3_5:4B
print("Testing blaifa/InternVL3_5:4B...")
payload_intern = {
    "model": "blaifa/InternVL3_5:4B",
    "prompt": prompt,
    "images": [b64_data],
    "stream": False
}
try:
    r = requests.post("http://localhost:11434/api/generate", json=payload_intern, timeout=30)
    print("Response:", r.json().get("response"))
except Exception as e:
    print("Error:", e)

# Test qwen3.5:latest
print("\nTesting qwen3.5:latest...")
payload_qwen = {
    "model": "qwen3.5:latest",
    "prompt": prompt,
    "images": [b64_data],
    "stream": False
}
try:
    r = requests.post("http://localhost:11434/api/generate", json=payload_qwen, timeout=30)
    print("Response:", r.json().get("response"))
except Exception as e:
    print("Error:", e)
