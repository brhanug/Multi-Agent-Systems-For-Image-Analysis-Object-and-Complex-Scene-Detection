#!/usr/bin/env python3
import os
import yaml

# Clean list of physically detectable objects from the top 50 unconstrained entities
OPEN_VOCAB_CLASSES = [
    "boy", "hat", "dog", "tree", "children", "girl", "kitchen", "child", 
    "window", "sack", "field", "head", "cat", "table", "rooster", "crown", 
    "woods", "book", "boys", "gun", "room", "sheep", "face", "crest", 
    "garden", "animals", "chicken", "apples", "chair", "bed", "bird", 
    "grass", "water", "horse", "floor", "men", "pond"
]

# De-duplicate just in case, though they are unique
OPEN_VOCAB_CLASSES = list(dict.fromkeys(OPEN_VOCAB_CLASSES))

YAML_PATH = "/data/brhanu/thesis_project/data/yolo11_open_vocab.yaml"

def generate_yaml():
    data = {
        'path': '/data/brhanu/thesis_project/data/yolo11_open_vocab',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/val',
        'names': {i: cls_name for i, cls_name in enumerate(OPEN_VOCAB_CLASSES)}
    }
    
    os.makedirs(os.path.dirname(YAML_PATH), exist_ok=True)
    with open(YAML_PATH, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
    print(f"✅ Generated Open-Vocabulary YOLO YAML at: {YAML_PATH}")
    print(f"🧠 Total Classes: {len(OPEN_VOCAB_CLASSES)}")
    for i, name in enumerate(OPEN_VOCAB_CLASSES):
        print(f"  {i}: {name}")

if __name__ == "__main__":
    generate_yaml()
