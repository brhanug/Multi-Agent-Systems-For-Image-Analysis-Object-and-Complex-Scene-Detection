import os
from ultralytics import YOLO
import pandas as pd

# CONFIG
BEST_WEIGHTS = "/data/brhanu/thesis_project/results/yolo11_final_open_vocab/exp_open_vocab3/weights/best.pt"
DATA_YAML = "/data/brhanu/thesis_project/data/yolo11_open_vocab/data.yaml"

if not os.path.exists(BEST_WEIGHTS):
    print(f"❌ Error: Weights not found at {BEST_WEIGHTS}")
    exit(1)

# Initialize model
model = YOLO(BEST_WEIGHTS)

# Run validation
print(f"🚀 Running validation on {BEST_WEIGHTS}...")
results = model.val(data=DATA_YAML, split='val', save_json=True)

# Extract per-class metrics
# results.results_dict contains top-level [metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)]
# results.mean_results() gives the overall
# results.class_result(i) gives metrics for class idx i

class_names = model.names
per_class_data = []

for i, name in class_names.items():
    # results.box.p[i] precision, r[i] recall, ap50[i] ap50, ap[i] ap50-95
    # Since different versions might have different attr names, let's use the standard ones
    p = results.box.p[i]
    r = results.box.r[i]
    map50 = results.box.ap50[i]
    map50_95 = results.box.ap[i]
    
    # We also need instance counts. These are in results.box.nc
    # Actually instances are in results.box.stats[idx][...]
    # Better yet, results output has maps per class.
    
    per_class_data.append({
        "Class": name,
        "Precision": f"{p:.4f}",
        "Recall": f"{r:.4f}",
        "mAP50": f"{map50:.4f}",
        "mAP50-95": f"{map50_95:.4f}"
    })

df = pd.DataFrame(per_class_data)
output_path = "/data/brhanu/thesis_project/results/per_class_37_final.csv"
df.to_csv(output_path, index=False)
print(f"✅ Per-class metrics saved to {output_path}")
print(df.to_string())
