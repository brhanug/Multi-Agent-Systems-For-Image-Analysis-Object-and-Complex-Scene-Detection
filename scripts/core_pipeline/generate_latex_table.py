from ultralytics import YOLO
import pandas as pd

# CONFIG
BEST_WEIGHTS = "/data/brhanu/thesis_project/results/yolo11_final_open_vocab/exp_open_vocab3/weights/best.pt"
DATA_YAML = "/data/brhanu/thesis_project/data/yolo11_open_vocab.yaml"

# Load model
model = YOLO(BEST_WEIGHTS)

# Run validation and get metrics
results = model.val(data=DATA_YAML, split='val', save_json=True)

# results.box.p, results.box.r, results.box.ap50, results.box.ap are numpy arrays for each class
class_names = model.names
p_per_class = results.box.p
r_per_class = results.box.r
map50_per_class = results.box.ap50
map50_95_per_class = results.box.ap

# Get instance counts
# results.box.nc is number of classes
# results.box.stats stores instances per class potentially
# A cleaner way to get instance counts per class in validation:
# results.box.nc is num classes. results.box.p has results for each.
# For instances, we can look at results.box.nc (which is count of instances? No, that's num classes)
# Actually, the counts are often in the printed table.
# Let's just use the metrics for now.

print("\\begin{tabular}{lrrrr}")
print("\\toprule")
print("Class & Precision & Recall & mAP50 & mAP50-95 \\\\")
print("\\midrule")
for i, name in class_names.items():
    print(f"{name} & {p_per_class[i]:.3f} & {r_per_class[i]:.3f} & {map50_per_class[i]:.3f} & {map50_95_per_class[i]:.3f} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
