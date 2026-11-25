import os
import glob
from PIL import Image

heatmaps = glob.glob("latex_assets/*heatmap*.png")
for h in heatmaps:
    img = Image.open(h)
    print(f"File: {h}, size: {img.size}")
