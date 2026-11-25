import os
import pandas as pd
from PIL import Image

image_path = "results/figures/thesis_agent_diversity_heatmap.png"
if os.path.exists(image_path):
    img = Image.open(image_path)
    print("Heatmap image size:", img.size)
else:
    print("Heatmap image does not exist!")

csv_path = "results/multi_agent/agent_diversity_matrix.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    print("CSV contents:")
    print(df)
else:
    print("CSV does not exist!")
