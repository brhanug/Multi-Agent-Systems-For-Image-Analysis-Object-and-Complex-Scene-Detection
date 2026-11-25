import os
import pandas as pd
from PIL import Image

image_path = "results/figures/thesis_agent_diversity_heatmap.png"
if os.path.exists(image_path):
    img = Image.open(image_path)
    print("results/figures/thesis_agent_diversity_heatmap.png size:", img.size)
else:
    print("results/figures/thesis_agent_diversity_heatmap.png does not exist")

latex_image_path = "latex_assets/thesis_agent_diversity_heatmap.png"
if os.path.exists(latex_image_path):
    img2 = Image.open(latex_image_path)
    print("latex_assets/thesis_agent_diversity_heatmap.png size:", img2.size)
else:
    print("latex_assets/thesis_agent_diversity_heatmap.png does not exist")

# Let's check the CSV used.
csv_path = "results/multi_agent/agent_diversity_matrix.csv"
df = pd.read_csv(csv_path, index_col=0)
print("CSV columns count:", len(df.columns), list(df.columns))

# Let's count the number of colored blocks in the heatmap.
# Matplotlib saves images with ticks. We can regenerate a 4x4 and a 6x6 and see if the output is different.
