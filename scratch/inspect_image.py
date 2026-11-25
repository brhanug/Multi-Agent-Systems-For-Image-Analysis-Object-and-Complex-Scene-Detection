from PIL import Image
import numpy as np

img = Image.open("results/figures/thesis_agent_diversity_heatmap.png")
print("Image dimensions:", img.size)
print("Image mode:", img.mode)

# Let us check if there is any text or other elements we can inspect.
# We can also check if the image has a different md5sum than before we ran the regeneration.
# Wait, let us check if the script runs on the correct agent_diversity_matrix.csv.
import pandas as pd
df = pd.read_csv("results/multi_agent/agent_diversity_matrix.csv", index_col=0)
print("\nCSV shape:", df.shape)
print("CSV columns:", list(df.columns))
print("CSV index:", list(df.index))
