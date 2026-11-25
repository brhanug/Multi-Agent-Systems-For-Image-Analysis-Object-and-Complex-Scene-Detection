import pandas as pd
from pathlib import Path

csv_path = Path("/data/brhanu/thesis_project/results/multi_agent/agent_diversity_matrix.csv")
df = pd.read_csv(csv_path, index_col=0)
print("Columns read by script:", list(df.columns))
print("Index read by script:", list(df.index))
print("Matrix values:\n", df)
