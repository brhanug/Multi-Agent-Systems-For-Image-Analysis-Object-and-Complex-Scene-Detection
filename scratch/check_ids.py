import pandas as pd
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")
df_A = pd.read_csv(BASE / "human_baseline_gold_kit" / "annotator_A_worksheet.csv")
df_ai = pd.read_csv(BASE / "results" / "multi_agent" / "agent_comparison_scores.csv")

def clean_id(name):
    # Standardize format by stripping images/, backslashes, and extensions
    s = str(name).replace("images/", "").replace("\\", "/")
    if s.endswith(".jpg") or s.endswith(".png"):
        s = s.rsplit(".", 1)[0]
    return s.strip()

df_A["clean_id"] = df_A["Image_ID"].apply(clean_id)
df_ai["clean_id"] = df_ai["image_id"].apply(clean_id)

matched = df_A["clean_id"].isin(df_ai["clean_id"])
print("Matched:", matched.sum(), "Unmatched:", (~matched).sum())
if (~matched).sum() > 0:
    print("Unmatched examples:", df_A.loc[~matched, "Image_ID"].head(10).tolist())
    # print some of df_ai clean_ids
    print("Some df_ai examples:", df_ai["clean_id"].head(10).tolist())
