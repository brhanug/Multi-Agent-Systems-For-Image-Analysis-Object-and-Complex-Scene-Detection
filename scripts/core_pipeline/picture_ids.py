"""One picture identity for every join: book and page, never the page alone.

The expert annotation file (human_spatial_audit/user_annotations_800.csv) keeps the page in its
`image_id` column ("00000083_1") and the book only in `raw_name` ("images/PPN183128149X_00000083_1.jpg").
Page numbers repeat across books, so a join on the page alone matched 548 of the 801 expert pictures
to another book's scores (found 3 November 2026). `full_id` keeps the book; `read_expert_annotations`
puts the full name into `image_id`, so existing merges on that column join the right picture.
"""
import re

import pandas as pd


def full_id(name):
    """'images/PPN1_00000083_1.jpg', 'PPN1/00000083_1' or 'PPN1_00000083_1' -> 'PPN1_00000083_1'."""
    p = str(name).replace("images/", "").replace("\\", "/")
    if re.search(r"\.(jpe?g|png)$", p, re.I):
        p = p.rsplit(".", 1)[0]
    parts = p.split("/")
    if parts[-1].startswith("PPN") or len(parts) == 1:
        return parts[-1]
    return parts[-2] + "_" + parts[-1]                   # "PPN1/00000083_1"


def read_expert_annotations(path, **kwargs):
    df = pd.read_csv(path, **kwargs)
    if "raw_name" in df.columns:
        df["image_id"] = df["raw_name"].map(full_id)
    return df
