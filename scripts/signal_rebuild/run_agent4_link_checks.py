#!/usr/bin/env python3
"""Two mechanical checks of Agent 4's cross-volume links (SigLIP image embeddings, nearest neighbour in a
different volume): how many links at similarity >= 0.92 join the same page number in both volumes (likely
the same book digitised twice), over the whole archive and among the 79 labelled pairs by judgement.

    python run_agent4_link_checks.py [--verify]
"""
import argparse, json, re
from pathlib import Path

import numpy as np
import pandas as pd

import run_best_config_repeated as rb

OUT = rb.SR / "agent4_link_checks.json"
TC = Path("/data/brhanu/thesis_project/self_annotation_package/task_c_cross_archive_duplicates")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    z = np.load(rb.SR / "siglip_embeddings.npz", allow_pickle=True); ids = [str(x) for x in z["ids"]]
    V = z["vecs"].astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True)
    vol = np.array([i.split("/")[0] for i in ids]); page = np.array([i.split("/")[1] for i in ids])
    best = np.full(len(ids), -1.0); arg = np.zeros(len(ids), int)
    for s in range(0, len(ids), 1024):
        S = V[s:s + 1024] @ V.T; S[vol[s:s + 1024, None] == vol[None, :]] = -1
        k = S.argmax(1); best[s:s + 1024] = S[np.arange(len(k)), k]; arg[s:s + 1024] = k
    m = best >= 0.92; same = page[m] == page[arg[m]]
    sh = pd.read_csv(TC / "labeling_sheet.csv").drop(columns="your_judgment")
    lab = sh.merge(pd.read_csv(TC / "labeling_sheet_LS_export.csv")[["pair_id", "your_judgment"]], on="pair_id")
    pg = lambda f: re.search(r"_(\d{8}_\d+)\.jpg", f).group(1)
    lab["same_page"] = [pg(x) == pg(y) for x, y in zip(lab.image_a, lab.image_b)]
    out = {"archive": {"pictures_with_link_at_0.92": int(m.sum()), "same_page_number": int(same.sum()),
                       "same_page_share": round(float(same.mean()), 4)},
           "labelled_pairs": {j: {"pairs": int((lab.your_judgment == j).sum()), "same_page_number": int(lab[lab.your_judgment == j].same_page.sum())}
                              for j in ("same_composition", "same_style_only", "unrelated")}}
    if a.verify:
        same_ = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same_ else "MISMATCH"); return 0 if same_ else 1
    OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
