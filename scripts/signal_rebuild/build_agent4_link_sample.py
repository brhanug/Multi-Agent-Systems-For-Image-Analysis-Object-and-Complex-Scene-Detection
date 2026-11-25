#!/usr/bin/env python3
"""A random sample of Agent 4's archive-wide links for judging
(self_annotation_package/more_experiments/PLAN_agent4_archive_links.md).

Recomputes each picture's nearest neighbour in another book over the stored SigLIP embeddings
(the same rule as run_agent4_embeddings_full_corpus.py), stops unless 2,434 pictures have one at
similarity >= 0.92, and draws 50 distinct pairs at random (seed 20261121). Writes
  self_annotation_package/agent4_links/items.csv   the pairs the page shows (no similarity)
  self_annotation_package/agent4_links/key.csv     similarity and same-page flag (never served)
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
EMB = BASE / "results/signal_rebuild/siglip_embeddings.npz"
OUT = BASE / "self_annotation_package/agent4_links"
SEED, N, T = 20261121, 50, 0.92


def main():
    d = np.load(EMB, allow_pickle=True)
    ids, V = list(d["ids"]), d["vecs"]
    vol = np.array([i.split("/")[0] for i in ids]); n = len(ids)
    best_s = np.full(n, -1.0, dtype=np.float32); best_j = np.full(n, -1, dtype=np.int64)
    for a in range(0, n, 512):
        S = V[a:a + 512] @ V.T
        for r in range(S.shape[0]):
            g = a + r; S[r, g] = -1.0; S[r, vol == vol[g]] = -1.0
            j = int(np.argmax(S[r])); best_s[g], best_j[g] = float(S[r, j]), j
    links = np.where(best_s >= T)[0]
    assert len(links) == 2434, f"expected 2,434 links, got {len(links)}"
    order = np.random.default_rng(SEED).permutation(links)
    seen, rows = set(), []
    for g in order:
        a, b = ids[g], ids[int(best_j[g])]
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"item_id": f"{a}|{b}", "a": a, "b": b, "similarity": round(float(best_s[g]), 4),
                     "same_page_number": a.split("/")[1] == b.split("/")[1]})
        if len(rows) == N:
            break
    k = pd.DataFrame(rows)
    missing = [x for x in list(k.a) + list(k.b) if not (BASE / f"final_dataset/images/original/{x}.jpg").exists()]
    assert not missing, f"{len(missing)} pictures have no image file, e.g. {missing[:3]}"
    OUT.mkdir(parents=True, exist_ok=True)
    k[["item_id", "a", "b"]].to_csv(OUT / "items.csv", index=False)
    k.to_csv(OUT / "key.csv", index=False)
    print(f"2,434 links reproduce; {len(k)} distinct pairs drawn, {int(k.same_page_number.sum())} with the same page number, "
          f"similarity {k.similarity.min():.3f}-{k.similarity.max():.3f}")


if __name__ == "__main__":
    main()
