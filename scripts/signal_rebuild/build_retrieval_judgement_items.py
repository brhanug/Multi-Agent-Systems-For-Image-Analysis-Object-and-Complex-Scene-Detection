#!/usr/bin/env python3
"""Items for the human relevance judgements of the retrieval pilot
(self_annotation_package/more_experiments/PLAN_retrieval_human.md).

Recomputes, with the pilot's own functions and input files, the top 10 of the semantic index and of
the caption-keyword baseline for each of the 10 queries; stops unless the stored proxy figures
(mean P@10 1.000 and 0.180) reproduce. Writes
  self_annotation_package/retrieval_judgements/items.csv     what the page shows (query, picture)
  self_annotation_package/retrieval_judgements/key.csv       which method found it, proxy label (never served)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
sys.path.insert(0, str(BASE / "scripts/core_pipeline"))
import run_semantic_retrieval_eval as rr  # noqa: E402

OUT = BASE / "self_annotation_package/retrieval_judgements"
SEED = 20261121


def main():
    scores = pd.read_csv(BASE / "results/multi_agent/agent_comparison_scores.csv")
    manifest = pd.read_csv(BASE / "final_dataset/metadata/manifest.csv")
    manifest["img_key"] = manifest["image_id"].apply(rr.normalize_id)
    scores["img_key"] = scores["image_id"].apply(rr.normalize_id)
    df = scores.merge(manifest[["img_key", "vqa_primary_scene", "vqa_total_objects", "blip2_caption"]],
                      on="img_key", how="left")
    rows, sem_p, cap_p = [], [], []
    rng = np.random.default_rng(SEED)
    for q in rr.QUERIES:
        df["rel"] = df.apply(lambda r: rr.relevance_label(r, q), axis=1)
        df["sem_sc"] = df.apply(lambda r: rr.semantic_score(r, q), axis=1)
        df["cap_sc"] = df.apply(lambda r: rr.caption_score(r, q), axis=1)
        top_s, top_c = df.nlargest(10, "sem_sc"), df.nlargest(10, "cap_sc")
        sem_p.append(top_s.rel.mean()); cap_p.append(top_c.rel.mean())
        found = {}
        for meth, top in (("semantic", top_s), ("keyword", top_c)):
            for _, r in top.iterrows():
                f = found.setdefault(r.img_key, {"methods": [], "proxy": int(r.rel)})
                f["methods"].append(meth)
        keys = list(found); rng.shuffle(keys)
        for k in keys:
            rows.append({"item_id": f"{q['id']}|{k}", "query_id": q["id"], "query_text": q["text"], "ppn_id": k,
                         "methods": "+".join(found[k]["methods"]), "proxy_relevant": found[k]["proxy"]})
    ms, mc = float(np.mean(sem_p)), float(np.mean(cap_p))
    assert abs(ms - 1.0) < 1e-9 and abs(mc - 0.18) < 1e-9, f"proxy figures do not reproduce: {ms} {mc}"
    d = pd.DataFrame(rows)
    missing = [k for k in d.ppn_id if not (BASE / f"final_dataset/images/original/{k}.jpg").exists()]
    assert not missing, f"{len(missing)} pictures have no image file, e.g. {missing[:3]}"
    OUT.mkdir(parents=True, exist_ok=True)
    d[["item_id", "query_id", "query_text", "ppn_id"]].to_csv(OUT / "items.csv", index=False)
    d.to_csv(OUT / "key.csv", index=False)
    print(f"proxy reproduces: semantic {ms:.3f}, keyword {mc:.3f}; {len(d)} items to judge "
          f"({(d.methods == 'semantic+keyword').sum()} found by both)")


if __name__ == "__main__":
    main()
