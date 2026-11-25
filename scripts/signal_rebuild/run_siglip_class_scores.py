#!/usr/bin/env python3
"""Per-class SigLIP scores over the whole corpus (Phase 1, C1).

SigLIP was only ever asked about the five scenes. Its picture embeddings for all
12,110 pictures are already stored, so a per-class score needs only the text side:
one prompt per class, scored exactly as SigLIP scores any image-text pair
(sigmoid of scaled cosine plus bias). Evaluated per class on the 300 human-annotated
pictures.

    python run_siglip_class_scores.py [--verify]
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
EMB = BASE / "results/signal_rebuild/siglip_embeddings.npz"
TRUTH = BASE / "results/signal_rebuild/class_scores/human_class_truth_300.csv"
OUT = BASE / "results/signal_rebuild/class_scores/siglip_class_scores.npz"
SUMMARY = BASE / "results/signal_rebuild/class_scores/siglip_class_scores_summary.json"
MODEL = "google/siglip-so400m-patch14-384"
NP = {"person": "a person", "child": "a child", "animal": "an animal", "horse": "a horse", "building": "a building",
      "weapon": "a weapon", "vehicle": "a vehicle", "tree": "a tree", "text": "printed text", "clothing": "clothing",
      "furniture": "furniture", "hat": "a hat"}
TEMPLATE = "a historical illustration of {}"


def scores():
    import torch
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    tx = proc(text=[TEMPLATE.format(v) for v in NP.values()], padding="max_length", return_tensors="pt")
    with torch.no_grad():
        t = torch.nn.functional.normalize(model.get_text_features(**tx).float(), dim=-1).numpy()
        scale, bias = float(model.logit_scale.exp()), float(model.logit_bias)
    e = np.load(EMB, allow_pickle=True)
    logits = e["vecs"].astype(np.float64) @ t.T.astype(np.float64) * scale + bias
    return e["ids"], 1 / (1 + np.exp(-logits))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    ids, S = scores()
    classes = list(NP)
    truth = pd.read_csv(TRUTH, keep_default_na=False).set_index("image_id")
    pos = {i: k for k, i in enumerate(ids)}
    rows = [pos[i] for i in truth.index]
    per = []
    for k, c in enumerate(classes):
        y = truth[c].values; s = S[rows, k]
        per.append({"class": c, "prompt": TEMPLATE.format(NP[c]), "positives_in_300": int(y.sum()),
                    "auroc_on_300": round(float(roc_auc_score(y, s)), 4) if 0 < y.sum() < len(y) else None,
                    "corpus_median_score": round(float(np.median(S[:, k])), 5)})
    summary = {"experiment": "SigLIP per-class scores (C1)", "model": MODEL, "pictures": int(len(ids)), "per_class": per}
    if a.verify:
        same = json.loads(SUMMARY.read_text()) == json.loads(json.dumps(summary))
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, ids=ids, classes=np.array(classes), scores=S.astype(np.float32))
    SUMMARY.write_text(json.dumps(summary, indent=2))
    for p in per: print(p)


if __name__ == "__main__":
    raise SystemExit(main())
