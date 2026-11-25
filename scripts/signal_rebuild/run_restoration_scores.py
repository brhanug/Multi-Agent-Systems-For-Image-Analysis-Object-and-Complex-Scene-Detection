#!/usr/bin/env python3
"""E5 of self_annotation_package/more_experiments/PLAN.md: SigLIP on the original scans and on the restored ones.

Scores the 992 Task S pictures and the 693 labelled random archive pictures twice, from
final_dataset/images/original and from final_dataset/images/restored, with the same model, the same
five scene prompts and the same seven "no scene" prompts as the corrected design. Both conditions go
through this one code path, so the only difference is the picture. A sanity check compares the
restored scene sigmoids with the stored ones (siglip_scene_full.jsonl).

    CUDA_VISIBLE_DEVICES=0 python run_restoration_scores.py
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
OUT = SR / "restoration_scores.npz"
MODEL = "google/siglip-so400m-patch14-384"
SCENES = ["teaching", "family", "playing", "landscape", "drawing"]
SCENE_PROMPTS = [f"a historical illustration of a {c} scene" for c in SCENES]
NEG_PROMPTS = ["a page of printed text", "a page of sheet music", "a decorative ornamental border", "a blank page",
               "a title page of a book", "a table of numbers", "a printed map"]


def picture_ids():
    import run_scene_classes as rs, run_final_check as fc, run_confirm_check as cc
    d, _ = rs.pool_labels()
    t1, _ = fc.labels()
    t2, _ = cc.labelled(cc.CC / "items.csv", cc.CC / "labels.csv")
    return sorted(set(d.ppn_id) | set(t1.ppn_id) | set(t2.ppn_id))


def main():
    ids = picture_ids()
    dev = "cuda"
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
    txt = proc(text=SCENE_PROMPTS + NEG_PROMPTS, padding="max_length", return_tensors="pt").to(dev)
    with torch.no_grad():
        temb = model.get_text_features(**txt); temb = temb / temb.norm(dim=-1, keepdim=True)
    out = {"ids": np.array(ids)}
    for cond in ("restored", "original"):
        S = np.full((len(ids), len(SCENE_PROMPTS) + len(NEG_PROMPTS)), np.nan)
        for i in range(0, len(ids), 32):
            chunk = ids[i:i + 32]; imgs, keep = [], []
            for k, iid in enumerate(chunk):
                p = BASE / "final_dataset/images" / cond / f"{iid}.jpg"
                if p.exists():
                    imgs.append(Image.open(p).convert("RGB")); keep.append(i + k)
            if not imgs: continue
            px = proc(images=imgs, return_tensors="pt").to(dev, torch.float16)
            with torch.no_grad():
                e = model.get_image_features(pixel_values=px["pixel_values"]); e = e / e.norm(dim=-1, keepdim=True)
                lg = (e @ temb.T).float() * model.logit_scale.exp().float() + model.logit_bias.float()
                S[keep] = torch.sigmoid(lg).cpu().numpy()
            if (i // 32) % 10 == 0: print(cond, i, flush=True)
        out[cond] = S
        print(cond, "missing", int(np.isnan(S[:, 0]).sum()), flush=True)
    stored = {}
    for line in open(SR / "siglip_scene_full.jsonl"):
        r = json.loads(line); iid = r["image_id"].replace("_", "/", 1); stored[iid] = [r["sigmoid"][c] for c in SCENES]
    pos = {iid: k for k, iid in enumerate(ids)}
    chk = [iid for iid in ids if iid in stored][:50]
    diff = np.array([np.abs(np.array(stored[i]) - out["restored"][pos[i], :5]).max() for i in chk])
    out["sanity_max_abs_diff"] = np.array([float(diff.max())])
    print("sanity: largest difference to stored restored sigmoids over 50 pictures", float(diff.max()), flush=True)
    np.savez(OUT, **out)
    print("DONE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
