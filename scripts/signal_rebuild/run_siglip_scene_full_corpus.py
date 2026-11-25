import os
"""
Score all 12,110 deployment images with SigLIP over the five scene categories.

The deployed scene agent was a manifest fallback taking two values ({0.5, 1.0}),
and the only real SigLIP scores on disk covered 231 images. This closes that gap
so the scene signal is a genuine, continuous, independently-measured quantity.

Writes one JSON object per line: image_id, per-category probabilities, and the
summary scalars the rebuilt ensemble uses.
"""
import json, os, sys, glob
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

BASE = Path("/data/brhanu/thesis_project")
IMG_DIR = BASE / "final_dataset/images/restored"
OUT = BASE / "results/signal_rebuild/siglip_scene_full.jsonl"
MODEL = "google/siglip-so400m-patch14-384"
BATCH = 64

CATEGORIES = ["teaching", "family", "playing", "landscape", "drawing"]
PROMPTS = [f"a historical illustration of a {c} scene" for c in CATEGORIES]


# A re-run can be pointed at a shadow directory so the committed outputs are not
# overwritten and the two can be compared.
_root = os.environ.get("OUT_ROOT")
if _root:
    Path(_root).mkdir(parents=True, exist_ok=True)
    OUT = Path(_root) / OUT.name


def main():
    paths = sorted(glob.glob(str(IMG_DIR / "**" / "*.jpg"), recursive=True))
    print(f"images found: {len(paths)}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if OUT.exists():
        with open(OUT) as f:
            for line in f:
                try: done.add(json.loads(line)["image_id"])
                except Exception: pass
        print(f"resuming, already scored: {len(done)}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
    txt = proc(text=PROMPTS, padding="max_length", return_tensors="pt").to(dev)
    with torch.no_grad():
        temb = model.get_text_features(**txt)
        temb = temb / temb.norm(dim=-1, keepdim=True)

    def image_id(p):
        return f"{Path(p).parent.name}_{Path(p).stem}"

    todo = [p for p in paths if image_id(p) not in done]
    print(f"to score: {len(todo)}", flush=True)
    fh = open(OUT, "a")
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        imgs, ids = [], []
        for p in chunk:
            try:
                imgs.append(Image.open(p).convert("RGB")); ids.append(image_id(p))
            except Exception as e:
                print(f"  skip {p}: {e}", flush=True)
        if not imgs:
            continue
        px = proc(images=imgs, return_tensors="pt").to(dev, torch.float16)
        with torch.no_grad():
            iemb = model.get_image_features(pixel_values=px["pixel_values"])
            iemb = iemb / iemb.norm(dim=-1, keepdim=True)
            logits = (iemb @ temb.T).float() * model.logit_scale.exp().float() + model.logit_bias.float()
            probs = torch.sigmoid(logits)          # SigLIP: per-pair sigmoid, not softmax
            soft = torch.softmax(logits, dim=-1)   # relative preference across the five
        for j, iid in enumerate(ids):
            p_sig = probs[j].tolist(); p_soft = soft[j].tolist()
            fh.write(json.dumps({
                "image_id": iid,
                "sigmoid": {c: round(v, 6) for c, v in zip(CATEGORIES, p_sig)},
                "softmax": {c: round(v, 6) for c, v in zip(CATEGORIES, p_soft)},
                "max_sigmoid": round(max(p_sig), 6),
                "max_softmax": round(max(p_soft), 6),
                "argmax": CATEGORIES[int(max(range(5), key=lambda k: p_sig[k]))],
            }) + "\n")
        fh.flush()
        if (i // BATCH) % 20 == 0:
            print(f"  {i + len(chunk)}/{len(todo)}", flush=True)
    fh.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
