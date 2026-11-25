#!/usr/bin/env python3
"""Kosmos-2.5 grounded OCR over the whole corpus, sharded across GPUs. Resumable."""
import argparse, glob, json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

BASE = Path("/data/brhanu/thesis_project")
IMG_GLOB = str(BASE / "final_dataset/images/restored/**/*.jpg")
OUTDIR = BASE / "results/signal_rebuild/kosmos_full"
MODEL = "microsoft/kosmos-2.5"
MAX_SIDE = 1024


def image_id(p):
    p = Path(p)
    return f"{p.parent.name}/{p.stem}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=None,
                        help="Write under this directory instead of results/signal_rebuild. "
                             "Lets a re-run be compared against the committed outputs without "
                             "overwriting them.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    if args.out_root:
        _r = Path(args.out_root)
        _r.mkdir(parents=True, exist_ok=True)
        globals()['OUTDIR'] = _r / OUTDIR.name
        print(f'writing to {OUTDIR}', flush=True)
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForVision2Seq

    paths = sorted(glob.glob(IMG_GLOB, recursive=True))
    paths = [p for i, p in enumerate(paths) if i % args.num_shards == args.shard]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / f"shard_{args.shard}.jsonl"
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["image_id"])
                except Exception:
                    pass
    todo = [p for p in paths if image_id(p) not in done]
    if args.max_images:
        todo = todo[: args.max_images]
    print(f"[kosmos {args.shard}] own {len(paths)} | done {len(done)} | todo {len(todo)}", flush=True)
    if not todo:
        return 0

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL, trust_remote_code=True, dtype=torch.float16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    t0 = time.time(); n = 0
    with open(out_path, "a") as fh:
        for p in todo:
            txt = ""
            try:
                img = Image.open(p).convert("RGB")
                if max(img.size) > MAX_SIDE:
                    r = MAX_SIDE / max(img.size)
                    img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))))
                inputs = proc(text="<md>", images=img, return_tensors="pt")
                inputs = {k: (v.to(dev, torch.float16) if v.dtype.is_floating_point else v.to(dev))
                          for k, v in inputs.items()}
                with torch.no_grad():
                    ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                txt = proc.batch_decode(ids, skip_special_tokens=True)[0]
            except Exception as e:
                txt = ""
            fh.write(json.dumps({"image_id": image_id(p), "kosmos_output": txt}) + "\n")
            n += 1
            if n % 25 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"[kosmos {args.shard}] {n}/{len(todo)} {n/el:.2f} img/s "
                      f"eta {(len(todo)-n)/max(n/el,1e-9)/60:.1f} min", flush=True)
    el = time.time() - t0
    print(f"[kosmos {args.shard}] DONE {n} in {el/60:.1f} min ({n/el:.2f} img/s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
