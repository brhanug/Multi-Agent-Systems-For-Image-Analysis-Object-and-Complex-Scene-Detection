#!/usr/bin/env python3
"""AR5 pilot: do repetition controls fix Kosmos-2.5's looping without breaking its reading?

Half the corpus's Kosmos output is degenerate repetition (49.8%, unique/total words < 0.3), which
looks like a generation setting rather than a model limit: the full-corpus run used greedy decoding
up to 512 new tokens with no repetition control. The architecture track suggests forbidding repeated
n-grams and penalising repetition. Those controls can also suppress repeats that are really on the
page, so before spending hours on 12,110 pictures this tries them on the 53 pictures of Task D,
whose text was transcribed by hand.

  as run     greedy, max_new_tokens 512                           (the full-corpus setting)
  no-repeat  greedy, max_new_tokens 512, no_repeat_ngram_size 4
  suggested  greedy, max_new_tokens 256, no_repeat_ngram_size 4, repetition_penalty 1.2

Measured on each: the share of degenerate outputs; word error rate against the hand transcription
on the pictures that have text; and how well a repetition-discounted substance score separates
pictures with text from pictures without (Kosmos's proposed role is negative evidence, "this is a
text page", so that separation is the number that decides its role).

Generations are cached, so the scoring is exactly reproducible: --verify re-scores the cache.

    python run_kosmos_generation_pilot.py --device 2      # generate and score
    python run_kosmos_generation_pilot.py --verify         # re-score the cache
"""
import argparse, json, re, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
BASE = Path("/data/brhanu/thesis_project")
TASK = BASE / "self_annotation_package/task_d_ocr_ground_truth"
SR = BASE / "results/signal_rebuild"
CACHE = SR / "kosmos_generation_pilot_outputs.jsonl"
OUT = SR / "kosmos_generation_pilot.json"
MODEL, MAX_SIDE = "microsoft/kosmos-2.5", 1024
CONFIGS = {
    "as run":    {"max_new_tokens": 512},
    "no-repeat": {"max_new_tokens": 512, "no_repeat_ngram_size": 4},
    "suggested": {"max_new_tokens": 256, "no_repeat_ngram_size": 4, "repetition_penalty": 1.2},
}
PLACEHOLDERS = {"image_tag", "link_solt", "link_slot", "image", "tag"}


def words(t):
    t = re.sub(r"[*#>\[\]()!_`|]", " ", str(t).lower())
    return [w for w in t.split() if w]


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    if not r: return None
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / len(r)


def repetition(t):
    w = words(t)
    return 1.0 - len(set(w)) / len(w) if w else 0.0


def substance(t):
    """Distinct real words, discounted for repetition: long looping output scores low."""
    w = [x for x in words(t) if x.isalpha() and len(x) >= 2 and x not in PLACEHOLDERS]
    return len(set(w)) * (1.0 - repetition(t))


def generate(device):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForVision2Seq
    sheet = pd.read_csv(TASK / "labeling_sheet.csv")
    model = AutoModelForVision2Seq.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float16).to(device).eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    CACHE.unlink(missing_ok=True)
    with open(CACHE, "a") as fh:
        for name, cfg in CONFIGS.items():
            t0 = time.time()
            for _, row in sheet.iterrows():
                img = Image.open(TASK / "images" / row.image_file).convert("RGB")
                if max(img.size) > MAX_SIDE:
                    r = MAX_SIDE / max(img.size)
                    img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))))
                inputs = proc(text="<md>", images=img, return_tensors="pt")
                inputs = {k: (v.to(device, torch.float16) if v.dtype.is_floating_point else v.to(device))
                          for k, v in inputs.items()}
                with torch.no_grad():
                    ids = model.generate(**inputs, do_sample=False, **cfg)
                fh.write(json.dumps({"config": name, "id": int(row.id),
                                     "text": proc.batch_decode(ids, skip_special_tokens=True)[0]}) + "\n")
            fh.flush()
            print(f"{name}: {len(sheet)} pictures in {(time.time() - t0) / 60:.1f} min", flush=True)


def score():
    gold = pd.read_csv(TASK / "labeling_sheet_LS_export.csv").set_index("id").ground_truth_transcription
    has_text = gold.fillna("NO_TEXT").str.strip().str.upper().ne("NO_TEXT")
    out = {"experiment": "AR5 pilot -- repetition controls for Kosmos-2.5 on the 53 hand-transcribed pictures",
           "configs": CONFIGS, "pictures": int(len(gold)), "with_text": int(has_text.sum()),
           "degenerate_definition": "unique/total words < 0.3, as in the thesis's corpus-wide count",
           "results": {}}
    rows = [json.loads(l) for l in open(CACHE)]
    for name in CONFIGS:
        by = {r["id"]: r["text"] for r in rows if r["config"] == name}
        ids = sorted(by)
        rep = np.array([repetition(by[i]) for i in ids])
        nw = np.array([len(words(by[i])) for i in ids])
        deg = (nw > 0) & (1 - rep < 0.3)
        w = [wer(gold[i], by[i]) for i in ids if has_text[i]]
        w = np.array([x for x in w if x is not None])
        sub = np.array([substance(by[i]) for i in ids])
        y = has_text.loc[ids].astype(int).values
        out["results"][name] = {
            "degenerate_share": round(float(deg.mean()), 4),
            "median_words_emitted": int(np.median(nw)),
            "wer_on_pictures_with_text": {"median": round(float(np.median(w)), 4), "mean": round(float(np.mean(w)), 4),
                                          "share_below_0.5": round(float((w < 0.5).mean()), 4)},
            "text_vs_no_text_auroc_of_substance": round(float(roc_auc_score(y, sub)), 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(score()))
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    dev = a.device if a.device.startswith("cuda") else f"cuda:{a.device}"
    generate(dev)
    out = score()
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n{out['pictures']} pictures, {out['with_text']} with text\n")
    print(f"{'config':10s} {'degenerate':>10s} {'median words':>12s} {'median WER':>10s} {'WER<0.5':>8s} {'text AUROC':>10s}")
    for n, r in out["results"].items():
        print(f"{n:10s} {100*r['degenerate_share']:9.1f}% {r['median_words_emitted']:12d} "
              f"{r['wer_on_pictures_with_text']['median']:10.3f} {100*r['wer_on_pictures_with_text']['share_below_0.5']:7.1f}% "
              f"{r['text_vs_no_text_auroc_of_substance']:10.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
