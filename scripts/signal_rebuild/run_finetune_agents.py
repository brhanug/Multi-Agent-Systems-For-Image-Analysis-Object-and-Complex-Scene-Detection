#!/usr/bin/env python3
"""Fine-tune the scene matcher (SigLIP) and the describer (LLaVA-OneVision 7B) on the archive, cross-fitted by book.

Plan: self_annotation_package/more_experiments/PLAN_finetune.md (fixed before any run). The 550 books of the
labelled pool are split once into 5 folds; for fold k a model is trained on the class-labelled pictures of the
other folds and scores the pool pictures of fold k, so every pool picture gets an out-of-fold report in the
same form as the zero-shot one (SigLIP: 5 scene sigmoids + the highest "not a scene" sigmoid; LLaVA: 13
yes-strengths). evaluate_finetune.py then runs the thesis's pipeline on these reports.

    python run_finetune_agents.py folds
    python run_finetune_agents.py siglip --fold k --device g [--epochs 0]   # --epochs 0: zero-shot, same code
    python run_finetune_agents.py llava  --fold k --device g [--epochs 0]
Outputs: results/signal_rebuild/finetune/{folds.json, siglip_fold{k}.jsonl, llava_fold{k}.jsonl, ...}
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
OUT = SR / "finetune"
IMG = BASE / "final_dataset/images/restored"
SEED, K_FOLDS = 20261125, 5
ANSWERS = ["teaching", "family", "playing", "landscape", "drawing", "none"]
SCENES = ANSWERS[:5]
SCENE_PROMPTS = [f"a historical illustration of a {c} scene" for c in SCENES]
NEG_PROMPTS = ["a page of printed text", "a page of sheet music", "a decorative ornamental border", "a blank page",
               "a title page of a book", "a table of numbers", "a printed map"]
SCENE_Q = {"teaching": "a teaching or school scene", "family": "a family scene or a family portrait",
           "playing": "people playing or a crowd", "landscape": "a landscape", "drawing": "a drawing"}
OBJECT_Q = {"person": "a person", "child": "a child", "animal": "an animal", "building": "a building",
            "weapon": "a weapon", "vehicle": "a vehicle", "tree": "a tree", "text": "written or printed text"}
QUESTION = "Does this picture show {}? Answer yes or no."
MAX_SIDE = 640


def img_path(ppn_id):
    return IMG / f"{ppn_id}.jpg"


def pool():
    """The 1,250 pool pictures with their book and, for the 992, the scene class ("one rule")."""
    import pandas as pd
    import run_best_config_repeated as rb
    import run_scene_classes as rs
    sp = pd.read_csv(rb.SPLIT)[["ppn_id", "ppn", "has_scene"]]
    d, _ = rs.pool_labels()
    cls = dict(zip(d.ppn_id, d["one rule"]))
    sp["cls"] = sp.ppn_id.map(cls)
    return sp


def make_folds():
    sp = pool()
    books = np.array(sorted(sp.ppn.unique())); rng = np.random.default_rng(SEED); rng.shuffle(books)
    fold_of = {b: int(i % K_FOLDS) for i, b in enumerate(books)}
    out = {"seed": SEED, "folds": K_FOLDS, "books": len(books),
           "fold_of_picture": {r.ppn_id: fold_of[r.ppn] for r in sp.itertuples()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "folds.json").write_text(json.dumps(out, indent=1))
    sizes = np.bincount(list(out["fold_of_picture"].values()), minlength=K_FOLDS)
    print("pictures per fold", sizes.tolist())


def split(fold):
    sp = pool(); f = json.loads((OUT / "folds.json").read_text())["fold_of_picture"]
    sp["fold"] = sp.ppn_id.map(f)
    train = sp[(sp.fold != fold) & sp.cls.notna()]
    score = sp[sp.fold == fold]
    return train, score


def seed_all(s=0):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_image(p):
    from PIL import Image
    img = Image.open(p).convert("RGB")
    if max(img.size) > MAX_SIDE:
        r = MAX_SIDE / max(img.size); img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))))
    return img


# ------------------------------------------------------------------ SigLIP
def siglip(fold, device, epochs):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    seed_all(0)
    name = "google/siglip-so400m-patch14-384"
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float32).to(device)
    train, score = split(fold)
    with torch.no_grad():
        t = proc(text=SCENE_PROMPTS + NEG_PROMPTS, padding="max_length", return_tensors="pt").to(device)
        T = torch.nn.functional.normalize(model.get_text_features(**t), dim=-1)
    for p in model.parameters():
        p.requires_grad = False
    vm = model.vision_model
    train_params = list(vm.encoder.layers[-4:].parameters()) + list(vm.post_layernorm.parameters()) + list(vm.head.parameters())
    for p in train_params:
        p.requires_grad = True
    model.logit_scale.requires_grad = True; model.logit_bias.requires_grad = True
    train_params += [model.logit_scale, model.logit_bias]

    def logits(pix):
        f = torch.nn.functional.normalize(model.get_image_features(pixel_values=pix), dim=-1)
        return f @ T.T * model.logit_scale.exp() + model.logit_bias

    if epochs > 0:
        opt = torch.optim.AdamW(train_params, lr=1e-5, weight_decay=0.0)
        rows = list(train.itertuples()); bce = torch.nn.BCEWithLogitsLoss()
        model.train()
        for ep in range(epochs):
            random.shuffle(rows); tot = 0.0
            for i in range(0, len(rows), 16):
                b = rows[i:i + 16]
                pix = proc(images=[Image.open(img_path(r.ppn_id)).convert("RGB") for r in b], return_tensors="pt")["pixel_values"].to(device)
                y = torch.zeros(len(b), 6, device=device)
                for j, r in enumerate(b):
                    if r.cls == "none": y[j, 5] = 1.0
                    else: y[j, SCENES.index(r.cls)] = 1.0
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    L = logits(pix)
                L = L.float()
                out = torch.cat([L[:, :5], L[:, 5:].max(1, keepdim=True).values], 1)
                loss = bce(out, y)
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
            print(f"[siglip fold {fold}] epoch {ep + 1}/{epochs} loss {tot / len(rows):.4f}", flush=True)
    model.eval()
    res = []
    with torch.no_grad():
        rows = list(score.itertuples())
        for i in range(0, len(rows), 32):
            b = rows[i:i + 32]
            pix = proc(images=[Image.open(img_path(r.ppn_id)).convert("RGB") for r in b], return_tensors="pt")["pixel_values"].to(device)
            P = torch.sigmoid(logits(pix).float()).cpu().numpy()
            for r, p in zip(b, P):
                res.append({"image_id": r.ppn_id, "scene": {s: round(float(p[j]), 6) for j, s in enumerate(SCENES)},
                            "neg": round(float(p[5:].max()), 6)})
    tag = "siglip" if epochs > 0 else "siglip_zeroshot"
    with open(OUT / f"{tag}_fold{fold}.jsonl", "w") as fh:
        for r in res: fh.write(json.dumps(r) + "\n")
    print(f"[{tag} fold {fold}] trained on {len(train)}, scored {len(res)}", flush=True)


# ------------------------------------------------------------------ LLaVA-OneVision
def llava(fold, device, epochs):
    import torch
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
    seed_all(0)
    name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    proc = AutoProcessor.from_pretrained(name)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16).to(device)
    tok = proc.tokenizer
    vocab = [tok.decode([i]).strip().lower() for i in range(len(tok))]
    yes_ids = torch.tensor([i for i, v in enumerate(vocab) if v == "yes"], device=device)
    no_ids = torch.tensor([i for i, v in enumerate(vocab) if v == "no"], device=device)
    train, score = split(fold)

    def prompt(q):
        return proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION.format(q)}]}],
                                        add_generation_prompt=True)

    if epochs > 0:
        from peft import LoraConfig, get_peft_model
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                         target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)")
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
        ans_yes = tok.encode("Yes", add_special_tokens=False)[0]; ans_no = tok.encode("No", add_special_tokens=False)[0]
        ex = [(r.ppn_id, s, r.cls == s) for r in train.itertuples() for s in SCENES]
        rng = random.Random(0); rng.shuffle(ex)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
        accum, t0 = 8, time.time(); model.train(); opt.zero_grad(); run = 0.0
        for n, (iid, s, y) in enumerate(ex, 1):
            inp = proc(images=load_image(img_path(iid)), text=prompt(SCENE_Q[s]), return_tensors="pt").to(device, torch.bfloat16)
            out = model(**inp)
            logit = out.logits[0, -1].float()
            loss = torch.nn.functional.cross_entropy(logit[None], torch.tensor([ans_yes if y else ans_no], device=device)) / accum
            loss.backward(); run += float(loss) * accum
            if n % accum == 0 or n == len(ex):
                opt.step(); opt.zero_grad()
            if n % 200 == 0:
                el = time.time() - t0
                print(f"[llava fold {fold}] {n}/{len(ex)} loss {run / 200:.4f}  {n / el:.2f} ex/s  eta {(len(ex) - n) / (n / el) / 60:.0f} min", flush=True); run = 0.0
        model = model.merge_and_unload()
    model.eval()
    qs = list(SCENE_Q.items()) + list(OBJECT_Q.items())
    tag = "llava" if epochs > 0 else "llava_zeroshot"
    path = OUT / f"{tag}_fold{fold}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["image_id"] for l in open(path)}
    # The picture part of the prompt is run once and its cache reused for the 13 questions (about 10x faster
    # than one full pass per question; bf16 attention makes scores differ from full passes by up to ~0.025,
    # so the zero-shot reference is scored with this same code).
    k0 = prompt("x").index("Does"); prefix = prompt("x")[:k0]
    with open(path, "a") as fh, torch.no_grad():
        for r in score.itertuples():
            if r.ppn_id in done: continue
            img = load_image(img_path(r.ppn_id)); rec = {}
            ip = proc(images=img, text=prefix, return_tensors="pt").to(device, torch.bfloat16)
            past = model(**ip, use_cache=True).past_key_values; L = ip["input_ids"].shape[1]
            for c, q in qs:
                ids = proc(images=img, text=prompt(q), return_tensors="pt")["input_ids"][0]
                assert torch.equal(ids[:L].to(device), ip["input_ids"][0])
                p = torch.softmax(model(input_ids=ids[L:].to(device)[None], past_key_values=past, use_cache=True).logits[0, -1].float(), -1)
                past.crop(L)
                y, n_ = float(p[yes_ids].sum()), float(p[no_ids].sum())
                rec[c] = None if y + n_ == 0 else round(y / (y + n_), 6)
            fh.write(json.dumps({"image_id": r.ppn_id, "p_yes": rec}) + "\n"); fh.flush()
    print(f"[{tag} fold {fold}] trained on {len(train)} pictures, scored {len(score)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["folds", "siglip", "llava"])
    ap.add_argument("--fold", type=int, default=0); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    if a.what == "folds":
        return make_folds()
    if a.what == "siglip":
        return siglip(a.fold, a.device, 5 if a.epochs is None else a.epochs)
    return llava(a.fold, a.device, 1 if a.epochs is None else a.epochs)


if __name__ == "__main__":
    raise SystemExit(main())
