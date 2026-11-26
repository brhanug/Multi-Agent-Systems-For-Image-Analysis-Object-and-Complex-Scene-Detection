#!/usr/bin/env python3
"""Run the three agents on the running example and record what happens inside them, for the defence talk.

For the restored running example (PPN1756648816/00000272_0) this records the real tensor shapes and values
at each step, and draws small figures from those tensors:
  YOLOv11m (the retrained object finder): the letterboxed input, the three feature maps the detection head
      reads, the raw head output, the candidates above the score floor, the boxes left after NMS, and the
      best score per class (checked against new_student_class_scores.jsonl).
  SigLIP so400m-patch14-384 (the scene matcher): the pixel tensor, the 729 patch tokens, the pooled image
      embedding, the 12 text embeddings, the learned scale and bias, cosine, logit and sigmoid per sentence
      (checked against siglip_scene_full.jsonl).
  LLaVA-OneVision 7B (the describer): the crops, vision tokens per crop, projected tokens, prompt length,
      the next-token distribution for one question (top 10 tokens) and the 13 yes-strengths (compared with
      the stored scores).
Writes presentation/defence_assets/model_internals.json and *_internals_*.png.

    python extract_model_internals.py [--yolo-device cuda:1] [--llava-device cuda:2]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BASE = Path("/data/brhanu/thesis_project")
OUT = BASE / "presentation/defence_assets"
SR = BASE / "results/signal_rebuild"
EXAMPLE = "PPN1756648816/00000272_0"
IMG = BASE / "final_dataset/images/restored" / f"{EXAMPLE}.jpg"
DETECTOR = BASE / "runs/detect/yolo11_gdino_consensus/weights/best.pt"
CLASSES = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]
SCENES = ["teaching", "family", "playing", "landscape", "drawing"]
SCENE_PROMPTS = [f"a historical illustration of a {c} scene" for c in SCENES]
NEG_PROMPTS = ["a page of printed text", "a page of sheet music", "a decorative ornamental border", "a blank page",
               "a title page of a book", "a table of numbers", "a printed map"]
SCENE_Q = {"teaching": "a teaching or school scene", "family": "a family scene or a family portrait",
           "playing": "people playing or a crowd", "landscape": "a landscape", "drawing": "a drawing"}
OBJECT_Q = {"person": "a person", "child": "a child", "animal": "an animal", "building": "a building",
            "weapon": "a weapon", "vehicle": "a vehicle", "tree": "a tree", "text": "written or printed text"}
QUESTION = "Does this picture show {}? Answer yes or no."


def heat(a, path, cmap="magma", size=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = np.asarray(a, float); a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    img = (plt.get_cmap(cmap)(a)[..., :3] * 255).astype(np.uint8)
    im = Image.fromarray(img)
    if size: im = im.resize(size, Image.NEAREST)
    im.save(path)


def shape(t):
    return list(t.shape)


def yolo(device):
    from ultralytics import YOLO
    m = YOLO(str(DETECTOR)); net = m.model
    cap = {}
    head = net.model[-1]
    def pre(mod, inp):
        cap["input"] = inp[0].detach().cpu()
    def post(mod, inp, out):
        cap["feats"] = [x.detach().cpu() for x in inp[0]]
        cap["raw"] = (out[0] if isinstance(out, (tuple, list)) else out).detach().cpu()
    h1 = net.register_forward_pre_hook(pre); h2 = head.register_forward_hook(post)   # the last call is the real picture
    r = m.predict(source=str(IMG), conf=0.01, iou=0.7, device=device, verbose=False)[0]
    h1.remove(); h2.remove()
    raw = cap["raw"][0]                     # (4 + nc, N)
    nc = raw.shape[0] - 4
    cls_scores = raw[4:]
    best = cls_scores.max(0).values
    dets = [{"class": CLASSES[int(c)], "score": round(float(s), 4), "box_xyxy": [round(float(v), 1) for v in b]}
            for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
    per_class = {c: 0.0 for c in CLASSES}
    for d in dets: per_class[d["class"]] = max(per_class[d["class"]], d["score"])
    stored = None
    for line in open(SR / "class_scores/new_student_class_scores.jsonl"):
        rr = json.loads(line)
        if rr["image_id"] == EXAMPLE: stored = rr["max_conf"]; break
    for k, f in enumerate(cap["feats"]):
        heat(f[0].abs().mean(0).numpy(), OUT / f"yolo_internals_p{k + 3}.png", size=(f.shape[3] * 4, f.shape[2] * 4))
    inp = cap["input"][0].permute(1, 2, 0).numpy()
    Image.fromarray((inp * 255).clip(0, 255).astype(np.uint8)).save(OUT / "yolo_internals_input.png")
    return {"original_size_wh": list(Image.open(IMG).size),
            "network_input": shape(cap["input"]),
            "feature_maps": [shape(f) for f in cap["feats"]],
            "strides": [8, 16, 32],
            "raw_head_output": shape(cap["raw"]),
            "candidates": int(raw.shape[1]),
            "candidates_score_ge_0.01": int((best >= 0.01).sum()),
            "after_nms": len(dets), "detections": dets,
            "best_score_per_class": {k: round(v, 4) for k, v in per_class.items()},
            "stored_max_conf": stored, "classes": nc}


def siglip(device):
    from transformers import AutoModel, AutoProcessor
    name = "google/siglip-so400m-patch14-384"
    proc = AutoProcessor.from_pretrained(name); model = AutoModel.from_pretrained(name, dtype=torch.float32).to(device).eval()
    img = Image.open(IMG).convert("RGB")
    cap = {}
    def emb(m, i, o):
        cap["patch_emb"] = o.detach().cpu()
    hk = model.vision_model.embeddings.register_forward_hook(emb)
    with torch.no_grad():
        px = proc(images=img, return_tensors="pt")["pixel_values"].to(device)
        vo = model.vision_model(pixel_values=px)
        v = torch.nn.functional.normalize(vo.pooler_output, dim=-1)
        t = proc(text=SCENE_PROMPTS + NEG_PROMPTS, padding="max_length", return_tensors="pt").to(device)
        to = model.text_model(input_ids=t["input_ids"])
        u = torch.nn.functional.normalize(to.pooler_output, dim=-1)
        scale, bias = float(model.logit_scale.exp()), float(model.logit_bias)
        cos = (v @ u.T)[0]; logit = cos * scale + bias; sig = torch.sigmoid(logit)
    hk.remove()
    last = vo.last_hidden_state[0].detach().cpu()
    heat(last.norm(dim=-1).reshape(27, 27).numpy(), OUT / "siglip_internals_patchnorm.png", size=(216, 216))
    heat(v[0].cpu().reshape(32, 36).numpy(), OUT / "siglip_internals_image_embedding.png", cmap="coolwarm", size=(288, 256))
    stored = None
    for line in open(SR / "siglip_scene_full.jsonl"):
        r = json.loads(line); iid = r["image_id"]; iid = iid if "/" in iid else iid.replace("_", "/", 1)
        if iid == EXAMPLE: stored = r["sigmoid"]; break
    sents = SCENE_PROMPTS + NEG_PROMPTS
    return {"pixel_values": shape(px), "patch_embeddings": shape(cap["patch_emb"]), "last_hidden_state": shape(vo.last_hidden_state),
            "image_embedding": shape(v), "image_embedding_first8": [round(float(x), 4) for x in v[0, :8]],
            "text_input_ids": shape(t["input_ids"]), "text_embeddings": shape(u),
            "text_embedding_first8_family": [round(float(x), 4) for x in u[1, :8]],
            "logit_scale_exp": round(scale, 3), "logit_bias": round(bias, 3),
            "per_sentence": [{"sentence": s, "cos": round(float(c), 4), "logit": round(float(l), 3), "sigmoid": round(float(p), 4)}
                             for s, c, l, p in zip(sents, cos, logit, sig)],
            "report_6": {**{s: round(float(sig[i]), 4) for i, s in enumerate(SCENES)}, "no scene (max of 7)": round(float(sig[5:].max()), 4)},
            "stored_scene_sigmoid": stored}


def llava(device):
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
    name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    proc = AutoProcessor.from_pretrained(name)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
    tok = proc.tokenizer
    img = Image.open(IMG).convert("RGB")
    if max(img.size) > 640:
        r = 640 / max(img.size); img = img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))))
    def prompt(q):
        return proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION.format(q)}]}], add_generation_prompt=True)
    core = getattr(model, "model", model)
    cap = {}
    def vis(m, i, o):
        cap["vision"] = (o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]).detach().float().cpu()
    def prj(m, i, o):
        cap["proj"] = o.detach().float().cpu()
    h1 = core.vision_tower.register_forward_hook(vis); h2 = core.multi_modal_projector.register_forward_hook(prj)
    vocab = [tok.decode([i]).strip().lower() for i in range(len(tok))]
    yes_ids = [i for i, x in enumerate(vocab) if x == "yes"]; no_ids = [i for i, x in enumerate(vocab) if x == "no"]
    res = {}
    with torch.no_grad():
        inp = proc(images=img, text=prompt(OBJECT_Q["child"]), return_tensors="pt").to(device, torch.bfloat16)
        out = model(**inp)
        h1.remove(); h2.remove()
        logits = out.logits[0, -1].float(); p = torch.softmax(logits, -1)
        top = torch.topk(p, 10)
        img_tok = getattr(model.config, "image_token_index", None) or getattr(model.config, "image_token_id", None)
        res = {"resized_image_wh": list(img.size), "pixel_values": shape(inp["pixel_values"]), "image_sizes": inp["image_sizes"].tolist(),
               "vision_tower_output": shape(cap["vision"]), "projector_output": shape(cap["proj"]),
               "prompt_tokens": int(inp["input_ids"].shape[1]), "image_tokens_in_prompt": int((inp["input_ids"] == img_tok).sum()),
               "logits": shape(out.logits), "vocab_size": int(out.logits.shape[-1]),
               "question": QUESTION.format(OBJECT_Q["child"]),
               "top10_next_tokens": [{"token": tok.decode([int(i)]), "p": round(float(v), 4)} for v, i in zip(top.values, top.indices)],
               "p_yes": round(float(p[yes_ids].sum()), 4), "p_no": round(float(p[no_ids].sum()), 4),
               "yes_token_variants": [tok.decode([i]) for i in yes_ids], "no_token_variants": [tok.decode([i]) for i in no_ids]}
        res["yes_strength"] = round(res["p_yes"] / (res["p_yes"] + res["p_no"]), 4)
        all13 = {}
        for c, q in list(SCENE_Q.items()) + list(OBJECT_Q.items()):
            i2 = proc(images=img, text=prompt(q), return_tensors="pt").to(device, torch.bfloat16)
            pp = torch.softmax(model(**i2).logits[0, -1].float(), -1)
            y, n = float(pp[yes_ids].sum()), float(pp[no_ids].sum()); all13[c] = round(y / (y + n), 4)
        res["yes_strength_13"] = all13
    v = cap["vision"]; heat(v[0].norm(dim=-1).reshape(27, 27).numpy(), OUT / "llava_internals_crop0_tokens.png", size=(216, 216))
    pv = inp["pixel_values"][0].float().cpu().numpy(); ip = proc.image_processor
    mean, std = np.array(ip.image_mean)[:, None, None], np.array(ip.image_std)[:, None, None]
    crops = [Image.fromarray(((c * std + mean).transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)) for c in pv]
    W = Image.new("RGB", (len(crops) * 210 - 10, 200), "white")
    for k, c in enumerate(crops): W.paste(c.resize((200, 200)), (k * 210, 0))
    W.save(OUT / "llava_internals_crops.png")
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--yolo-device", default="cuda:1"); ap.add_argument("--llava-device", default="cuda:2")
    ap.add_argument("--only", choices=["yolo", "siglip", "llava"], default=None)
    a = ap.parse_args()
    path = OUT / "model_internals.json"
    out = json.loads(path.read_text()) if path.exists() else {"picture": EXAMPLE}
    if a.only in (None, "yolo"): out["yolo"] = yolo(a.yolo_device)
    if a.only in (None, "siglip"): out["siglip"] = siglip(a.yolo_device)
    if a.only in (None, "llava"): out["llava"] = llava(a.llava_device)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    raise SystemExit(main())
