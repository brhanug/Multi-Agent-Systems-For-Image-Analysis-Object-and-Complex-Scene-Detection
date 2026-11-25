#!/usr/bin/env python3
"""Each Tier 1 model, and the fusions, on the 300 human-annotated pictures (Prof. Mandl, experiment 1).

The 300 pictures of the spatial audit carry, from one human rater, a scene category (or none)
and bounding boxes over ten object classes. This asks, model by model, where each one works
and where it fails, on four tasks:

  1. scene presence       is there a scene at all          (240 yes / 60 no)
  2. scene category       which of the five kinds          (240 pictures)
  3. object presence      is class c anywhere in the picture, per class
  4. object boxes         does the detector draw the right boxes (mAP50 per class)

Two leakage checks shape how the numbers must be read, and both are applied here.
  - 136 of the 300 are in the TRAIN split of the pseudo-labels the detector learned from, so
    every detector figure is reported for all 300, the 136 seen and the 164 unseen.
  - 124 of the 300 are in the TRAIN split of the corrected evaluation pool, so the fitted
    fusion is refitted with the 300 removed, and every decision threshold is chosen on the
    pool's calibration split with the 300 removed. Nothing is fitted or tuned on these images.

Scene presence is scored threshold-free (ROC-AUC, bootstrap 95% CI) and at a threshold chosen
on calibration (balanced accuracy). SigLIP is not given object prompts in the deployed pipeline,
so it has no object-presence entry; Kosmos can only speak to the class "text".

Output: results/signal_rebuild/per_model_performance_300.json
"""
import argparse, glob, json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

warnings.filterwarnings("ignore")
BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
AUD = BASE / "human_spatial_audit"
OUT = SR / "per_model_performance_300.json"
WEIGHTS = BASE / "runs/detect/yolo11_v2_augmented_refresh/weights/best.pt"
AUDIT_NAMES = ["person", "child", "horse", "building", "weapon", "vehicle", "tree", "clothing", "text", "animal"]
SCENES = ["teaching", "family", "playing", "landscape", "drawing"]
WORD = re.compile(r"^[A-Za-zÄÖÜäöüß]{2,20}$")
RNG_SEED, B = 20260915, 2000


def norm(i):
    i = str(i); return i if "/" in i else i.replace("_", "/", 1)


def rj(paths, key="image_id"):
    rows = []
    for p in paths:
        for line in open(p):
            try: rows.append(json.loads(line))
            except Exception: pass
    d = pd.DataFrame(rows); d[key] = d[key].map(norm)
    return d.drop_duplicates(key).set_index(key)


def auc_ci(y, s, rng):
    y, s = np.asarray(y), np.asarray(s)
    point = float(roc_auc_score(y, s))
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i): bs.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return round(point, 4), [round(float(lo), 4), round(float(hi), 4)]


def bal_acc(pred, y):
    y, pred = np.asarray(y), np.asarray(pred)
    tpr = ((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum())
    tnr = ((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum())
    return float((tpr + tnr) / 2)


def prf(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else None; r = tp / (tp + fn) if tp + fn else None
    f = (2 * p * r / (p + r)) if p and r else 0.0
    return {"support": int(y.sum()), "predicted": int(pred.sum()), "tp": tp,
            "precision": None if p is None else round(p, 4), "recall": None if r is None else round(r, 4), "f1": round(f, 4)}


def validity(t):
    toks = re.findall(r"\S+", str(t))
    if not toks: return 0.0
    rep = 1 - len(set(toks)) / len(toks)
    wl = sum(bool(WORD.match(w.strip("#*>[]()|-.,:;"))) for w in toks) / len(toks)
    return max(0.0, (1 - rep) * wl)


def build_corpus():
    y = rj([SR / "yolo_full_corpus.jsonl"])
    s = rj([SR / "siglip_scene_full.jsonl"])
    v = rj(sorted(glob.glob(str(SR / "llava_full/shard_*.jsonl"))))
    k = rj(sorted(glob.glob(str(SR / "kosmos_full/shard_*.jsonl"))))
    z = np.load(SR / "negative_evidence_scores.npz", allow_pickle=True)
    neg = pd.DataFrame({"scene_ev": z["scene"], "neg_ev": z["neg"]}, index=[norm(i) for i in z["ids"]])
    st = pd.read_csv(BASE / "results/multi_agent/agent_comparison_scores.csv")
    st["image_id"] = st.image_id.map(norm); st = st.set_index("image_id")
    d = pd.DataFrame(index=y.index)
    d["yolo_conf_max"] = y.conf_max.astype(float)
    d["yolo_labels"] = y.labels.apply(lambda L: set(L) if isinstance(L, list) else set())
    sig = pd.DataFrame(list(s.sigmoid), index=s.index)[SCENES]
    d = d.join(pd.DataFrame({"siglip_softmax": s.max_softmax.astype(float), "siglip_sigmoid": s.max_sigmoid.astype(float),
                             "siglip_category": sig.idxmax(axis=1)}))
    txt = v.answers.apply(lambda a: " ".join(map(str, a)).lower() if isinstance(a, list) else "")
    d = d.join(pd.DataFrame({"llava_text": txt}))
    d["llava_named"] = d.llava_text.fillna("").apply(lambda t: {c for c in AUDIT_NAMES if re.search(rf"\b{c}", t)})
    d["llava_share_named"] = d.llava_named.map(len) / len(AUDIT_NAMES)
    d = d.join(pd.DataFrame({"kosmos_text": k.kosmos_output.fillna("")}))
    d["kosmos_amount"] = np.minimum(d.kosmos_text.fillna("").str.len() / 80.0, 1.0)
    d["kosmos_validity"] = d.kosmos_text.fillna("").map(validity)
    d["kosmos_wordlike_tokens"] = d.kosmos_text.fillna("").map(
        lambda t: sum(bool(WORD.match(w.strip("#*>[]()|-.,:;"))) for w in re.findall(r"\S+", t)))
    d = d.join(neg)
    d["negative_margin"] = d.scene_ev - d.neg_ev
    d = d.join(st[["existing_pipeline_agent", "scene_agent", "vlm_agent"]])
    d["fusion_deployed_stored"] = 0.6 * d.existing_pipeline_agent + 0.2 * d.scene_agent + 0.2 * d.vlm_agent
    d["fusion_rebuilt_handset"] = 0.6 * d.yolo_conf_max.clip(0, 1) + 0.2 * d.siglip_sigmoid + 0.2 * d.llava_share_named
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    rng = np.random.default_rng(RNG_SEED)
    d = build_corpus()
    ann = pd.read_csv(AUD / "user_annotations_300.csv"); ann["image_id"] = ann.image_id.map(norm)
    ann = ann.set_index("image_id")
    ids300 = list(ann.index)
    train_det = {norm(p.stem) for p in (BASE / "data/yolo11_v2_augmented/labels/train").glob("*.txt")}
    seen = np.array([i in train_det for i in ids300])

    # --- the corrected fusion, fitted with the 300 held out, and thresholds from calibration minus the 300
    pool = pd.read_csv(BASE / "results/multi_agent/corrected_clean_holdout_split.csv")
    pool["image_id"] = pool.ppn_id.map(norm); pool = pool[~pool.image_id.isin(ids300)]
    tr = pool[pool.split == "train"].join(d, on="image_id"); cal = pool[pool.split == "calibration"].join(d, on="image_id")
    cols = ["yolo_conf_max", "siglip_sigmoid", "neg_ev"]
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(tr[cols].values, tr.has_scene.values)
    w = lr.coef_[0] / np.abs(lr.coef_[0]).sum()
    d["fusion_corrected_fitted"] = d[cols].values @ w
    # rebuild the calibration view now that every score column exists
    cal = pool[pool.split == "calibration"].join(d, on="image_id")

    h = d.loc[ids300]
    y_scene = ann.scene.notna().astype(int).values

    SCORES = [("yolo_conf_max", "YOLO — highest box confidence"),
              ("siglip_softmax", "SigLIP — as deployed (softmax)"),
              ("siglip_sigmoid", "SigLIP — corrected (sigmoid)"),
              ("negative_margin", "SigLIP — with “no scene” prompts"),
              ("llava_share_named", "LLaVA — share of classes named"),
              ("kosmos_amount", "Kosmos — amount of text"),
              ("fusion_deployed_stored", "Fusion — as deployed"),
              ("fusion_rebuilt_handset", "Fusion — rebuilt, hand-set weights"),
              ("fusion_corrected_fitted", "Fusion — corrected, fitted without the 300")]
    scene_presence = []
    for col, label in SCORES:
        auc, ci = auc_ci(y_scene, h[col].values, rng)
        cs = cal[col].values; grid = np.unique(np.quantile(cs, np.linspace(0.01, 0.99, 197)))
        thr = float(grid[int(np.argmax([bal_acc((cs >= t).astype(int), cal.has_scene.values) for t in grid]))])
        pred = (h[col].values >= thr).astype(int)
        scene_presence.append({"score": col, "label": label, "roc_auc": auc, "ci95": ci,
                               "threshold_from_calibration": round(thr, 4),
                               "balanced_accuracy": round(bal_acc(pred, y_scene), 4),
                               "scene_recall": round(float(pred[y_scene == 1].mean()), 4),
                               "no_scene_recall": round(float(1 - pred[y_scene == 0].mean()), 4)})

    # --- paired comparisons: overlapping intervals do not settle which score is better
    def paired(c1, c2):
        s1, s2 = h[c1].values, h[c2].values; ds = []
        for _ in range(B):
            i = rng.integers(0, len(y_scene), len(y_scene))
            if 0 < y_scene[i].sum() < len(i):
                ds.append(roc_auc_score(y_scene[i], s1[i]) - roc_auc_score(y_scene[i], s2[i]))
        lo, hi = np.percentile(ds, [2.5, 97.5])
        return {"first": c1, "second": c2, "delta": round(float(roc_auc_score(y_scene, s1) - roc_auc_score(y_scene, s2)), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)], "excludes_zero": bool(lo > 0 or hi < 0)}
    scene_presence_paired = [paired("fusion_deployed_stored", "fusion_corrected_fitted"),
                             paired("fusion_deployed_stored", "negative_margin"),
                             paired("fusion_corrected_fitted", "negative_margin"),
                             paired("negative_margin", "siglip_sigmoid"),
                             paired("siglip_sigmoid", "siglip_softmax"),
                             paired("fusion_deployed_stored", "yolo_conf_max")]

    # --- scene category, SigLIP only (the one model asked about these five)
    has = ann.scene.notna().values
    truth = ann.scene[has].values; guess = h.siglip_category.values[has]
    cm = confusion_matrix(truth, guess, labels=SCENES)
    per = {s: round(float(cm[i, i] / cm[i].sum()), 4) if cm[i].sum() else None for i, s in enumerate(SCENES)}
    scene_category = {"model": "SigLIP (highest of the five prompts)", "n": int(has.sum()),
                      "accuracy": round(float((truth == guess).mean()), 4),
                      "balanced_accuracy": round(float(np.nanmean([x for x in per.values() if x is not None])), 4),
                      "recall_per_category": per, "support": {s: int((truth == s).sum()) for s in SCENES},
                      "confusion_rows_truth_cols_guess": {"labels": SCENES, "matrix": cm.tolist()},
                      "always_guess_most_common_accuracy": round(float((truth == pd.Series(truth).mode()[0]).mean()), 4)}

    # --- object presence per class
    truth_obj = {}
    for i in ids300:
        lf = AUD / "labels" / f"{i.replace('/', '_')}.txt"
        ids_ = {int(float(l.split()[0])) for l in lf.read_text().splitlines() if l.strip()} if lf.exists() else set()
        truth_obj[i] = {AUDIT_NAMES[c] for c in ids_}
    object_presence = []
    for c in AUDIT_NAMES:
        yt = np.array([c in truth_obj[i] for i in ids300]).astype(int)
        row = {"class": c, "pictures_with_class": int(yt.sum())}
        if yt.sum() == 0:
            row["note"] = "no human box of this class in the 300 -- not measurable"; object_presence.append(row); continue
        yp = np.array([c in h.yolo_labels.loc[i] for i in ids300]).astype(int)
        row["yolo_all"] = prf(yp, yt); row["yolo_seen"] = prf(yp[seen], yt[seen]); row["yolo_unseen"] = prf(yp[~seen], yt[~seen])
        lp = np.array([c in h.llava_named.loc[i] for i in ids300]).astype(int)
        row["llava"] = prf(lp, yt)
        if c == "text":
            kp = (h.kosmos_wordlike_tokens.values >= 3).astype(int)
            row["kosmos_three_or_more_words"] = prf(kp, yt)
        object_presence.append(row)

    # --- furniture and hat: on the rater's list but not the detector's. The Label Studio export mapped
    # only the detector's ten names, so these boxes never reached the label files; read them from the source.
    import sqlite3
    human_only = {"furniture": set(), "hat": set()}
    con = sqlite3.connect(Path.home() / ".local/share/label-studio/label_studio.sqlite3")
    for ds, rs in con.execute("SELECT t.data, tc.result FROM task t JOIN task_completion tc ON t.id = tc.task_id WHERE t.id <= 300"):
        res = json.loads(rs)
        for it in res if isinstance(res, list) else []:
            labs = it.get("value", {}).get("rectanglelabels") if it.get("type") == "rectanglelabels" else None
            if labs and labs[0].lower() in human_only:
                human_only[labs[0].lower()].add(norm(Path(json.loads(ds).get("image", "")).stem))
    for c, pics in human_only.items():
        assert pics <= set(ids300), f"{c}: pictures outside the 300"
        yt = np.array([i in pics for i in ids300]).astype(int)
        lp = h.llava_text.fillna("").str.contains(rf"\b{c}s?\b", case=False, regex=True).values.astype(int)
        object_presence.append({"class": c, "pictures_with_class": int(yt.sum()),
                                "detector": "cannot output this class: not in its ten names", "llava": prf(lp, yt)})

    out = {"experiment": "Per-model performance on the 300 human-annotated pictures (Prof. Mandl, experiment 1)",
           "n_pictures": len(ids300), "scene_yes": int(y_scene.sum()), "scene_no": int((1 - y_scene).sum()),
           "leakage_controls": {
               "pictures_in_detector_training_split": int(seen.sum()),
               "pictures_in_corrected_pool_removed_before_fitting": int(len(set(ids300) & set(pd.read_csv(BASE / "results/multi_agent/corrected_clean_holdout_split.csv").ppn_id.map(norm)))),
               "corrected_fusion_weights_object_scene_negative": [round(float(x), 4) for x in w]},
           "scene_presence": scene_presence, "scene_presence_paired": scene_presence_paired, "scene_category": scene_category, "object_presence": object_presence}

    if not a.verify:
        from ultralytics import YOLO
        import tempfile, os
        m = YOLO(str(WEIGHTS)); tmp = Path(tempfile.mkdtemp(dir="/tmp"))
        names = "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(AUDIT_NAMES))
        det = {}
        imgs = sorted((AUD / "images").glob("*.jpg"))
        subsets = {"all": imgs, "seen": [p for p in imgs if norm(p.stem) in train_det],
                   "unseen": [p for p in imgs if norm(p.stem) not in train_det]}
        for tag, lst in subsets.items():
            (tmp / f"{tag}.txt").write_text("\n".join(map(str, lst)) + "\n")
            (tmp / f"{tag}.yaml").write_text(f"path: /\ntrain: {tmp/tag}.txt\nval: {tmp/tag}.txt\n{names}")
            r = m.val(data=str(tmp / f"{tag}.yaml"), split="val", batch=16, device="cpu", plots=False, verbose=False,
                      project=str(tmp / "runs"), name=tag, exist_ok=True)
            det[tag] = {"pictures": len(lst), "map50": round(float(r.box.map50), 4), "map50_95": round(float(r.box.map), 4),
                        "per_class": {m.names[int(c)]: {"precision": round(float(r.box.class_result(i)[0]), 4),
                                                         "recall": round(float(r.box.class_result(i)[1]), 4),
                                                         "ap50": round(float(r.box.class_result(i)[2]), 4)}
                                      for i, c in enumerate(r.box.ap_class_index)}}
        out["object_boxes_detector"] = det
        OUT.write_text(json.dumps(out, indent=2))
    else:
        prev = json.loads(OUT.read_text())
        mine = {k: v for k, v in out.items()}; theirs = {k: v for k, v in prev.items() if k != "object_boxes_detector"}
        same = json.dumps(theirs, sort_keys=True) == json.dumps(mine, sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1

    print(f"pictures {len(ids300)}  scene yes {y_scene.sum()} / no {(1-y_scene).sum()}  | seen by detector {seen.sum()}")
    print("corrected fusion weights (object, scene, negative):", [round(float(x), 3) for x in w])
    print("\nSCENE PRESENCE")
    for r in scene_presence:
        print(f"  {r['label']:44s} AUROC {r['roc_auc']:.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]  bal.acc {r['balanced_accuracy']:.3f}  scene rec {r['scene_recall']:.2f}  no-scene rec {r['no_scene_recall']:.2f}")
    print("\nPAIRED (AUROC difference, 95% CI)")
    for q in scene_presence_paired:
        print(f"  {q['first']:26s} - {q['second']:26s} {q['delta']:+.3f} [{q['ci95'][0]:+.3f},{q['ci95'][1]:+.3f}]  {'excludes 0' if q['excludes_zero'] else 'spans 0'}")
    sc = scene_category
    print(f"\nSCENE CATEGORY (SigLIP, n={sc['n']}): accuracy {sc['accuracy']:.3f}  balanced {sc['balanced_accuracy']:.3f}  "
          f"(always 'drawing' would score {sc['always_guess_most_common_accuracy']:.3f})")
    print("  recall per category:", sc["recall_per_category"], " support:", sc["support"])
    print("\nOBJECT PRESENCE (F1)")
    for r in object_presence:
        if "note" in r: print(f"  {r['class']:9s} n=0   {r['note']}"); continue
        if "detector" in r: print(f"  {r['class']:9s} n={r['pictures_with_class']:<3d} detector cannot output it          LLaVA {r['llava']['f1']:.2f}"); continue
        extra = f"  kosmos {r['kosmos_three_or_more_words']['f1']:.2f}" if "kosmos_three_or_more_words" in r else ""
        print(f"  {r['class']:9s} n={r['pictures_with_class']:<3d} YOLO all {r['yolo_all']['f1']:.2f} (seen {r['yolo_seen']['f1']:.2f} / unseen {r['yolo_unseen']['f1']:.2f})  LLaVA {r['llava']['f1']:.2f}{extra}")
    print("\nOBJECT BOXES (detector mAP50):", {k: v["map50"] for k, v in out["object_boxes_detector"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
