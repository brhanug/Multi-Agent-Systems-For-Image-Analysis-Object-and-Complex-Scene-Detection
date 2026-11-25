#!/usr/bin/env python3
"""What the scene class "drawing" contains (self_annotation_package/more_experiments/PLAN_drawing_clusters.md).

Exploratory, added after the results freeze. The archive pictures outside the labelled pool that the
system calls "drawing" are clustered by their SigLIP image embeddings (k-means, k = 12); each cluster
is described by its most distinctive words in the describer's free-text answers, by the mean of the
system's object probabilities, and by the pictures nearest its centre. Writes
results/signal_rebuild/drawing_clusters.json and, with --sheet DIR, one contact sheet per cluster
(archive scans: never commit them).

    python run_drawing_clusters.py [--sheet DIR] [--verify]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # k-means sums in a fixed order: identical output on every run
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, ast, json, re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
OUT = SR / "drawing_clusters.json"
SEED, K, ALT = 20261121, 12, (8, 16)
OBJ = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]
# words that describe the medium or the prompt rather than what a picture shows
EXTRA_STOP = {"image", "picture", "illustration", "drawing", "depicts", "depicted", "shows", "shown", "appears",
              "present", "visible", "subject", "predicate", "object", "relational", "triplets", "interaction",
              "interactions", "primary", "found", "items", "list", "identify", "background", "scene", "style",
              "black", "white", "sketch", "engraving", "artwork", "art", "figure", "figures", "represents", "located"}


def describer_text(pid):
    f = BASE / "results_v1/llava" / (pid.replace("/", "_") + ".json")
    if not f.exists():
        return ""
    r = json.loads(f.read_text()).get("responses", "")
    if isinstance(r, str):
        try:
            r = ast.literal_eval(r)
        except Exception:
            return re.sub(r"[^A-Za-z ]", " ", r)
    keep = [v for k, v in r.items() if k.startswith(("Identify objects", "List relational"))]
    return re.sub(r"[^A-Za-z ]", " ", " ".join(map(str, keep)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sheet"); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    ans = pd.read_csv(SR / "archive_answers.csv")
    dr = ans[(~ans.in_pool) & (ans.scene_class == "drawing")].reset_index(drop=True)
    e = np.load(SR / "siglip_embeddings.npz", allow_pickle=True)
    pos = {i: k for k, i in enumerate(e["ids"])}
    X = e["vecs"][[pos[p] for p in dr.picture]]
    X = X.astype(np.float64); X = X / np.linalg.norm(X, axis=1, keepdims=True)
    km = KMeans(K, random_state=SEED, n_init=10).fit(X)
    lab = km.labels_
    sizes_alt = {str(k): sorted(np.bincount(KMeans(k, random_state=SEED, n_init=10).fit(X).labels_).tolist(), reverse=True)
                 for k in ALT}
    texts = [describer_text(p) for p in dr.picture]
    stop = list(ENGLISH_STOP_WORDS | EXTRA_STOP)
    tv = TfidfVectorizer(stop_words=stop, min_df=20, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", sublinear_tf=True)
    T = tv.fit_transform(texts); vocab = np.array(tv.get_feature_names_out())
    clusters = []
    for c in np.argsort(-np.bincount(lab)):
        m = lab == c
        diff = np.asarray(T[m].mean(0)).ravel() - np.asarray(T[~m].mean(0)).ravel()
        d = X[m] @ km.cluster_centers_[c] / np.linalg.norm(km.cluster_centers_[c])
        near = dr.picture[m].values[np.argsort(-d)[:8]].tolist()
        clusters.append({"cluster": int(c), "pictures": int(m.sum()), "share": round(float(m.mean()), 4),
                         "distinctive_words": vocab[np.argsort(-diff)[:12]].tolist(),
                         "mean_object_probability": {o: round(float(dr[f"p_{o}"][m].mean()), 3) for o in OBJ},
                         "mean_class_probability": round(float(dr.scene_class_p[m].mean()), 3),
                         "nearest_centre": near})
    out = {"extra": "exploratory, added after the results freeze (PLAN_drawing_clusters.md)",
           "pictures": int(len(dr)), "share_of_outside_pool": round(len(dr) / int((~ans.in_pool).sum()), 4),
           "k": K, "seed": SEED, "sizes_other_k": sizes_alt, "clusters": clusters,
           "groups_8_to_10": {"pictures": int(sum(c["pictures"] for c in clusters[7:10])), "share": round(float(sum(c["pictures"] for c in clusters[7:10]) / len(dr)), 4)}}
    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    for c in clusters:
        print(f"{c['pictures']:5d} {c['share']:.3f} {' '.join(c['distinctive_words'])}")
    if a.sheet:
        from PIL import Image, ImageDraw
        sd = Path(a.sheet); sd.mkdir(parents=True, exist_ok=True)
        for r, c in enumerate(clusters):
            sh = Image.new("RGB", (8 * 230, 250), "white"); dw = ImageDraw.Draw(sh)
            for k, p in enumerate(c["nearest_centre"]):
                im = Image.open(BASE / "final_dataset/images/original" / f"{p}.jpg").convert("RGB"); im.thumbnail((220, 220))
                sh.paste(im, (k * 230 + 5, 5))
            dw.text((5, 232), f"#{r + 1}  n={c['pictures']}  {' '.join(c['distinctive_words'][:8])}", fill="black")
            sh.save(sd / f"cluster_{r + 1:02d}.jpg", quality=80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
