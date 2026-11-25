#!/usr/bin/env python3
"""Rare-class mining sheets (Phase 1, A2; also the single-content-book tactic of A1).

Several ways of finding rare content, each turned into contact sheets you label by hand,
so their yield per minute can be compared (the tactics of the current version are listed below).

Negative controls are deliberately not mined here: hand-picked negatives cannot enter a
test split (decision D4); they keep coming from the randomly ordered Task A2 sheet.

Pictures already in the 300-picture audit or the negative-control pool are left out.
Every sheet lists its pictures in labeling_sheet.csv; record the classes you see and the
minutes each sheet took. Sheets are images of archive scans: keep them out of git.

Repeats are removed (v2, 1 Oct). Every picture appears once; a picture whose SigLIP embedding is
within cosine NEAR_DUP of one already chosen is skipped; pictures SigLIP reads as an ornament, blank
or text page (negative evidence >= MAX_NEG) are skipped; a cluster or prompt gives at most PER_BOOK
pictures from one book; gaps are refilled from the next candidates.

Targets corrected (v3, 6 Oct). A check of v2 against the describer's estimates showed that the
smallest clusters held *fewer* rare objects than the archive at large (building 12.8% against 19.6%,
weapon 11.0% against 17.1%), that the single-content books mostly failed, that half the prompt picks
went to horse, furniture and hat -- none of them among the eight frozen classes -- and that no rare
scene was searched at all. So now:

  prompt   the main tactic: for each rare frozen object class (vehicle, building, weapon) and each
           rare scene type (teaching, family, landscape), the pictures ranked highest by the mean of
           SigLIP's and LLaVA's within-archive percentiles for that class -- two systems, so one
           system's quirk does not fill the sheet; PROMPT_TOP pictures each
  cluster  a small sample from the smallest k-means clusters, kept only to compare yields
  book     a small sample from the books that look single-content for a rare object class
  random   pictures drawn at random from everything usable: the baseline every tactic is compared
           against, and the slice that shows what the ranked tactics miss

Pictures already in any other labelling task (A, A2, E, F, H), and the three labelled in v2, are left out
as well.

    python build_rare_class_mining.py
"""
import glob, json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
EMB = SR / "siglip_embeddings.npz"
MATRIX = SR / "class_scores/class_score_matrix.npz"
IMG = BASE / "final_dataset/images/restored"
OUT = BASE / "self_annotation_package/task_g_rare_class_mining"
PKG = BASE / "self_annotation_package"
K, SEED, PER_SHEET, THUMB = 200, 20260916, 16, 220
RARE_OBJECTS = ["vehicle", "building", "weapon"]
RARE_SCENES = ["teaching", "family", "landscape"]
SCENES = ["teaching", "family", "playing", "landscape", "drawing"]
PROMPT_TOP, CLUSTER_N, BOOK_N, RANDOM_N, BOOK_MIN = 40, 24, 24, 24, 8
NEG = SR / "negative_evidence_scores.npz"
NEAR_DUP, MAX_NEG, PER_BOOK = 0.90, 0.2, 3


def norm(i):
    i = str(i)
    return i if "/" in i else i.replace("_", "/", 1)


def sheet(ids, path, title):
    cols = 4; rows = (len(ids) + cols - 1) // cols
    im = Image.new("RGB", (cols * (THUMB + 10) + 10, rows * (THUMB + 28) + 40), "white"); d = ImageDraw.Draw(im)
    d.text((10, 10), title, fill="black")
    for k, i in enumerate(ids):
        try:
            t = Image.open(IMG / f"{i}.jpg").convert("RGB"); t.thumbnail((THUMB, THUMB))
        except Exception:
            continue
        x, y = 10 + (k % cols) * (THUMB + 10), 36 + (k // cols) * (THUMB + 28)
        im.paste(t, (x, y)); d.text((x, y + THUMB + 4), f"{k + 1}", fill="black")
    im.save(path)


def other_tasks():
    """Every picture already in another labelling task."""
    out = set()
    for f in (glob.glob(str(PKG / "task_[a-f]*/*.csv")) + glob.glob(str(PKG / "task_h*/*.csv"))
              + [str(OUT / "labels_v2_labelstudio.csv")]):      # the three pictures labelled in v2
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        for col in ("image_id", "ppn_id"):
            if col in df:
                out |= {norm(x) for x in df[col] if x.startswith("PPN")}
    return out


def pct(x):
    return pd.Series(x).rank(pct=True).values


def main():
    from sklearn.cluster import KMeans
    e = np.load(EMB, allow_pickle=True); ids = np.array([str(i) for i in e["ids"]]); V = e["vecs"]
    m = np.load(MATRIX, allow_pickle=True); assert (m["ids"] == ids).all()
    mc, ms = [str(x) for x in m["classes"]], [str(x) for x in m["systems"]]
    rank = {c: (m["percentile"][:, ms.index("siglip"), mc.index(c)] + m["percentile"][:, ms.index("llava"), mc.index(c)]) / 2
            for c in RARE_OBJECTS}
    sig = {}
    for line in open(SR / "siglip_scene_full.jsonl"):
        r = json.loads(line); sig[norm(r["image_id"])] = r["sigmoid"]
    lla = {}
    for f in sorted(glob.glob(str(SR / "class_scores/llava_scene_scores/shard_*.jsonl"))):
        for line in open(f):
            r = json.loads(line); lla[norm(r["image_id"])] = r["p_yes"]
    for c in RARE_SCENES:
        rank[c] = (pct([sig.get(i, {}).get(c, 0.0) for i in ids]) + pct([lla.get(i, {}).get(c, 0.0) for i in ids])) / 2

    known = set(pd.read_csv(SR / "class_scores/human_class_truth_300.csv").image_id)
    known |= set(pd.read_csv(BASE / "results/multi_agent/expanded/expanded_clean_pool.csv").ppn_id)
    in_tasks = other_tasks()
    fresh = np.array([i not in known and i not in in_tasks for i in ids])
    z = np.load(NEG, allow_pickle=True)
    neg_of = dict(zip((str(i) for i in z["ids"]), z["neg"]))
    neg = np.array([neg_of.get(i, neg_of.get(i.replace("/", "_", 1), 0.0)) for i in ids])
    usable = fresh & (neg < MAX_NEG)
    U = V / np.linalg.norm(V, axis=1, keepdims=True)
    pos = {i: k for k, i in enumerate(ids)}
    chosen = []                                   # row indices already on some sheet

    def take(candidates, n, per_book=None):
        """The first n candidates that are usable, not yet shown, and not a near-copy of one shown."""
        out, books = [], {}
        for i in candidates:
            k = pos[i]
            if not usable[k] or k in chosen:
                continue
            b = i.split("/")[0]
            if per_book and books.get(b, 0) >= per_book:
                continue
            if chosen and float((U[chosen] @ U[k]).max()) >= NEAR_DUP:
                continue
            chosen.append(k); out.append(i); books[b] = books.get(b, 0) + 1
            if len(out) == n:
                break
        return out

    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    for old in (OUT / "sheets").glob("*.png"):
        old.unlink()
    rows, summary = [], {"version": 3, "k": K, "seed": SEED, "excluded_already_labelled": int(sum(i in known for i in ids)),
                         "excluded_in_other_tasks": int(sum(i in in_tasks and i not in known for i in ids)),
                         "excluded_ornament_or_page": int((fresh & ~usable).sum()),
                         "near_duplicate_cosine": NEAR_DUP, "targets": RARE_OBJECTS + RARE_SCENES, "sheets": []}

    def add(source, key, picks, title):
        name = f"{source}_{key}"
        for j in range(0, len(picks), PER_SHEET):
            part = picks[j:j + PER_SHEET]; sid = f"{name}_{j // PER_SHEET + 1}"
            sheet(part, OUT / "sheets" / f"{sid}.png", f"{sid}  ({source}: {title})")
            rows.extend({"sheet": sid, "position": p + 1, "image_id": i, "source": source, "key": key,
                         "classes_you_see": "", "scene_you_see": "", "minutes_for_this_sheet": ""} for p, i in enumerate(part))
            summary["sheets"].append({"sheet": sid, "source": source, "key": key, "pictures": len(part)})

    for c in RARE_OBJECTS + RARE_SCENES:
        add("prompt", c, take(list(ids[np.argsort(-rank[c], kind="stable")]), PROMPT_TOP, per_book=PER_BOOK),
            f"top pictures for '{c}' (SigLIP and LLaVA)")
    km = KMeans(n_clusters=K, n_init=4, random_state=SEED).fit(V)
    sizes = np.bincount(km.labels_, minlength=K)
    picks = []
    for c in np.argsort(sizes, kind="stable"):
        picks += take([ids[i] for i in np.where(km.labels_ == c)[0]], CLUSTER_N - len(picks), per_book=PER_BOOK)
        if len(picks) >= CLUSTER_N:
            break
    add("cluster", "smallest", picks, "smallest k-means clusters")
    top = np.array(mc)[m["percentile"][:, ms.index("siglip"), :].argmax(1)]
    books = pd.DataFrame({"image_id": ids, "book": [i.split("/")[0] for i in ids], "top": top})
    picks = []
    for c in RARE_OBJECTS:
        g = books.groupby("book").agg(n=("top", "size"), share=("top", lambda t: float((t == c).mean())))
        g = g[g.n >= BOOK_MIN].sort_values("share", ascending=False)
        for b in g.index:
            got = take(list(books.image_id[books.book == b]), BOOK_N // len(RARE_OBJECTS))
            if got:
                picks += got; break
    add("book", "single_content", picks, "books that look single-content for vehicle, building or weapon")
    order = np.random.default_rng(SEED).permutation(len(ids))
    add("random", "baseline", take(list(ids[order]), RANDOM_N), "random pictures (baseline)")

    pd.DataFrame(rows).to_csv(OUT / "labeling_sheet.csv", index=False)
    (OUT / "sheets.json").write_text(json.dumps(summary, indent=2))
    (OUT / "README.md").write_text(
        "# Task G - rare-class mining (v3)\n\nFor every picture, write the object classes you see (person, child, animal, "
        "building, weapon, vehicle, tree, text) in `classes_you_see`, and the scene type (teaching, family, playing, "
        "landscape, drawing, or none) in `scene_you_see`. Note the minutes each sheet took once, on its first row. "
        "Easiest in the thumbnail page (`thumbnail_page/`), which saves into the same columns.\n\n"
        "Sources: `prompt` = the top pictures for one rare class or scene by SigLIP and LLaVA together; `cluster` = "
        "smallest k-means clusters; `book` = books that look single-content; `random` = the baseline. Comparing the "
        "yield of rare content per minute across the four is the result.\n")
    by = pd.DataFrame(summary["sheets"]).groupby("source").pictures.agg(["count", "sum"])
    print(json.dumps({k: v for k, v in summary.items() if k != "sheets"}, indent=1))
    print(by.rename(columns={"count": "sheets", "sum": "pictures"})); print("wrote", OUT)


if __name__ == "__main__":
    main()
