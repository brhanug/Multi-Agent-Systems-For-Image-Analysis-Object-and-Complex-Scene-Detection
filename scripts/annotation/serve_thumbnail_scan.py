#!/usr/bin/env python3
"""The thumbnail scan: Prof. Mandl's "thumbnail mode and scroll", as a labelling tool.

A grid of small pictures, one target at a time (a rare object class or a rare scene type). You
click every picture that shows the target and go to the next screen; every picture shown is
recorded, hit or not, with the seconds the screen took, so the yield per minute can be compared
with the other mining tactics of Task G.

Two orders per target:
  ranked  highest first by the mean of SigLIP's and LLaVA's within-archive percentiles for the
          target -- the rare pictures come first, and you stop once they run out (the page
          suggests stopping after two screens in a row without a hit)
  random  the same pictures in random order -- slower, but it shows what the ranked order misses,
          so it is the honest check on the ranked one; do a few screens of it per target

Left out: pictures already labelled or in another labelling task (A, A2, E, F, H, Task G v3),
and pictures SigLIP reads as an ornament, blank or text page. Labels go to
self_annotation_package/thumbnail_scan/scan_labels.csv; a reload continues where you stopped.

    python serve_thumbnail_scan.py [--port 8090]
    then on your own machine:  ssh -L 8090:localhost:8090 <this server>  and open http://localhost:8090
"""
import argparse, csv, glob, io, json, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

import numpy as np
import pandas as pd
from PIL import Image

BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
PKG = BASE / "self_annotation_package"
IMG = BASE / "final_dataset/images/restored"
HERE = PKG / "thumbnail_scan"
CACHE = HERE / "cache"
LABELS = HERE / "scan_labels.csv"
PAGE = Path(__file__).with_name("thumbnail_scan.html")
OBJECTS = ["vehicle", "building", "weapon", "animal", "child", "tree"]
SCENES = ["teaching", "family", "landscape", "playing"]
FIELDS = ["image_id", "target", "order", "position", "screen", "hit", "screen_seconds", "recorded_at"]
MAX_NEG, SEED, THUMB = 0.2, 20261006, 200
lock = threading.Lock()


def norm(i):
    i = str(i)
    return i if "/" in i else i.replace("_", "/", 1)


def pct(x):
    return pd.Series(x).rank(pct=True).values


def build_orders():
    m = np.load(SR / "class_scores/class_score_matrix.npz", allow_pickle=True)
    ids = np.array([str(i) for i in m["ids"]])
    mc, ms = [str(x) for x in m["classes"]], [str(x) for x in m["systems"]]
    rank = {c: (m["percentile"][:, ms.index("siglip"), mc.index(c)] + m["percentile"][:, ms.index("llava"), mc.index(c)]) / 2
            for c in OBJECTS}
    sig, lla = {}, {}
    for line in open(SR / "siglip_scene_full.jsonl"):
        r = json.loads(line); sig[norm(r["image_id"])] = r["sigmoid"]
    for f in sorted(glob.glob(str(SR / "class_scores/llava_scene_scores/shard_*.jsonl"))):
        for line in open(f):
            r = json.loads(line); lla[norm(r["image_id"])] = r["p_yes"]
    for c in SCENES:
        rank[c] = (pct([sig.get(i, {}).get(c, 0.0) for i in ids]) + pct([lla.get(i, {}).get(c, 0.0) for i in ids])) / 2

    taken = set(pd.read_csv(SR / "class_scores/human_class_truth_300.csv").image_id)
    taken |= set(pd.read_csv(BASE / "results/multi_agent/expanded/expanded_clean_pool.csv").ppn_id)
    for f in glob.glob(str(PKG / "task_*/*.csv")):
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        for col in ("image_id", "ppn_id"):
            if col in df:
                taken |= {norm(x) for x in df[col] if x.startswith("PPN")}
    z = np.load(SR / "negative_evidence_scores.npz", allow_pickle=True)
    neg_of = dict(zip((str(i) for i in z["ids"]), z["neg"]))
    neg = np.array([neg_of.get(i, neg_of.get(i.replace("/", "_", 1), 0.0)) for i in ids])
    ok = np.array([i not in taken for i in ids]) & (neg < MAX_NEG)
    pool = ids[ok]
    shuffled = list(pool[np.random.default_rng(SEED).permutation(len(pool))])
    orders = {}
    for c, r in rank.items():
        orders[(c, "ranked")] = list(ids[np.argsort(-r, kind="stable")][ok[np.argsort(-r, kind="stable")]])
        orders[(c, "random")] = shuffled
    return orders, len(ids), int((~ok).sum())


def done_rows():
    if not LABELS.exists():
        return []
    with open(LABELS) as f:
        return list(csv.DictReader(f))


def thumb(iid, size):
    p = CACHE / f"{size}" / (iid.replace("/", "_") + ".jpg")
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(IMG / f"{iid}.jpg").convert("RGB"); im.thumbnail((size, size)); im.save(p, quality=82)
    return p.read_bytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype, code=200):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "max-age=86400")
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/":
            return self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
        if u.path.startswith("/img/"):
            iid = unquote(u.path[5:]); size = THUMB if q.get("s", ["t"])[0] == "t" else 900
            if ".." in iid or not (IMG / f"{iid}.jpg").exists():
                return self.send(b"not found", "text/plain", 404)
            return self.send(thumb(iid, size), "image/jpeg")
        if u.path == "/api/targets":
            rows = done_rows(); out = []
            for (c, o), order in ORDERS.items():
                mine = [r for r in rows if r["target"] == c and r["order"] == o]
                out.append({"target": c, "kind": "scene" if c in SCENES else "object", "order": o,
                            "shown": len(mine), "hits": sum(r["hit"] == "1" for r in mine),
                            "screens": len({r["screen"] for r in mine}),
                            "minutes": round(sum({r["screen"]: float(r["screen_seconds"]) for r in mine}.values()) / 60, 1),
                            "pool": len(order)})
            return self.send(json.dumps({"targets": out, "excluded": EXCLUDED, "archive": ARCHIVE}).encode(), "application/json")
        if u.path == "/api/screen":
            c, o, n = q["target"][0], q["order"][0], int(q.get("n", ["60"])[0])
            rows = [r for r in done_rows() if r["target"] == c and r["order"] == o]
            seen = {r["image_id"] for r in rows}
            order = ORDERS[(c, o)]
            nxt = [(k + 1, i) for k, i in enumerate(order) if i not in seen][:n]
            screens = sorted({int(r["screen"]) for r in rows})
            dry = 0
            for s in reversed(screens):
                if any(r["hit"] == "1" for r in rows if int(r["screen"]) == s): break
                dry += 1
            return self.send(json.dumps({"screen": (screens[-1] + 1) if screens else 1, "dry_screens": dry,
                                         "items": [{"id": i, "position": p} for p, i in nxt]}).encode(), "application/json")
        self.send(b"not found", "text/plain", 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/save":
            return self.send(b"not found", "text/plain", 404)
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if (d["target"], d["order"]) not in ORDERS:
            return self.send(b'{"error":"unknown target"}', "application/json", 400)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with lock:
            new = not LABELS.exists()
            seen = {(r["image_id"], r["target"], r["order"]) for r in done_rows()}
            with open(LABELS, "a", newline="") as f:
                w = csv.DictWriter(f, FIELDS)
                if new: w.writeheader()
                for it in d["items"]:
                    if (it["id"], d["target"], d["order"]) in seen: continue
                    w.writerow({"image_id": it["id"], "target": d["target"], "order": d["order"], "position": it["position"],
                                "screen": d["screen"], "hit": 1 if it["hit"] else 0,
                                "screen_seconds": round(float(d["seconds"]), 1), "recorded_at": now})
        self.send(b'{"saved":true}', "application/json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8090); a = ap.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    ORDERS, ARCHIVE, EXCLUDED = build_orders()
    print(f"{len(next(iter(ORDERS.values())))}+ pictures per target; serving http://localhost:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
