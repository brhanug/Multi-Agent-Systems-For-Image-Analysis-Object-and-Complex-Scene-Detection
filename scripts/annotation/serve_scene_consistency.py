#!/usr/bin/env python3
"""Scene consistency check: one rule for "scene" across the whole labelled pool.

In the 1,250-picture pool the expert CVAT labels count almost every picture as a scene (796 of 801),
sheet music, book covers and borders included, while the 2026 relabelling rounds label the same kinds
of picture "no scene". This page shows the expert "scene" pictures that may break the rule (324,
picked by a pre-sort with SigLIP and Claude, not shown here) mixed with 40 randomly chosen pictures the
pre-sort kept as scenes, in a fixed random order. You label each one yourself; the pre-sort is never
shown, so the labels are made blind to it.

The rule, as applied in the 2026 relabelling rounds:
  scene     people, animals, a landscape, a building or a room is shown
  no scene  sheet music, text, a cover, marbled or blank paper, a border, an ornament, an initial
            without figures, a diagram, map or table, an emblem, coat of arms or stamp
  unsure    anything else (a single object, a plant study); say which rule you used in the note

Labels go to self_annotation_package/scene_consistency/labels.csv (last click per picture counts);
a reload continues where you stopped.

    python serve_scene_consistency.py [--port 8091] [--list verify_60.csv] [--dir final_check --list items.csv]

--dir final_check serves the final check: 400 pictures drawn at random from the archive outside the
pool, shuffled with the 304 scene-consistency pictures not in verify_60.csv; labels go to
final_check/labels.csv (PLAN.md there fixes how they are used).

--list picks the pictures shown (default candidates.csv). verify_60.csv is a random 60 of the 364,
used to check the labels Claude made under the same rule (claude_labels.csv) against a person's.
    then on your own machine:  ssh -L 8091:localhost:8091 <this server>  and open http://localhost:8091
"""
import argparse, csv, io, json, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd
from PIL import Image

BASE = Path("/data/brhanu/thesis_project")
HERE = BASE / "self_annotation_package/scene_consistency"
IMG = BASE / "final_dataset/images/original"
CACHE = HERE / "cache"
LABELS = HERE / "labels.csv"
FIELDS = ["ppn_id", "label", "recorded_at"]
THUMB, BIG = 260, 900
lock = threading.Lock()

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Scene check</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f5f1;--ink:#22241f;--muted:#6b6e64;--card:#fff;--line:#d9d7cf;--scene:#2f6f4f;--none:#8a3b2e;--unsure:#8a6d1f}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;z-index:2}
h1{font-size:18px;margin:0 0 4px} .rule{color:var(--muted);font-size:13px;max-width:90ch}
.bar{display:flex;gap:12px;align-items:center;margin-top:6px;flex-wrap:wrap}
button{font:inherit;border:1px solid var(--line);background:var(--card);border-radius:6px;padding:4px 10px;cursor:pointer}
button:focus-visible{outline:2px solid var(--ink)}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px;padding:16px}
.t{background:var(--card);border:2px solid var(--line);border-radius:8px;padding:6px}
.t img{width:100%;height:240px;object-fit:contain;cursor:zoom-in;background:#eee}
.t .b{display:flex;gap:6px;margin-top:6px}.t .b button{flex:1}
.t[data-l=scene]{border-color:var(--scene)}.t[data-l=none]{border-color:var(--none)}.t[data-l=unsure]{border-color:var(--unsure)}
.t[data-l=scene] .s,.t[data-l=none] .n,.t[data-l=unsure] .u{color:#fff}
.t[data-l=scene] .s{background:var(--scene)}.t[data-l=none] .n{background:var(--none)}.t[data-l=unsure] .u{background:var(--unsure)}
#z{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:5}
#z img{max-width:95vw;max-height:95vh}
</style></head><body>
<header><h1>Scene check: <span id="c"></span></h1>
<div class="rule"><b>Scene</b>: people, animals, a landscape or a room is shown. <b>No scene</b>: sheet music, text, a cover, blank or marbled paper, a border, an ornament, an initial without figures, a diagram, map or table, an emblem, coat of arms or stamp (even with a figure), a single object, a plant study. <b>Unsure</b> only if you cannot tell. Click a picture to enlarge it.</div>
<div class="bar"><button id="p">Previous screen</button><span id="pg"></span><button id="nx">Next screen</button><span class="rule">Labels save on every click.</span></div></header>
<main id="g"></main><div id="z"><img alt=""></div>
<script>
let items=[],lab={},page=0;const N=30;
const g=document.getElementById('g'),z=document.getElementById('z');
async function load(){const r=await fetch('/api/state');const d=await r.json();items=d.items;lab=d.labels;
 const first=items.findIndex(i=>!lab[i]);page=first<0?0:Math.floor(first/N);draw()}
function draw(){g.innerHTML='';const sl=items.slice(page*N,page*N+N);
 sl.forEach(id=>{const t=document.createElement('div');t.className='t';t.dataset.l=lab[id]||'';
  t.innerHTML=`<img loading="lazy" alt="picture" src="/img?id=${encodeURIComponent(id)}&s=t"><div class="b"><button class="s">Scene</button><button class="n">No scene</button><button class="u">Unsure</button></div>`;
  t.querySelector('img').onclick=()=>{z.querySelector('img').src='/img?id='+encodeURIComponent(id)+'&s=b';z.style.display='flex'};
  for(const [c,v] of [['s','scene'],['n','none'],['u','unsure']]) t.querySelector('.'+c).onclick=async()=>{
   await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ppn_id:id,label:v})});
   lab[id]=v;t.dataset.l=v;count()};
  g.appendChild(t)});
 document.getElementById('pg').textContent=`Screen ${page+1} of ${Math.ceil(items.length/N)}`;count();scrollTo(0,0)}
function count(){document.getElementById('c').textContent=`${Object.keys(lab).filter(k=>items.includes(k)).length} of ${items.length} labelled`}
document.getElementById('p').onclick=()=>{if(page>0){page--;draw()}};
document.getElementById('nx').onclick=()=>{if((page+1)*N<items.length){page++;draw()}};
z.onclick=()=>z.style.display='none';load();
</script></body></html>"""


LIST = "candidates.csv"


def items():
    return pd.read_csv(HERE / LIST).ppn_id.tolist()


def labels():
    if not LABELS.exists():
        return {}
    return {r["ppn_id"]: r["label"] for r in csv.DictReader(LABELS.open())}


def thumb(iid, size):
    p = CACHE / str(size) / (iid.replace("/", "_") + ".jpg")
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        src = IMG / f"{iid}.jpg"
        if not src.exists():
            src = BASE / "final_dataset/images/restored" / f"{iid}.jpg"
        im = Image.open(src).convert("RGB"); im.thumbnail((size, size)); im.save(p, quality=85)
    return p.read_bytes()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype, code=200):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "max-age=86400")
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/":
            return self.send(PAGE.encode(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            return self.send(json.dumps({"items": items(), "labels": labels()}).encode(), "application/json")
        if u.path == "/img":
            iid = q.get("id", [""])[0]
            if iid not in set(items()):
                return self.send(b"unknown picture", "text/plain", 404)
            return self.send(thumb(iid, BIG if q.get("s", ["t"])[0] == "b" else THUMB), "image/jpeg")
        self.send(b"not found", "text/plain", 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/label":
            return self.send(b"not found", "text/plain", 404)
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if d.get("ppn_id") not in set(items()) or d.get("label") not in ("scene", "none", "unsure"):
            return self.send(b"bad label", "text/plain", 400)
        with lock:
            new = not LABELS.exists()
            with LABELS.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow({"ppn_id": d["ppn_id"], "label": d["label"], "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self.send(b'{"ok":true}', "application/json")


def main():
    global LIST, HERE, CACHE, LABELS
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--list", default="candidates.csv")
    ap.add_argument("--dir", default=None, help="folder under self_annotation_package holding the list; labels.csv goes there")
    a = ap.parse_args(); LIST = a.list
    if a.dir:
        HERE = BASE / "self_annotation_package" / a.dir; CACHE = HERE / "cache"; LABELS = HERE / "labels.csv"
    print(f"Scene consistency page on http://localhost:{a.port} ({len(items())} pictures, {len(labels())} labelled)")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
