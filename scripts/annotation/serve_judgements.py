#!/usr/bin/env python3
"""Two small judgement tasks, planned in self_annotation_package/more_experiments/:

  --task retrieval  PLAN_retrieval_human.md: for each query and picture, is the picture relevant?
                    relevant / not relevant / unsure. The method that found it is never shown.
  --task links      PLAN_agent4_archive_links.md: two pictures from different books, side by side:
                    same picture / same style only / unrelated / unsure. The similarity is never shown.

Labels go to self_annotation_package/<task folder>/labels.csv (last click per item counts); a reload
continues where you stopped.

    python serve_judgements.py --task retrieval --port 8096
    python serve_judgements.py --task links --port 8097
    then on your own machine:  ssh -L 8096:localhost:8096 <this server>  and open http://localhost:8096
"""
import argparse, csv, json, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd
from PIL import Image

BASE = Path("/data/brhanu/thesis_project")
IMG = BASE / "final_dataset/images/original"
TASKS = {
    "retrieval": {"dir": "retrieval_judgements", "title": "Retrieval: is the picture relevant to the query?",
                  "options": [["relevant", "Relevant"], ["not_relevant", "Not relevant"], ["unsure", "Unsure"]],
                  "help": "Judge only whether the picture shows what the query asks for. Click a picture to enlarge it."},
    "links": {"dir": "agent4_links", "title": "Reprint links: are these the same picture?",
              "options": [["same_picture", "Same picture"], ["same_style_only", "Same style only"],
                          ["unrelated", "Unrelated"], ["unsure", "Unsure"]],
              "help": "Same picture: the same illustration (a reprint, or the same scan in another book), even if cropped or coloured differently. "
                      "Same style only: similar kind of picture, but a different composition. Click a picture to enlarge it."},
}
THUMB, BIG = 320, 1000
lock = threading.Lock()
CFG = HERE = None

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Judgements</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f5f1;--ink:#22241f;--muted:#6b6e64;--card:#fff;--line:#d9d7cf;--on:#2f6f4f}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;z-index:2}
h1{font-size:18px;margin:0 0 4px}.rule{color:var(--muted);font-size:13px;max-width:100ch}
.bar{display:flex;gap:12px;align-items:center;margin-top:6px;flex-wrap:wrap}
button{font:inherit;border:1px solid var(--line);background:var(--card);border-radius:6px;padding:4px 10px;cursor:pointer}
button:focus-visible{outline:2px solid var(--ink)}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--w),1fr));gap:12px;padding:16px}
.t{background:var(--card);border:2px solid var(--line);border-radius:8px;padding:8px}
.t.done{border-color:var(--on)}
.q{font-weight:600;font-size:16px;margin-bottom:6px}
.pics{display:flex;gap:6px}.pics img{flex:1;min-width:0;height:260px;object-fit:contain;background:#eee;cursor:zoom-in}
.b{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}.b button{flex:1 1 22%;font-size:13px;padding:4px}
.b button.on{background:var(--on);color:#fff}
#z{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:5}
#z img{max-width:95vw;max-height:95vh}
</style></head><body>
<header><h1 id="h"></h1><div class="rule" id="help"></div>
<div class="bar"><button id="p">Previous screen</button><span id="pg"></span><button id="nx">Next screen</button><span id="c" class="rule"></span></div></header>
<main id="g"></main><div id="z"><img alt=""></div>
<script>
let S,lab={},page=0;const N=24;const g=document.getElementById('g'),z=document.getElementById('z');
async function load(){S=await (await fetch('/api/state')).json();lab=S.labels;
 document.getElementById('h').textContent=S.title;document.getElementById('help').textContent=S.help;
 g.style.setProperty('--w',S.pair?'560px':'300px');
 const first=S.items.findIndex(i=>!lab[i.item_id]);page=first<0?0:Math.floor(first/N);draw()}
function img(id){return `<img loading="lazy" alt="picture" src="/img?id=${encodeURIComponent(id)}&s=t" data-id="${id}">`}
function draw(){g.innerHTML='';S.items.slice(page*N,page*N+N).forEach(it=>{const t=document.createElement('div');t.className='t'+(lab[it.item_id]?' done':'');
 t.innerHTML=(it.query?`<div class="q">“${it.query}”</div>`:'')+`<div class="pics">${it.pics.map(img).join('')}</div><div class="b">${S.options.map(([v,l])=>`<button data-v="${v}" class="${lab[it.item_id]===v?'on':''}">${l}</button>`).join('')}</div>`;
 t.querySelectorAll('img').forEach(im=>im.onclick=()=>{z.querySelector('img').src='/img?id='+encodeURIComponent(im.dataset.id)+'&s=b';z.style.display='flex'});
 t.querySelectorAll('.b button').forEach(b=>b.onclick=async()=>{const v=b.dataset.v;
  const r=await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({item_id:it.item_id,label:v})});
  if(!r.ok){alert('Not saved: '+await r.text());return}
  lab[it.item_id]=v;t.classList.add('done');t.querySelectorAll('.b button').forEach(x=>x.classList.toggle('on',x.dataset.v===v));count()});
 g.appendChild(t)});
 document.getElementById('pg').textContent=`Screen ${page+1} of ${Math.ceil(S.items.length/N)}`;count();scrollTo(0,0)}
function count(){document.getElementById('c').textContent=`${S.items.filter(i=>lab[i.item_id]).length} of ${S.items.length} judged · saved on every click`}
document.getElementById('p').onclick=()=>{if(page>0){page--;draw()}};
document.getElementById('nx').onclick=()=>{if((page+1)*N<S.items.length){page++;draw()}};
z.onclick=()=>z.style.display='none';load();
</script></body></html>"""


def items():
    d = pd.read_csv(HERE / "items.csv")
    if "query_text" in d:
        return [{"item_id": r.item_id, "query": r.query_text, "pics": [r.ppn_id]} for r in d.itertuples()]
    return [{"item_id": r.item_id, "query": None, "pics": [r.a, r.b]} for r in d.itertuples()]


def labels():
    p = HERE / "labels.csv"
    return {r["item_id"]: r["label"] for r in csv.DictReader(p.open())} if p.exists() else {}


def thumb(iid, size):
    p = HERE / "cache" / str(size) / (iid.replace("/", "_") + ".jpg")
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(IMG / f"{iid}.jpg").convert("RGB"); im.thumbnail((size, size)); im.save(p, quality=85)
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
            its = items()
            return self.send(json.dumps({"items": its, "labels": labels(), "title": CFG["title"], "help": CFG["help"],
                                         "options": CFG["options"], "pair": its[0]["query"] is None}).encode(), "application/json")
        if u.path == "/img":
            iid = q.get("id", [""])[0]
            if iid not in {p for it in items() for p in it["pics"]}:
                return self.send(b"unknown picture", "text/plain", 404)
            return self.send(thumb(iid, BIG if q.get("s", ["t"])[0] == "b" else THUMB), "image/jpeg")
        self.send(b"not found", "text/plain", 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/label":
            return self.send(b"not found", "text/plain", 404)
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if d.get("item_id") not in {it["item_id"] for it in items()} or d.get("label") not in {v for v, _ in CFG["options"]}:
            return self.send(b"bad label", "text/plain", 400)
        with lock:
            p = HERE / "labels.csv"; new = not p.exists()
            with p.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["item_id", "label", "recorded_at"])
                if new:
                    w.writeheader()
                w.writerow({"item_id": d["item_id"], "label": d["label"], "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self.send(b'{"ok":true}', "application/json")


def main():
    global CFG, HERE
    ap = argparse.ArgumentParser(); ap.add_argument("--task", choices=TASKS, required=True); ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args(); CFG = TASKS[a.task]; HERE = BASE / "self_annotation_package" / CFG["dir"]
    print(f"{a.task} page on http://localhost:{a.port} ({len(items())} items, {len(labels())} judged)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
