#!/usr/bin/env python3
"""Quick labels: the Label Studio tasks still open, as fast grids (like the scene consistency page).

Three tabs, built from the open tasks of the Label Studio projects (exported to
self_annotation_package/quick_labels/*_tasks.csv, read straight from the Label Studio database):

  scene    Task E (12, auto-approved pictures) and Task A2 remaining (13, catalogue default "no scene")
           merged into one list of 871 different pictures in a fixed random order, so the page does
           not show where a picture came from; Task A's 55 open pictures are all in Task A2.
           Rule: people, animals, a landscape or a room = scene; sheet music, text, covers, blank or
           marbled paper, borders, ornaments, initials without figures, diagrams, maps, emblems,
           coats of arms, stamps, single objects and plant studies = no scene.
  object   Task B (5): is the object the describing model named really in the picture? (26 rows)
  classes  Task G v3 (15): every object class shown and the one scene type that fits (295 pictures).

Labels go to self_annotation_package/quick_labels/<tab>_labels.csv (last click per item counts); a
reload continues where you stopped.

    python serve_quick_labels.py [--port 8092]
    then on your own machine:  ssh -L 8092:localhost:8092 <this server>  and open http://localhost:8092
"""
import argparse, csv, json, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd
from PIL import Image

BASE = Path("/data/brhanu/thesis_project")
HERE = BASE / "self_annotation_package/quick_labels"
CACHE = HERE / "cache"
THUMB, BIG = 260, 900
CLASSES = ["person", "child", "animal", "building", "weapon", "vehicle", "tree", "text"]
SCENES = ["teaching", "family", "playing", "landscape", "drawing", "none"]
lock = threading.Lock()


def load():
    s = pd.read_csv(HERE / "scene_items.csv")
    b = pd.read_csv(HERE / "B_tasks.csv"); g = pd.read_csv(HERE / "G_tasks.csv")
    return {
        "scene": [{"key": r.image_id, "path": r.path} for r in s.itertuples()],
        "object": [{"key": str(r.ls_task_id), "path": r.path, "noun": r.claimed_noun} for r in b.itertuples()],
        "classes": [{"key": r.image_id, "path": r.path} for r in g.itertuples()],
    }


ITEMS = load()
PATHS = {t: {i["key"]: i["path"] for i in v} for t, v in ITEMS.items()}


def labels(tab):
    p = HERE / f"{tab}_labels.csv"
    return {r["key"]: r["label"] for r in csv.DictReader(p.open())} if p.exists() else {}


def thumb(path, size):
    c = CACHE / str(size) / (Path(path).stem + ".jpg")
    if not c.exists():
        c.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(path).convert("RGB"); im.thumbnail((size, size)); im.save(c, quality=85)
    return c.read_bytes()


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Quick labels</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f5f1;--ink:#22241f;--muted:#6b6e64;--card:#fff;--line:#d9d7cf;--yes:#2f6f4f;--no:#8a3b2e;--mid:#8a6d1f}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;z-index:2}
h1{font-size:18px;margin:0 0 6px}.rule{color:var(--muted);font-size:13px;max-width:100ch;margin:4px 0}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{font:inherit;border:1px solid var(--line);background:var(--card);border-radius:6px;padding:4px 10px;cursor:pointer;color:var(--ink)}
button:focus-visible{outline:2px solid var(--ink)} .tab.on{background:var(--ink);color:#fff}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px;padding:16px}
.t{background:var(--card);border:2px solid var(--line);border-radius:8px;padding:6px}
.t.done{border-color:var(--yes)} .t img{width:100%;height:240px;object-fit:contain;cursor:zoom-in;background:#eee}
.q{font-weight:600;margin:4px 0}.b{display:flex;gap:6px;margin-top:6px;flex-wrap:wrap}.b button{flex:1}
.b button.on{color:#fff}.b button.on.y{background:var(--yes)}.b button.on.n{background:var(--no)}.b button.on.m{background:var(--mid)}
.chips button{flex:0 0 auto;font-size:13px;padding:2px 8px}.chips button.on{background:var(--ink);color:#fff}
#z{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:5}
#z img{max-width:95vw;max-height:95vh}
</style></head><body>
<header><h1>Quick labels</h1>
<div class="row"><button class="tab" data-t="scene">Scene or no scene</button><button class="tab" data-t="object">Is the object there?</button><button class="tab" data-t="classes">Objects and scene type</button></div>
<div class="rule" id="rule"></div>
<div class="row"><button id="p">Previous screen</button><span id="pg"></span><button id="nx">Next screen</button><span id="c" class="rule"></span></div></header>
<main id="g"></main><div id="z"><img alt=""></div>
<script>
const RULES={scene:"<b>Scene</b>: people, animals, a landscape or a room is shown. <b>No scene</b>: sheet music, text, a cover, blank or marbled paper, a border, an ornament, an initial without figures, a diagram, map or table, an emblem, coat of arms or stamp, a single object, a plant study. <b>Unsure</b> only if you cannot tell. Click a picture to enlarge it.",
 object:"A describing model said the picture shows the object named above it. Is it really there? Click a picture to enlarge it.",
 classes:"Tick every object class you can see, then pick the one scene type that fits best (none if there is no scene). The picture counts as labelled once a scene type is picked."};
const CL=%CLASSES%, SC=%SCENES%, N=30;
let tab='scene',items=[],lab={},page=0;
const g=document.getElementById('g'),z=document.getElementById('z');
async function load(t){tab=t;document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('on',b.dataset.t==t));
 document.getElementById('rule').innerHTML=RULES[t];const d=await (await fetch('/api/state?tab='+t)).json();items=d.items;lab=d.labels;
 const f=items.findIndex(i=>!lab[i.key]);page=f<0?0:Math.floor(f/N);draw()}
async function save(key,label){await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tab,key,label})});lab[key]=label;count()}
function btns(it,t,opts){const b=document.createElement('div');b.className='b';
 for(const [v,txt,cls] of opts){const x=document.createElement('button');x.textContent=txt;x.className=cls+(lab[it.key]==v?' on':'');
  x.onclick=async()=>{await save(it.key,v);b.querySelectorAll('button').forEach(y=>y.classList.remove('on'));x.classList.add('on');t.classList.add('done')};b.appendChild(x)}
 return b}
function classTile(it,t){let st=lab[it.key]?JSON.parse(lab[it.key]):{classes:[],scene:''};
 const ch=document.createElement('div');ch.className='b chips';
 CL.forEach(c=>{const x=document.createElement('button');x.textContent=c;x.classList.toggle('on',st.classes.includes(c));
  x.onclick=async()=>{st.classes=st.classes.includes(c)?st.classes.filter(y=>y!=c):[...st.classes,c];x.classList.toggle('on');if(st.scene)await save(it.key,JSON.stringify(st))};ch.appendChild(x)});
 const sc=document.createElement('div');sc.className='b chips';
 SC.forEach(s=>{const x=document.createElement('button');x.textContent=s;x.classList.toggle('on',st.scene==s);
  x.onclick=async()=>{st.scene=s;sc.querySelectorAll('button').forEach(y=>y.classList.remove('on'));x.classList.add('on');await save(it.key,JSON.stringify(st));t.classList.add('done')};sc.appendChild(x)});
 t.appendChild(ch);t.appendChild(sc)}
function draw(){g.innerHTML='';items.slice(page*N,page*N+N).forEach(it=>{const t=document.createElement('div');t.className='t'+(lab[it.key]?' done':'');
  if(tab=='object'){const q=document.createElement('div');q.className='q';q.textContent='Is there a '+it.noun+'?';t.appendChild(q)}
  const im=document.createElement('img');im.loading='lazy';im.alt='picture';im.src=`/img?tab=${tab}&key=${encodeURIComponent(it.key)}&s=t`;
  im.onclick=()=>{z.querySelector('img').src=`/img?tab=${tab}&key=${encodeURIComponent(it.key)}&s=b`;z.style.display='flex'};t.appendChild(im);
  if(tab=='scene')t.appendChild(btns(it,t,[['scene','Scene','y'],['none','No scene','n'],['unsure','Unsure','m']]));
  else if(tab=='object')t.appendChild(btns(it,t,[['present','Present','y'],['absent','Absent','n'],['ambiguous','Ambiguous','m']]));
  else classTile(it,t);g.appendChild(t)});
 document.getElementById('pg').textContent=`Screen ${page+1} of ${Math.max(1,Math.ceil(items.length/N))}`;count();scrollTo(0,0)}
function count(){document.getElementById('c').textContent=`${items.filter(i=>lab[i.key]).length} of ${items.length} labelled; saved on every click`}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>load(b.dataset.t));
document.getElementById('p').onclick=()=>{if(page>0){page--;draw()}};
document.getElementById('nx').onclick=()=>{if((page+1)*N<items.length){page++;draw()}};
z.onclick=()=>z.style.display='none';load('scene');
</script></body></html>""".replace("%CLASSES%", json.dumps(CLASSES)).replace("%SCENES%", json.dumps(SCENES))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype, code=200):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "max-age=86400")
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query); tab = q.get("tab", [""])[0]
        if u.path == "/":
            return self.send(PAGE.encode(), "text/html; charset=utf-8")
        if u.path == "/api/state" and tab in ITEMS:
            items = [{k: v for k, v in i.items() if k != "path"} for i in ITEMS[tab]]
            return self.send(json.dumps({"items": items, "labels": labels(tab)}).encode(), "application/json")
        if u.path == "/img" and tab in PATHS:
            path = PATHS[tab].get(q.get("key", [""])[0])
            if not path:
                return self.send(b"unknown picture", "text/plain", 404)
            return self.send(thumb(path, BIG if q.get("s", ["t"])[0] == "b" else THUMB), "image/jpeg")
        self.send(b"not found", "text/plain", 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/label":
            return self.send(b"not found", "text/plain", 404)
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tab, key, label = d.get("tab"), d.get("key"), d.get("label")
        ok = {"scene": ("scene", "none", "unsure"), "object": ("present", "absent", "ambiguous")}
        if tab not in PATHS or key not in PATHS[tab] or (tab in ok and label not in ok[tab]):
            return self.send(b"bad label", "text/plain", 400)
        if tab == "classes":
            st = json.loads(label)
            if not set(st.get("classes", [])) <= set(CLASSES) or st.get("scene") not in SCENES:
                return self.send(b"bad label", "text/plain", 400)
        with lock:
            p = HERE / f"{tab}_labels.csv"; new = not p.exists()
            with p.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["key", "label", "recorded_at"])
                if new:
                    w.writeheader()
                w.writerow({"key": key, "label": label, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self.send(b'{"ok":true}', "application/json")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8092); a = ap.parse_args()
    print(f"Quick labels on http://localhost:{a.port}: " + ", ".join(f"{t} {len(v)}" for t, v in ITEMS.items()), flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
