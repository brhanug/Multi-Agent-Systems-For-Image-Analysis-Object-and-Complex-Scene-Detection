from html import escape as e
from pathlib import Path
import json
# every status, link and note lives in roadmap.json: update it after each run, rebuild, republish
D = json.loads((Path(__file__).parent / "roadmap.json").read_text())
COLUMNS = [(c["name"], c["window"]) for c in D["columns"]]
NODES = [(n["col"], n["row"], n["id"], n["title"], n["status"], n["have"], n["next"]) for n in D["nodes"]]
EDGES = [(e["from"], e["to"], e["label"], e["critical"]) for e in D["edges"]]
TL = [(t["kind"], t["date"], t["text"]) for t in D["timeline"]]
QUICK = [(q["who"], q["title"], q["text"]) for q in D["this_week"]]
FIXES = D["corrections"]
from roadmap_css import CSS

HERE = Path(__file__).parent
NW, NH, PITCH, X0, Y0, RP = 164, 56, 212, 14, 84, 80
W = X0 + PITCH * (len(COLUMNS) - 1) + NW + 40
H = Y0 + RP * 7 + 8
STATUS_WORD = {"done": "done", "partly": "partly done", "todo": "not started", "you": "needs you", "milestone": ""}
node = {n[2]: dict(col=n[0], row=n[1], id=n[2], title=n[3], st=n[4], have=n[5], nxt=n[6],
                   x=X0 + n[0] * PITCH, y=Y0 + n[1] * RP) for n in NODES}
rects = [(v["x"] - 2, v["y"] - 2, v["x"] + NW + 2, v["y"] + NH + 2) for v in node.values()]


def free(x, y, w, h=13):
    box = (x - w / 2, y - h, x + w / 2, y + 3)
    if box[0] < 0 or box[2] > W: return False
    return not any(box[0] < r[2] and box[2] > r[0] and box[1] < r[3] and box[3] > r[1] for r in rects)


def bez(p0, p1, p2, p3, t):
    return tuple((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t * t * c + t ** 3 * d for a, b, c, d in zip(p0, p1, p2, p3))


svg, labels, unplaced = [], [], set()
for c, (name, sub) in enumerate(COLUMNS):
    x = X0 + c * PITCH
    svg.append(f'<rect class="col-band" x="{x-8}" y="58" width="{NW+16}" height="{H-62}" rx="8"/>')
    svg.append(f'<text class="col-h" x="{x}" y="26">{e(name)}</text><text class="col-s" x="{x}" y="44">{e(sub)}</text>')
for s, t, lab, crit in sorted(EDGES, key=lambda k: k[3]):
    a, b = node[s], node[t]
    cls = "edge crit" if crit else "edge"
    ay, by = a["y"] + NH / 2, b["y"] + NH / 2
    if a["col"] == b["col"]:
        sx = a["x"] + NW; k = 24
        p = [(sx, ay), (sx + k, ay), (sx + k, by), (sx + 7, by)]
        head = f'{sx},{by} {sx+8},{by-4.5} {sx+8},{by+4.5}'
    else:
        sx, tx = a["x"] + NW, b["x"]; k = min(70, (tx - sx) / 2)
        p = [(sx, ay), (sx + k, ay), (tx - k, by), (tx - 7, by)]
        head = f'{tx},{by} {tx-8},{by-4.5} {tx-8},{by+4.5}'
    svg.append(f'<path class="{cls}" d="M{p[0][0]},{p[0][1]} C{p[1][0]},{p[1][1]} {p[2][0]},{p[2][1]} {p[3][0]},{p[3][1]}"/>'
               f'<polygon class="ah{" crit" if crit else ""}" points="{head}"/>')
    if lab:
        wlab = len(lab) * 6.0 + 8
        for tt in (0.5, 0.35, 0.65, 0.25, 0.75):
            lx, ly = bez(*p, tt)
            if free(lx, ly - 4, wlab):
                labels.append(f'<text class="elab{" crit" if crit else ""}" x="{lx:.0f}" y="{ly-6:.0f}" text-anchor="middle">{e(lab)}</text>')
                rects.append((lx - wlab / 2, ly - 19, lx + wlab / 2, ly - 3))
                break
        else:
            unplaced.add((s, t))
for v in node.values():
    x, y = v["x"], v["y"]
    if v["st"] == "milestone":
        svg.append(f'<g class="n milestone"><rect class="box" x="{x}" y="{y+8}" width="{NW}" height="{NH-16}" rx="20"/>'
                   f'<text class="nt" x="{x+NW/2}" y="{y+NH/2+5}" text-anchor="middle">{e(v["title"])}</text></g>')
        continue
    svg.append(f'<g class="n {v["st"]}"><title>{e(v["id"])} {e(v["title"])}: {e(STATUS_WORD[v["st"]])}</title>'
               f'<rect class="box" x="{x}" y="{y}" width="{NW}" height="{NH}" rx="7"/>'
               f'<text class="nid" x="{x+11}" y="{y+19}">{e(v["id"])}</text>'
               f'<text class="nst" x="{x+NW-10}" y="{y+19}" text-anchor="end">{e(STATUS_WORD[v["st"]])}</text>'
               f'<text class="nt" x="{x+11}" y="{y+40}">{e(v["title"])}</text></g>')
svg_doc = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Dependency map of the thesis plan: decisions and Phase 0 feed '
           f'annotation and per-class scoring, which feed the decisive RQ1 test, the risk-control chapter and the submission.">'
           + "".join(svg) + "".join(labels) + "</svg>")

counts = {k: sum(1 for v in node.values() if v["st"] == k) for k in ("done", "partly", "todo", "you")}
feeds = {k: [] for k in node}
for s, t, lab, crit in EDGES:
    feeds[s].append(f'{t} {node[t]["title"]}' + (f' ({lab})' if lab else ""))
rows_html = []
for c, (name, sub) in enumerate(COLUMNS):
    items = sorted((v for v in node.values() if v["col"] == c and v["st"] != "milestone"), key=lambda v: v["row"])
    r = "".join(
        f'<div class="row"><div class="who"><span class="mono">{e(v["id"])}</span><strong>{e(v["title"])}</strong>'
        f'<span class="pill {v["st"]}">{e(STATUS_WORD[v["st"]])}</span></div>'
        f'<div><span class="lbl">What exists</span><p>{e(v["have"])}</p></div>'
        f'<div><span class="lbl">What is next</span><p>{e(v["nxt"])}</p>'
        + (f'<p style="margin-top:6px;color:var(--muted);font-size:13px">Feeds → {e("; ".join(feeds[v["id"]]))}</p>' if feeds[v["id"]] else "")
        + '</div></div>' for v in items)
    rows_html.append(f'<div class="phase"><div class="eyebrow">{e(name)} · {e(sub)}</div>{r}</div>')

html = f"""<title>Colibri Thesis Roadmap</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=Instrument+Sans:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>{CSS}</style>
<main class="wrap">
<header>
  <div class="eyebrow">Master's thesis · Universität Hildesheim · status on {D["status_date"]}</div>
  <h1>Colibri thesis roadmap</h1>
  <p class="lede">Every item in the experiment plan: what already exists, what is missing, and what it unlocks. Read the map left to right. Your decisions feed Phase 0, annotation feeds the per-class work, and everything converges on the 29 September document and the 7 December submission.</p>
  <div class="tally">
    <span class="chip s-done"><span class="dot"></span><b>{counts['done']}</b> done</span>
    <span class="chip s-partly"><span class="dot"></span><b>{counts['partly']}</b> partly done</span>
    <span class="chip s-todo"><span class="dot"></span><b>{counts['todo']}</b> not started</span>
    <span class="chip s-you"><span class="dot"></span><b>{counts['you']}</b> need your input</span>
    <span class="chip"><span class="dot crit"></span>critical path: annotation → negatives → decisive test</span>
  </div>
</header>
<section aria-label="Timeline"><div class="timeline">{"".join(f'<div class="tl {k}"><strong>{d}</strong>{e(t)}</div>' for k, d, t in TL)}</div></section>
<figure style="margin:0"><div class="mapbox">{svg_doc}</div>
<figcaption>Boxes are plan items, coloured by status; arrows show what each item feeds. The teal path is the critical one: without new annotation and more confirmed negatives, the decisive per-class test (C4) runs on the same 38 negatives as today. On a phone, scroll the map sideways.</figcaption></figure>
<section><h2>This week</h2><div class="quick">{"".join(f'<div class="qw"><div class="when">{e(w)}</div><h3>{e(h)}</h3><p>{e(p)}</p></div>' for w, h, p in QUICK)}</div></section>
<section class="fixes"><h2>Corrections waiting for your go-ahead</h2><p style="margin:6px 0 0;color:var(--muted);font-size:14px">Found while checking the plan against the data. None has been applied to the thesis, slides or emails yet (Q3).</p>
<ul>{"".join(f"<li>{e(f)}</li>" for f in FIXES)}</ul></section>
<details open><summary>Every item, in detail</summary>{"".join(rows_html)}</details>
</main>"""
(HERE / "roadmap.html").write_text(html)
print("wrote", HERE / "roadmap.html", "| counts", counts, "| labels not placed:", sorted(unplaced))
