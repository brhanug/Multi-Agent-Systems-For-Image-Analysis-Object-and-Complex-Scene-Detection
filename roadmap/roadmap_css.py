CSS = r"""
:root{
  --ground:#F2F4F1; --surface:#FFFFFF; --ink:#17201B; --muted:#5A6861; --line:#C9D2CC; --grid:#E3E8E4;
  --accent:#1B6A72; --accent-soft:#D9ECEE;
  --done:#2B7A4B; --done-bg:#E1F0E6; --partly:#A0650F; --partly-bg:#FAEDD5;
  --todo:#6F7B8E; --todo-bg:#FFFFFF; --you:#B03A2A; --you-bg:#FAE5E0; --mile:#17201B; --mile-ink:#FFFFFF;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#0E1311; --surface:#151C19; --ink:#E3EAE6; --muted:#9AA79F; --line:#35413B; --grid:#212A26;
    --accent:#56B7C0; --accent-soft:#16313A;
    --done:#63C189; --done-bg:#16301F; --partly:#E3A84D; --partly-bg:#33270E;
    --todo:#95A0B2; --todo-bg:#151C19; --you:#F08270; --you-bg:#3A1C17; --mile:#E3EAE6; --mile-ink:#0E1311;
  }
}
:root[data-theme="dark"]{
  --ground:#0E1311; --surface:#151C19; --ink:#E3EAE6; --muted:#9AA79F; --line:#35413B; --grid:#212A26;
  --accent:#56B7C0; --accent-soft:#16313A;
  --done:#63C189; --done-bg:#16301F; --partly:#E3A84D; --partly-bg:#33270E;
  --todo:#95A0B2; --todo-bg:#151C19; --you:#F08270; --you-bg:#3A1C17; --mile:#E3EAE6; --mile-ink:#0E1311;
}
body{background:var(--ground);color:var(--ink);font-family:"Instrument Sans",system-ui,-apple-system,"Segoe UI",sans-serif;font-size:15px;line-height:1.55}
.wrap{max-width:1280px;margin:0 auto;padding-inline:20px;padding-block:28px 64px;display:grid;gap:36px}
h1,h2{font-family:"Bricolage Grotesque","Instrument Sans",system-ui,sans-serif;text-wrap:balance;margin:0;letter-spacing:-.01em}
h1{font-size:clamp(28px,4vw,40px);line-height:1.1}
h2{font-size:21px}
.mono{font-family:"JetBrains Mono",ui-monospace,Menlo,monospace}
.lede{color:var(--muted);max-width:68ch;margin:10px 0 0}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.tally{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px}
.chip{display:inline-flex;align-items:center;gap:8px;padding:5px 12px;border-radius:999px;border:1px solid var(--line);background:var(--surface);font-size:13px}
.chip b{font-variant-numeric:tabular-nums}
.dot{width:10px;height:10px;border-radius:3px;border:1.5px solid}
.s-done .dot,.dot.s-done{background:var(--done-bg);border-color:var(--done)}
.s-partly .dot,.dot.s-partly{background:var(--partly-bg);border-color:var(--partly)}
.s-todo .dot,.dot.s-todo{background:var(--todo-bg);border-color:var(--todo)}
.s-you .dot,.dot.s-you{background:var(--you-bg);border-color:var(--you)}
.dot.crit{background:var(--accent);border-color:var(--accent);height:3px;width:18px;border-radius:2px}
/* timeline */
.timeline{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));border-top:2px solid var(--ink);margin-top:6px}
.tl{padding:10px 8px 0 0;font-size:13px;color:var(--muted);position:relative}
.tl::before{content:"";position:absolute;top:-7px;left:0;width:12px;height:12px;border-radius:50%;background:var(--ground);border:2px solid var(--ink)}
.tl.now::before{background:var(--accent);border-color:var(--accent)}
.tl.key::before{background:var(--ink)}
.tl strong{display:block;color:var(--ink);font-size:14px}
/* map */
.mapbox{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px;overflow-x:auto}
.mapbox svg{display:block;min-width:1180px;width:100%;height:auto}
figcaption{color:var(--muted);font-size:13px;margin-top:10px;max-width:90ch}
.col-h{font-family:"Bricolage Grotesque",system-ui,sans-serif;font-size:15px;font-weight:600;fill:var(--ink)}
.col-s{font-size:11.5px;fill:var(--muted)}
.col-band{fill:var(--grid);opacity:.55}
.edge{fill:none;stroke:var(--line);stroke-width:1.5}
.edge.crit{stroke:var(--accent);stroke-width:2.5}
.elab{font-size:10.5px;fill:var(--muted);paint-order:stroke;stroke:var(--surface);stroke-width:4px}
.elab.crit{fill:var(--accent);font-weight:600}
.ah{fill:var(--line)} .ah.crit{fill:var(--accent)}
.n rect.box{stroke-width:1.5}
.n.done rect.box{fill:var(--done-bg);stroke:var(--done)}
.n.partly rect.box{fill:var(--partly-bg);stroke:var(--partly)}
.n.todo rect.box{fill:var(--todo-bg);stroke:var(--todo)}
.n.you rect.box{fill:var(--you-bg);stroke:var(--you);stroke-width:2.5}
.n.milestone rect.box{fill:var(--mile);stroke:var(--mile)}
.n .nid{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:11px;font-weight:600;fill:var(--muted)}
.n .nst{font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;fill:var(--muted)}
.n.done .nst{fill:var(--done)} .n.partly .nst{fill:var(--partly)} .n.you .nst{fill:var(--you)}
.n .nt{font-size:13px;font-weight:600;fill:var(--ink)}
.n.milestone .nt{fill:var(--mile-ink);font-size:13.5px}
/* next-steps + details */
.quick{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin-top:14px}
.qw{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.qw .when{font-size:12px;color:var(--muted)}
.qw h3{margin:4px 0 6px;font-size:15.5px}
.qw p{margin:0;color:var(--muted);font-size:14px}
.phase{display:grid;gap:10px;margin-top:14px}
.row{display:grid;grid-template-columns:150px 1fr 1fr;gap:18px;padding:12px 0;border-top:1px solid var(--line);align-items:start}
.row .who{display:grid;gap:4px}
.row .who .mono{font-size:13px;color:var(--muted)}
.row .who strong{font-size:15px;line-height:1.3}
.row .lbl{font-size:11.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);display:block;margin-bottom:2px}
.row p{margin:0;font-size:14px}
.pill{justify-self:start;font-size:11.5px;padding:2px 9px;border-radius:999px;border:1.5px solid}
.pill.done{color:var(--done);border-color:var(--done);background:var(--done-bg)}
.pill.partly{color:var(--partly);border-color:var(--partly);background:var(--partly-bg)}
.pill.todo{color:var(--todo);border-color:var(--todo)}
.pill.you{color:var(--you);border-color:var(--you);background:var(--you-bg)}
.fixes{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px 20px}
.fixes ul{margin:10px 0 0;padding-left:18px;display:grid;gap:6px}
.fixes li{font-size:14px}
details summary{cursor:pointer;font-family:"Bricolage Grotesque",system-ui,sans-serif;font-size:21px;font-weight:600}
details summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
@media (max-width:720px){
  .row{grid-template-columns:1fr;gap:6px}
  .timeline{grid-template-columns:repeat(2,minmax(0,1fr));row-gap:16px}
}
"""
