#!/usr/bin/env python3
"""Bring the documents' figures to a later rerun's results, through the figure checkers.

Each checker (thesis_rerun_figures.py, thesis_v4_chapter6_check.py, thesis_c1_figures_check.py) lists a
piece of text with ⟨⟩ where a figure sits and the result key it comes from. For every entry the old
value is rebuilt from the rerun's backup, and the text with that old value must occur exactly once in
the documents; otherwise the entry is refused (the mapping, not the number, would be wrong). All
figures are then replaced at once from their positions in the unchanged text, so entries whose context
holds another entry's figure (rows of a table) do not block each other. Finally the numbers inside the
checkers' own context strings are moved to the new values, so that the checker passes afterwards.

    python apply_rerun_to_documents.py --backup reruns/rerun_2026-11-03 [--dry-run]
"""
import argparse, importlib.util, json, re, sys
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")
CHECKERS = ["thesis_rerun_figures.py", "thesis_v4_chapter6_check.py", "thesis_c1_figures_check.py"]


def load(name):
    spec = importlib.util.spec_from_file_location(name[:-3], BASE / "scripts" / name)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def entries(mod, backup):
    """(ctx, mark, old figure options, new figure) per entry."""
    out = []
    if hasattr(mod, "E"):                                        # thesis_rerun_figures.py
        for ctx, src, fmt in mod.E:
            f = mod.FMT[fmt]; vo, vn = mod.get(backup / "results/signal_rebuild", src), mod.get(mod.NOW, src)
            nud = (lambda d: tuple(x + d for x in vo)) if isinstance(vo, tuple) else (lambda d: vo + d)
            out.append((ctx, mod.MARK, [f(vo), f(nud(1e-7)), f(nud(-1e-7))], f(vn)))
        return out
    now = [(ctx, mod.M, fmt(mod.val(spec))) for ctx, spec, fmt in mod.ENTRIES]
    for attr, sub in (("SR", "results/signal_rebuild"), ("MA", "results/multi_agent")):
        if hasattr(mod, attr):
            setattr(mod, attr, backup / sub)
    mod._cache.clear()
    for (ctx, M, new), (_c, spec, fmt) in zip(now, mod.ENTRIES):
        out.append((ctx, M, [fmt(mod.val(spec))], new))
    return out


def files_of(mod):
    return [mod.THESIS] if hasattr(mod, "THESIS") else list(mod.FILES)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--backup", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); backup = BASE / a.backup / "backup"
    mods = {c: load(c) for c in CHECKERS}
    texts = {}
    for m in mods.values():
        for p in files_of(m):
            texts.setdefault(p, p.read_text())
    plan, ctxfix, problems = {}, [], []                         # plan: (path, pos) -> (old, new)
    per_entry = []
    for c, m in mods.items():
        own = {q: texts[q] for q in files_of(m)}             # a checker looks only at its own files
        for i, (ctx, M, olds, new) in enumerate(entries(m, backup)):
            hit = None
            for old in dict.fromkeys(olds):
                oc = ctx.replace(M, old)
                found = [(p, t.find(oc)) for p, t in own.items() if t.count(oc) == 1]
                n = sum(t.count(oc) for t in own.values())
                if n == 1 and found:
                    hit = (found[0][0], found[0][1], old); break
            if hit is None:
                nc = ctx.replace(M, new)
                if sum(t.count(nc) for t in own.values()) == 1:
                    continue                                    # already current
                problems.append(f"{c}: old text not found once: {ctx.replace(M, olds[0])[:100]!r}"); continue
            p, start, old = hit; pos = start + ctx.index(M)
            if old != new:
                prev = plan.get((p, pos))
                if prev and prev[1] != new:
                    problems.append(f"{c}: two new values for one figure: {ctx[:80]!r}")
                plan[(p, pos)] = (old, new)
            per_entry.append((c, i, ctx, M, p, start, pos, old))
    # apply all figures from the end of each file backwards
    newtexts = dict(texts)
    for p in texts:
        t = texts[p]
        for (pp, pos), (old, new) in sorted(((k, v) for k, v in plan.items() if k[0] == p), key=lambda kv: -kv[0][1]):
            assert t[pos:pos + len(old)] == old
            t = t[:pos] + new + t[pos + len(old):]
        newtexts[p] = t
    # the checkers' contexts: numbers belonging to other entries' figures that moved
    for c, i, ctx, M, p, start, pos, old in per_entry:
        inside = [(q, v) for (pp, q), v in plan.items() if pp == p and start <= q < start + len(ctx.replace(M, old)) and q != pos]
        for q, (o, n) in inside:
            ctxfix.append((c, o, n))
    print(f"figures to change: {len(plan)}; context numbers to move in checkers: {len(set(ctxfix))}; problems: {len(problems)}")
    for pr in problems:
        print("  PROBLEM", pr)
    if a.dry_run or problems:
        print("nothing written" if problems else "dry run: nothing written"); return 1 if problems else 0
    for p, t in newtexts.items():
        if t != texts[p]:
            p.write_text(t)
    num = re.compile(r"-?\d+(?:\.\d+)?")
    for c in CHECKERS:
        src = (BASE / "scripts" / c).read_text(); lines = src.split("\n")
        for _c, o, n in set(x for x in ctxfix if x[0] == c):
            co, cn = num.findall(o), num.findall(n)
            if len(co) != 1 or len(cn) != 1:
                print(f"  {c}: move by hand: {o!r} -> {n!r}"); continue
            pat = re.compile(r"(?<![\d.])" + re.escape(co[0]) + r"(?![\d])")
            for k, line in enumerate(lines):
                if "⟨⟩" in line and pat.search(line):
                    lines[k] = pat.sub(cn[0], line)
        (BASE / "scripts" / c).write_text("\n".join(lines))
    print("written; now run each checker")
    return 0


if __name__ == "__main__":
    sys.exit(main())
