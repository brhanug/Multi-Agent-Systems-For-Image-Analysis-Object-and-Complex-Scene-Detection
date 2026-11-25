#!/usr/bin/env python3
"""After a full rerun: what changed, against the backup taken before it started.

Walks every JSON result under the backup's results/ and the live results/, and reports each file as
unchanged, changed, new or missing. For a changed file it lists the numeric values that moved, largest
move first, so the figures a document quotes can be checked against what the rerun produced.

    python rerun_change_report.py <backup_dir> <out_markdown>
"""
import json, sys
from pathlib import Path

BASE = Path("/data/brhanu/thesis_project")


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from leaves(v, f"{path}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield path, float(x)


def main():
    bk, out = Path(sys.argv[1]) / "results", Path(sys.argv[2])
    old = {p.relative_to(bk): p for p in bk.rglob("*.json")}
    new = {p.relative_to(BASE / "results"): p for p in (BASE / "results").rglob("*.json")}
    unchanged, changed, missing, added = [], [], sorted(set(old) - set(new)), sorted(set(new) - set(old))
    for rel in sorted(set(old) & set(new)):
        try:
            a, b = json.loads(old[rel].read_text()), json.loads(new[rel].read_text())
        except Exception:
            (unchanged if old[rel].read_bytes() == new[rel].read_bytes() else changed).append((rel, [])); continue
        if a == b:
            unchanged.append((rel, [])); continue
        la, lb = dict(leaves(a)), dict(leaves(b))
        moves = sorted(((k, la[k], lb[k]) for k in set(la) & set(lb) if abs(la[k] - lb[k]) > 1e-9),
                       key=lambda t: -abs(t[2] - t[1]))
        changed.append((rel, moves))
    lines = [f"# What the full rerun changed\n",
             f"Backup compared: `{sys.argv[1]}`\n",
             f"- unchanged: {len(unchanged)} files",
             f"- changed: {len(changed)} files",
             f"- new: {len(added)} files",
             f"- missing after the rerun: {len(missing)} files\n"]
    if missing:
        lines += ["## Missing after the rerun (a step did not produce them)\n"] + [f"- `{m}`" for m in missing] + [""]
    lines += ["## Changed files, largest numeric moves first\n"]
    for rel, moves in sorted(changed, key=lambda c: -max((abs(m[2] - m[1]) for m in c[1]), default=0)):
        lines.append(f"### `{rel}` — {len(moves)} values moved")
        for k, a, b in moves[:12]:
            lines.append(f"- `{k}`: {a:.6g} → {b:.6g}")
        if len(moves) > 12: lines.append(f"- … and {len(moves) - 12} more")
        lines.append("")
    if added:
        lines += ["## New files\n"] + [f"- `{a}`" for a in added]
    out.write_text("\n".join(lines) + "\n")
    print(f"unchanged {len(unchanged)}, changed {len(changed)}, new {len(added)}, missing {len(missing)} -> {out}")


if __name__ == "__main__":
    main()
