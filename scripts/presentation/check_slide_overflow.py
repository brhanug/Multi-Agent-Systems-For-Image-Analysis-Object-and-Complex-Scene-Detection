#!/usr/bin/env python3
"""Catch text Beamer has silently dropped off the bottom of a slide.

Beamer does not draw past the frame edge and does not warn: overflowing text simply
vanishes from the PDF. Every dense slide in this project has hit it at least once, so each
slide's final sentence is probed here against the rendered text.

Four things produce false alarms and are normalised away before comparing, because each
cost real time when it was not:

    ligatures          "conﬁdence" is one glyph, not "confidence"
    hyphenation        "charac-\nters" is one word split across a line
    curly quotes       LaTeX turns "x" into “x”
    unicode minus      $-0.3$ renders U+2212, not a hyphen

    python check_slide_overflow.py <file.pdf> "closing line one" "closing line two" ...

Exits non-zero if any probe is missing, so it can gate a build.
"""
import re, sys
import pymupdf

SUBS = {0xfb00: "ff", 0xfb01: "fi", 0xfb02: "fl", 0xfb03: "ffi", 0xfb04: "ffl",
        0x2212: "-", 0x2013: "-", 0x2014: "--",
        0x2018: "'", 0x2019: "'", 0x201c: '"', 0x201d: '"'}


def norm(s):
    for k, v in SUBS.items():
        s = s.replace(chr(k), v)
    s = re.sub(r"-\n", "", s)
    return re.sub(r"\s+", " ", s).strip()


def check(pdf, probes):
    doc = pymupdf.open(pdf)
    pages = {i + 1: norm(p.get_text()) for i, p in enumerate(doc)}
    missing = []
    for pr in probes:
        hit = [k for k, v in pages.items() if norm(pr) in v]
        print(f"  {'OK  p%-3d' % hit[0] if hit else 'MISS     '}  {pr[:66]}")
        if not hit:
            missing.append(pr)
    print(f"{pdf}: {len(probes) - len(missing)}/{len(probes)} render  ({doc.page_count} pages)")
    return missing


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    raise SystemExit(1 if check(sys.argv[1], sys.argv[2:]) else 0)
