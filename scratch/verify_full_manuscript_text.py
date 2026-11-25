#!/usr/bin/env python3
import re

THESIS_PATH = "/data/brhanu/thesis_project/presentation/thesis_v3_updated.tex"

with open(THESIS_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.split('\n')

print("=" * 60)
print("🔍 FULL MANUSCRIPT TEXT & FORMATTING VERIFICATION")
print("=" * 60)

# Check for common typos
common_typos = ['teh', 'adn', 'triage', 'experment', 'analysys', 'optimis', 'optimz', 'archival', 'historica', 'clasification']
found_typos = []

for i, line in enumerate(lines, 1):
    line_no_comment = line.split('%')[0]
    words = re.findall(r'\b[a-zA-Z]+\b', line_no_comment)
    for w in words:
        w_lower = w.lower()
        if w_lower in ['teh', 'adn', 'experment', 'analysys', 'clasification', 'historica']:
            found_typos.append((i, w, line.strip()))

print(f"\n[1] Spelling Typo Scan:")
if found_typos:
    for l, w, snip in found_typos:
        print(f"  Line {l}: Typo '{w}' in: {snip[:80]}")
else:
    print("  ✅ PASS: Zero common typos found")

# Check for math mode consistency ($ ... $)
math_inconsistencies = []
for i, line in enumerate(lines, 1):
    line_no_comment = line.split('%')[0]
    # Check if $ count is odd
    dollar_count = line_no_comment.count('$') - line_no_comment.count(r'\$')
    if dollar_count % 2 != 0:
        math_inconsistencies.append((i, line.strip()))

print(f"\n[2] Math Mode Unbalanced $ Scan:")
if math_inconsistencies:
    for l, snip in math_inconsistencies:
        print(f"  Line {l}: {snip[:80]}")
else:
    print("  ✅ PASS: All inline math environments $...$ are balanced")

# Check for section headers
headers = []
for i, line in enumerate(lines, 1):
    if line.strip().startswith('\\section') or line.strip().startswith('\\subsection') or line.strip().startswith('\\subsubsection'):
        headers.append((i, line.strip()))

print(f"\n[3] Total Section Headers: {len(headers)}")
for i, h in headers:
    print(f"  Line {i}: {h[:80]}")

print("\n" + "=" * 60)
