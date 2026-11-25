#!/usr/bin/env python3
import re

THESIS_PATH = "/data/brhanu/thesis_project/presentation/thesis_v3_updated.tex"

with open(THESIS_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.split('\n')

print("=" * 60)
print("🔍 DEEP CONTENT & TEXTUAL ERROR CHECK")
print("=" * 60)

# 1. Search for repeated words (e.g. "the the", "is is")
dup_words = []
dup_pattern = re.compile(r'\b([a-zA-Z]{2,})\s+\1\b', re.IGNORECASE)

for i, line in enumerate(lines, 1):
    # Ignore comments
    line_no_comment = line.split('%')[0]
    matches = dup_pattern.findall(line_no_comment)
    if matches:
        for m in matches:
            # ignore legitimate repeated words if any
            if m.lower() not in ['that']:
                dup_words.append((i, m, line.strip()))

print(f"\n[1] Duplicate Words Found ({len(dup_words)}):")
for line_num, word, snippet in dup_words:
    print(f"  Line {line_num}: '{word}' in: {snippet[:80]}")

# 2. Search for residual FIXME, TODO, XXX, PLACEHOLDER
placeholders = []
p_pattern = re.compile(r'\b(TODO|FIXME|XXX|PLACEHOLDER|TBD|INSERT)\b', re.IGNORECASE)

for i, line in enumerate(lines, 1):
    line_no_comment = line.split('%')[0]
    matches = p_pattern.findall(line_no_comment)
    if matches:
        placeholders.append((i, matches[0], line.strip()))

print(f"\n[2] Placeholder / TODO Tags Found ({len(placeholders)}):")
for line_num, tag, snippet in placeholders:
    print(f"  Line {line_num}: [{tag}] {snippet[:80]}")

# 3. Check for double punctuation or bad spacing
spacing_issues = []
sp_pattern = re.compile(r'(\s{2,}[,\.\?:;]|\s+[\.\,]\s+)')

for i, line in enumerate(lines, 1):
    line_no_comment = line.split('%')[0]
    if sp_pattern.search(line_no_comment):
        spacing_issues.append((i, line.strip()))

print(f"\n[3] Spacing/Punctuation Formatting Anomalies ({len(spacing_issues)}):")
for line_num, snippet in spacing_issues[:10]:
    print(f"  Line {line_num}: {snippet[:80]}")

# 4. Check for broken or unescaped % or _ in text outside math/commands
unescaped_underscore = []
for i, line in enumerate(lines, 1):
    line_no_comment = line.split('%')[0]
    # Check if there are underscores in plain text (not in \texttt, \label, \ref, \cite, \url, \includegraphics, or math $)
    cleaned = re.sub(r'\\(texttt|label|ref|cite|url|includegraphics|input|path)\{[^}]*\}', '', line_no_comment)
    cleaned = re.sub(r'\$[^\$]*\$', '', cleaned)
    cleaned = re.sub(r'\\[a-zA-Z]+', '', cleaned)
    if '_' in cleaned:
        unescaped_underscore.append((i, line.strip()))

print(f"\n[4] Unescaped Underscores in Plain Text ({len(unescaped_underscore)}):")
for line_num, snippet in unescaped_underscore[:10]:
    print(f"  Line {line_num}: {snippet[:80]}")

print("\n" + "=" * 60)
