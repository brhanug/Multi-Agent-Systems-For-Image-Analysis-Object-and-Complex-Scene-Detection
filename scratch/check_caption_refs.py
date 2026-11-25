#!/usr/bin/env python3
import re

def parse_nested_braces(text, command):
    matches = []
    pattern = re.compile(rf'\\{command}\s*{{')
    for m in pattern.finditer(text):
        start_idx = m.end()
        # Track braces to find the matching close brace
        stack = 1
        curr_idx = start_idx
        while stack > 0 and curr_idx < len(text):
            if text[curr_idx] == '{':
                stack += 1
            elif text[curr_idx] == '}':
                stack -= 1
            curr_idx += 1
        if stack == 0:
            matches.append((m.start(), text[start_idx:curr_idx-1]))
    return matches

with open('/data/brhanu/thesis_project/presentation/thesis_v3_updated.tex', 'r') as f:
    content = f.read()

# Check captions
captions = parse_nested_braces(content, 'caption')
print(f"Total captions found: {len(captions)}")
for start_pos, cap in captions:
    # Find line number
    line_num = content[:start_pos].count('\n') + 1
    ref_matches = re.findall(r'\\ref\{([^}]+)\}', cap)
    pageref_matches = re.findall(r'\\pageref\{([^}]+)\}', cap)
    if ref_matches or pageref_matches:
        print(f"Line {line_num:4d}: \\caption has refs {ref_matches + pageref_matches} | Content: {cap[:150]}...")

# Check footnotes
footnotes = parse_nested_braces(content, 'footnote')
print(f"Total footnotes found: {len(footnotes)}")
for start_pos, fn in footnotes:
    line_num = content[:start_pos].count('\n') + 1
    ref_matches = re.findall(r'\\ref\{([^}]+)\}', fn)
    pageref_matches = re.findall(r'\\pageref\{([^}]+)\}', fn)
    if ref_matches or pageref_matches:
        print(f"Line {line_num:4d}: \\footnote has refs {ref_matches + pageref_matches} | Content: {fn[:150]}...")
