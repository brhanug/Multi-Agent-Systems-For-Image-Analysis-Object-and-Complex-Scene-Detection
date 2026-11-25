#!/usr/bin/env python3
import re
from pathlib import Path

BASE_DIR = Path('/data/brhanu/thesis_project')

def parse_file(filepath, defined_labels, referenced_labels):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return

    # Stripping LaTeX comments to avoid parsing commented-out labels/refs
    lines = content.split('\n')
    clean_lines = []
    for line in lines:
        comment_idx = line.find('%')
        if comment_idx != -1:
            escaped = False
            if comment_idx > 0 and line[comment_idx - 1] == '\\':
                escaped = True
            if not escaped:
                line = line[:comment_idx]
        clean_lines.append(line)
    clean_content = '\n'.join(clean_lines)

    # Find labels
    for lbl in re.findall(r'\\label\{([^}]+)\}', clean_content):
        defined_labels[lbl] = filepath.name

    # Find refs
    for ref in re.findall(r'\\ref\{([^}]+)\}', clean_content):
        referenced_labels.setdefault(ref, []).append(filepath.name)
    for ref in re.findall(r'\\pageref\{([^}]+)\}', clean_content):
        referenced_labels.setdefault(ref, []).append(filepath.name)

    # Find inputs
    inputs = re.findall(r'\\input\{([^}]+)\}', clean_content)
    for inp in inputs:
        inp_path = BASE_DIR / inp
        if not inp_path.suffix:
            inp_path = inp_path.with_suffix('.tex')
        parse_file(inp_path, defined_labels, referenced_labels)

def main():
    defined_labels = {}
    referenced_labels = {}
    
    # Parse main file
    parse_file(BASE_DIR / 'thesis_v3_updated.tex', defined_labels, referenced_labels)
    
    print(f"Total defined labels found: {len(defined_labels)}")
    print(f"Total unique referenced labels: {len(referenced_labels)}")
    
    broken_refs = {}
    for ref, files in referenced_labels.items():
        if ref not in defined_labels:
            broken_refs[ref] = files
            
    if broken_refs:
        print("\n❌ BROKEN CROSS-REFERENCES FOUND:")
        for ref, files in sorted(broken_refs.items()):
            print(f"  - Label '{ref}' is referenced in {files} but is NEVER defined.")
    else:
        print("\n✅ No broken cross-references found in recursive scan!")

if __name__ == '__main__':
    main()
