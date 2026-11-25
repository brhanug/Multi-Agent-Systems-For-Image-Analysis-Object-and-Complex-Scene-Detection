#!/usr/bin/env python3
import re
from pathlib import Path

def find_labels_and_refs(filepath):
    print(f"Scanning {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all labels: \label{name}
    labels = set(re.findall(r'\\label\{([^}]+)\}', content))
    
    # Find all references: \ref{name} and \pageref{name} and \cite{name}
    refs = re.findall(r'\\ref\{([^}]+)\}', content)
    pagerefs = re.findall(r'\\pageref\{([^}]+)\}', content)
    
    # Check undefined
    undefined_refs = []
    for r in refs + pagerefs:
        if r not in labels:
            undefined_refs.append(r)
            
    print(f"Defined labels count: {len(labels)}")
    print(f"Undefined references count: {len(undefined_refs)}")
    if undefined_refs:
        print("Undefined references:")
        for r in sorted(set(undefined_refs)):
            # find line number
            for i, line in enumerate(content.split('\n'), 1):
                if f"\\ref{{{r}}}" in line or f"\\pageref{{{r}}}" in line:
                    print(f"  - '{r}' referenced at line {i}: {line[:100]}")
    else:
        print("No undefined references found!")

import sys

def find_labels_and_refs(filepath):
    print(f"\nScanning {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print("  File not found.")
        return

    # Find all labels: \label{name}
    labels = set(re.findall(r'\\label\{([^}]+)\}', content))
    
    # Find all references: \ref{name} and \pageref{name} and \cite{name}
    refs = re.findall(r'\\ref\{([^}]+)\}', content)
    pagerefs = re.findall(r'\\pageref\{([^}]+)\}', content)
    
    # Check undefined
    undefined_refs = []
    for r in refs + pagerefs:
        if r not in labels:
            undefined_refs.append(r)
            
    print(f"  Defined labels count: {len(labels)}")
    print(f"  Undefined references count: {len(undefined_refs)}")
    if undefined_refs:
        print("  Undefined references:")
        for r in sorted(set(undefined_refs)):
            # find line number
            for i, line in enumerate(content.split('\n'), 1):
                if f"\\ref{{{r}}}" in line or f"\\pageref{{{r}}}" in line:
                    print(f"    - '{r}' referenced at line {i}: {line[:100]}")

if __name__ == '__main__':
    files = sys.argv[1:] if len(sys.argv) > 1 else [
        '/data/brhanu/thesis_project/thesis_v3_updated.tex',
        '/data/brhanu/thesis_project/advisor_walkthrough_handout.tex',
        '/data/brhanu/thesis_project/thesis_progress_report_2026_05_15.tex'
    ]
    for filepath in files:
        find_labels_and_refs(filepath)
