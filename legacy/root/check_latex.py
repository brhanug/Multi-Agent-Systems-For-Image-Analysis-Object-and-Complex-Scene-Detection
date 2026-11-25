#!/usr/bin/env python3
import re
import sys

def check_latex_file(filepath):
    print(f"🔍 Analyzing LaTeX file: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    errors = []
    stack = []  # To track open braces
    env_stack = []  # To track environments
    labels = {}  # To track duplicate labels
    
    # Regular expressions
    begin_re = re.compile(r'\\begin\{([a-zA-Z0-9\*]+)\}')
    end_re = re.compile(r'\\end\{([a-zA-Z0-9\*]+)\}')
    label_re = re.compile(r'\\label\{([^}]+)\}')
    
    for i, line in enumerate(lines, 1):
        # Strip comments
        comment_idx = line.find('%')
        if comment_idx != -1:
            # Check if % is escaped
            escaped = False
            if comment_idx > 0 and line[comment_idx - 1] == '\\':
                escaped = True
            if not escaped:
                line = line[:comment_idx]
                
        # 1. Check duplicate labels
        for label in label_re.findall(line):
            if label in labels:
                errors.append(f"Line {i}: Duplicate label '{label}' (first defined at line {labels[label]})")
            else:
                labels[label] = i
                
        # 2. Check braces
        for char_idx, char in enumerate(line):
            if char == '{':
                stack.append((i, char_idx))
            elif char == '}':
                if stack:
                    stack.pop()
                else:
                    errors.append(f"Line {i}: Mismatched closing brace '}}' at position {char_idx}")
                    
        # 3. Check environments
        for env in begin_re.findall(line):
            env_stack.append((env, i))
            
        for env in end_re.findall(line):
            if env_stack:
                expected_env, start_line = env_stack.pop()
                if expected_env != env:
                    errors.append(f"Line {i}: Mismatched environment. Found '\\end{{{env}}}', but expected '\\end{{{expected_env}}}' (opened at line {start_line})")
            else:
                errors.append(f"Line {i}: Found '\\end{{{env}}}' without a matching '\\begin'")

    # Remaining open braces
    while stack:
        line_num, char_idx = stack.pop()
        errors.append(f"Line {line_num}: Unclosed open brace '{{'")
        
    # Remaining open environments
    while env_stack:
        env, line_num = env_stack.pop()
        errors.append(f"Line {line_num}: Environment '\\begin{{{env}}}' is never closed")
        
    # Summary of findings
    if errors:
        print(f"❌ Found {len(errors)} errors:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("✅ PASSED: No bracket or environment mismatches found. Zero duplicate labels.")
        return True

if __name__ == "__main__":
    files_to_check = ["thesis_v3_updated.tex", "thesis_progress_report_2026_05_15.tex"]
    if len(sys.argv) > 1:
        files_to_check = sys.argv[1:]
        
    all_passed = True
    for f in files_to_check:
        try:
            if not check_latex_file(f):
                all_passed = False
        except FileNotFoundError:
            print(f"⚠️  File not found: {f}")
            all_passed = False
            
    if not all_passed:
        sys.exit(1)
