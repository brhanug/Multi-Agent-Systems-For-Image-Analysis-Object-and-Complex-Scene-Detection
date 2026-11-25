import re
import os
import sys

def parse_tex_recursive(filepath, base_dir="."):
    # Read the file content
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return "", []

    # Replace comments
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

    # Find \input{...} or \include{...}
    # Pattern: \input{filename} or \input{filename.tex}
    input_pattern = re.compile(r'\\input\{([^}]+)\}')
    include_pattern = re.compile(r'\\include\{([^}]+)\}')
    
    subfiles = []
    for pattern in [input_pattern, include_pattern]:
        for match in pattern.findall(clean_content):
            fname = match.strip()
            if not fname.endswith('.tex'):
                fname += '.tex'
            subfiles.append(fname)
            
    return clean_content, subfiles

def check_project_refs(main_file):
    print(f"\n====================================")
    print(f"Checking references recursively starting from: {main_file}")
    
    visited = set()
    to_visit = [main_file]
    
    all_content = ""
    # We will map reference names to their lines/files for tracking
    refs = []
    labels = set()
    
    label_pattern = re.compile(r'\\label\{([^}]+)\}')
    ref_patterns = [
        re.compile(r'\\ref\{([^}]+)\}'),
        re.compile(r'\\cref\{([^}]+)\}'),
        re.compile(r'\\Cref\{([^}]+)\}'),
        re.compile(r'\\autoref\{([^}]+)\}')
    ]
    
    while to_visit:
        curr = to_visit.pop(0)
        if curr in visited:
            continue
        visited.add(curr)
        
        # Parse current file
        content, subfiles = parse_tex_recursive(curr)
        
        # Scan for labels and refs in this file line by line
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            # Find labels
            for lbl in label_pattern.findall(line):
                labels.add(lbl)
                
            # Find refs
            for pattern in ref_patterns:
                for r in pattern.findall(line):
                    for single_ref in r.split(','):
                        refs.append((single_ref.strip(), curr, i, line.strip()))
                        
        # Queue subfiles
        for sf in subfiles:
            # Try to resolve path
            if os.path.exists(sf):
                to_visit.append(sf)
            else:
                # Try relative to main file directory
                dir_name = os.path.dirname(main_file)
                sf_path = os.path.join(dir_name, sf)
                if os.path.exists(sf_path):
                    to_visit.append(sf_path)
                else:
                    print(f"⚠️ Warning: Could not find input file: {sf}")
                    
    print(f"Processed files: {list(visited)}")
    print(f"Total labels found: {len(labels)}")
    print(f"Total references found: {len(refs)}")
    
    undefined = []
    for ref, file, line_num, line_text in refs:
        if ref not in labels:
            undefined.append((ref, file, line_num, line_text))
            
    if undefined:
        print(f"❌ Found {len(undefined)} undefined references:")
        for ref, file, line_num, line_text in undefined:
            print(f"  {file}:{line_num}: ref '{ref}' is undefined.\n    Context: {line_text}")
    else:
        print("✅ No undefined references found in the project recursive check!")

if __name__ == '__main__':
    check_project_refs("thesis_v3_updated.tex")
    check_project_refs("advisor_walkthrough_handout.tex")
