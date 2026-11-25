import os
import re

def get_inputs(content, base_dir):
    inputs = re.findall(r'\\input\{([^}]+)\}', content)
    resolved_inputs = []
    for inp in inputs:
        if not inp.endswith('.tex'):
            inp += '.tex'
        full_path = os.path.join(base_dir, inp)
        if os.path.exists(full_path):
            resolved_inputs.append(full_path)
        else:
            if os.path.exists(inp):
                resolved_inputs.append(inp)
    return resolved_inputs

def extract_labels(filepath, base_dir, visited=None):
    if visited is None:
        visited = set()
    if filepath in visited:
        return {}
    visited.add(filepath)
    
    if not os.path.exists(filepath):
        return {}
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    lines = content.splitlines()
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
    
    labels_found = re.findall(r'\\label\{([^}]+)\}', clean_content)
    labels_map = {lbl: filepath for lbl in labels_found}
    
    inputs = get_inputs(clean_content, base_dir)
    for inp in inputs:
        labels_map.update(extract_labels(inp, os.path.dirname(inp), visited))
        
    return labels_map

def run():
    filepath = 'presentation/advisor_walkthrough_handout.tex'
    labels_map = extract_labels(filepath, os.path.dirname(filepath))
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Clean comments
    lines = content.splitlines()
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
    
    refs = re.findall(r'\\(?:ref|cref|Cref|autoref)\{([^}]+)\}', clean_content)
    unique_refs = set()
    for ref_str in refs:
        for part in ref_str.split(','):
            unique_refs.add(part.strip())
            
    print(f"Total unique references used in thesis: {len(unique_refs)}")
    print("Reference -> Defined In mapping:")
    for ref in sorted(unique_refs):
        if ref in labels_map:
            print(f"  {ref:35s} -> {os.path.basename(labels_map[ref])}")
        else:
            print(f"  {ref:35s} -> NOT DEFINED LOCALLY!")

if __name__ == '__main__':
    run()
