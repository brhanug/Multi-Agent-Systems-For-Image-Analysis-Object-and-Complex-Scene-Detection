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
    return resolved_inputs

def extract_labels_with_context(filepath, base_dir, visited=None):
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
    
    labels_map = {}
    current_caption = "Generic context"
    
    in_table_or_fig = False
    current_type = None
    
    for i, line in enumerate(clean_lines, 1):
        line_strip = line.strip()
        
        # Section detection
        sec_match = re.search(r'\\(section|subsection|subsubsection)\*?\{([^}]+)\}', line_strip)
        if sec_match:
            current_caption = f"{sec_match.group(1)}: {sec_match.group(2)}"
            
        # Table/Figure detection
        if r'\begin{table}' in line_strip or r'\begin{table*}' in line_strip or r'\begin{table[!htbp]}' in line_strip:
            in_table_or_fig = True
            current_type = 'Table'
            current_caption = None
        elif r'\begin{figure}' in line_strip or r'\begin{figure*}' in line_strip or r'\begin{figure}[!htbp]' in line_strip:
            in_table_or_fig = True
            current_type = 'Figure'
            current_caption = None
            
        cap_match = re.search(r'\\caption\{([^}]+)\}', line_strip)
        if cap_match:
            current_caption = f"{current_type or 'Caption'}: {cap_match.group(1)}"
            
        lbl_match = re.search(r'\\label\{([^}]+)\}', line_strip)
        if lbl_match:
            lbl = lbl_match.group(1)
            labels_map[lbl] = current_caption or f"Line {i}: {line_strip}"
            
        if r'\end{table}' in line_strip or r'\end{table*}' in line_strip or r'\end{figure}' in line_strip or r'\end{figure*}' in line_strip:
            in_table_or_fig = False
            current_type = None
            current_caption = "Generic context"
            
    # Resolve inputs recursively
    inputs = get_inputs('\n'.join(clean_lines), base_dir)
    for inp in inputs:
        labels_map.update(extract_labels_with_context(inp, os.path.dirname(inp), visited))
        
    return labels_map

def run():
    filepath = 'presentation/thesis_v3_updated.tex'
    labels_context = extract_labels_with_context(filepath, os.path.dirname(filepath))
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
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
        
    print("Checking for conceptual mismatches in thesis_v3_updated.tex...")
    for i, line in enumerate(clean_lines, 1):
        matches = re.finditer(r'\\(ref|cref|Cref|autoref)\{([^}]+)\}', line)
        for m in matches:
            ref_type = m.group(1)
            ref_str = m.group(2)
            for part in ref_str.split(','):
                part = part.strip()
                target_context = labels_context.get(part, "NOT FOUND")
                # print reference and target context
                # If target_context contains Table and reference text has Figure, or vice versa, flag it!
                line_context = line.strip()
                print(f"Line {i:4d} | Ref: {part:30s} | Target Context: {target_context[:80]} | Line: {line_context[:80]}")

if __name__ == '__main__':
    run()
