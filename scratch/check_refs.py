import re
import sys

def check_latex_refs(filepath):
    print(f"Checking references in {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all labels: \label{...}
    # We should ignore commented lines (lines starting with %)
    # But let's process line by line to handle comments properly.
    labels = set()
    refs = []
    
    # regex pattern for \label{...} and \ref{...}
    label_pattern = re.compile(r'\\label\{([^}]+)\}')
    ref_patterns = [
        re.compile(r'\\ref\{([^}]+)\}'),
        re.compile(r'\\cref\{([^}]+)\}'),
        re.compile(r'\\Cref\{([^}]+)\}'),
        re.compile(r'\\autoref\{([^}]+)\}')
    ]
    
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        # Strip comments
        comment_idx = line.find('%')
        if comment_idx != -1:
            escaped = False
            if comment_idx > 0 and line[comment_idx - 1] == '\\':
                escaped = True
            if not escaped:
                line = line[:comment_idx]
        
        # Find labels
        for lbl in label_pattern.findall(line):
            labels.add(lbl)
            
        # Find refs
        for pattern in ref_patterns:
            for r in pattern.findall(line):
                # Handle comma separated refs (e.g. \cref{ref1,ref2})
                for single_ref in r.split(','):
                    refs.append((single_ref.strip(), i, line.strip()))

    print(f"Found {len(labels)} labels and {len(refs)} references.")
    
    undefined = []
    for ref, line_num, line_text in refs:
        if ref not in labels:
            undefined.append((ref, line_num, line_text))
            
    if undefined:
        print(f"❌ Found {len(undefined)} undefined references:")
        for ref, line_num, line_text in undefined:
            print(f"  Line {line_num}: ref '{ref}' is undefined.\n    Context: {line_text}")
    else:
        print("✅ No undefined references found!")

if __name__ == '__main__':
    files = ["thesis_v3_updated.tex", "advisor_walkthrough_handout.tex"]
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    for filepath in files:
        check_latex_refs(filepath)
