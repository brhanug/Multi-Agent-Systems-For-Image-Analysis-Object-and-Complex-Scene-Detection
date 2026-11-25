import re

def check_exact():
    with open('/data/brhanu/thesis_project/presentation/advisor_walkthrough_handout.tex', 'r', encoding='utf-8') as f:
        content = f.read()

    # Find raw text inside \label{...} and \ref{...}
    labels = re.findall(r'\\label\{([^}]+)\}', content)
    refs = []
    
    ref_patterns = [
        r'\\ref\{([^}]+)\}',
        r'\\cref\{([^}]+)\}',
        r'\\Cref\{([^}]+)\}',
        r'\\pageref\{([^}]+)\}'
    ]
    for pattern in ref_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            # Split by comma for multi-refs
            for key in match.split(','):
                refs.append(key)

    print("Defined labels (exact strings):")
    for l in sorted(set(labels)):
        print(f"  '{l}'")
        
    print("\nUsed references (exact strings):")
    for r in sorted(set(refs)):
        print(f"  '{r}'")

    # Check for exact matches
    undefined = []
    for r in refs:
        if r not in labels:
            undefined.append(r)
            
    print("\nUndefined references without any stripping:")
    if undefined:
        for u in sorted(set(undefined)):
            print(f"  '{u}'")
    else:
        print("  None!")

if __name__ == '__main__':
    check_exact()
