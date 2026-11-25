import re
from pathlib import Path

tex_files = list(Path('presentation').glob('*.tex'))
print(f"Checking {len(tex_files)} TeX files in presentation/:\n")

for tf in sorted(tex_files):
    text = tf.read_text(encoding='utf-8', errors='ignore')
    
    begins = re.findall(r'\\begin\{([^}]+)\}', text)
    ends   = re.findall(r'\\end\{([^}]+)\}', text)
    
    b_counts, e_counts = {}, {}
    for b in begins: b_counts[b] = b_counts.get(b, 0) + 1
    for e in ends:   e_counts[e] = e_counts.get(e, 0) + 1
    
    mismatches = []
    for k in set(list(b_counts.keys()) + list(e_counts.keys())):
        bc = b_counts.get(k, 0)
        ec = e_counts.get(k, 0)
        if bc != ec:
            mismatches.append(f"{k}: begin={bc}, end={ec}")
            
    labels = set(re.findall(r'\\label\{([^}]+)\}', text))
    refs   = set(re.findall(r'\\ref\{([^}]+)\}', text) + re.findall(r'\\autoref\{([^}]+)\}', text))
    
    missing_refs = [r for r in refs if r not in labels]
    
    print(f"📄 {tf.name}:")
    print(f"   Lines: {len(text.splitlines())} | Labels: {len(labels)} | Refs: {len(refs)}")
    if mismatches:
        print("   ⚠️ Environment mismatches:", mismatches)
    else:
        print("   ✅ All \\begin/\\end environments balanced")
        
    if missing_refs:
        print(f"   ℹ️ External/Unresolved refs ({len(missing_refs)}):", missing_refs[:5])
    else:
        print("   ✅ All \\ref labels resolved internally")
    print()
