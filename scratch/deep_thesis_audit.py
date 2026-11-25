import re
import os

tex_path = "/data/brhanu/thesis_project/presentation/thesis_v3_updated.tex"

with open(tex_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Find all \label{...}
labels = set(re.findall(r'\\label\{([^}]+)\}', text))
print(f"Total defined labels: {len(labels)}")

# 2. Find all \ref{...}, \pageref{...}, \cite{...}
refs = re.findall(r'\\(?:ref|pageref|eqref)\{([^}]+)\}', text)
missing_refs = set()
for r in refs:
    if r not in labels:
        missing_refs.add(r)

print(f"Total references checked: {len(refs)}")
if missing_refs:
    print(f"❌ Missing labels referenced in text: {missing_refs}")
else:
    print("✅ All \\ref and \\eqref targets exist.")

# 3. Check citations vs references.bib
bib_path = "/data/brhanu/thesis_project/references.bib"
if os.path.exists(bib_path):
    with open(bib_path, 'r', encoding='utf-8') as f:
        bib_text = f.read()
    bib_keys = set(re.findall(r'@[a-zA-Z]+\s*\{\s*([^,\s]+),', bib_text))
    print(f"Total keys in references.bib: {len(bib_keys)}")
    
    cites = set()
    for raw_cite in re.findall(r'\\cite\{([^}]+)\}', text):
        for c in raw_cite.split(','):
            cites.add(c.strip())
            
    missing_cites = cites - bib_keys
    print(f"Total unique citations in TeX: {len(cites)}")
    if missing_cites:
        print(f"❌ Missing citations in references.bib: {missing_cites}")
    else:
        print("✅ All \\cite keys exist in references.bib.")
else:
    print(f"⚠️ references.bib not found at {bib_path}")

# 4. Check figures included
graphic_paths = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text)
print(f"Total included figures: {len(graphic_paths)}")
missing_imgs = []
for gp in graphic_paths:
    # check possible paths
    search_paths = [
        os.path.join("/data/brhanu/thesis_project/presentation", gp),
        os.path.join("/data/brhanu/thesis_project/latex_assets", gp),
        os.path.join("/data/brhanu/thesis_project/presentation/images", gp),
        os.path.join("/data/brhanu/thesis_project", gp)
    ]
    found = any(os.path.exists(sp) for sp in search_paths)
    if not found:
        missing_imgs.append(gp)

if missing_imgs:
    print(f"❌ Missing image files: {missing_imgs}")
else:
    print("✅ All \\includegraphics files exist on disk.")

