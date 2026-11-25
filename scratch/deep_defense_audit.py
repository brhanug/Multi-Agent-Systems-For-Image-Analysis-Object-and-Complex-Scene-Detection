import re
import os

tex_path = "/data/brhanu/thesis_project/presentation/defense_slides.tex"

with open(tex_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Find all \label{...}
labels = set(re.findall(r'\\label\{([^}]+)\}', text))
print(f"Total defined labels in defense_slides.tex: {len(labels)}")

# 2. Find all \ref{...}, \pageref{...}, \cite{...}
refs = re.findall(r'\\(?:ref|pageref|eqref)\{([^}]+)\}', text)
missing_refs = set()
for r in refs:
    if r not in labels:
        missing_refs.add(r)

print(f"Total references checked in defense_slides: {len(refs)}")
if missing_refs:
    print(f"❌ Missing labels referenced in defense_slides: {missing_refs}")
else:
    print("✅ All \\ref targets exist in defense_slides.")

# 3. Check included figures
graphic_paths = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text)
print(f"Total included figures in defense_slides: {len(graphic_paths)}")
missing_imgs = []
for gp in graphic_paths:
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
    print(f"❌ Missing image files in defense_slides: {missing_imgs}")
else:
    print("✅ All \\includegraphics files in defense_slides exist on disk.")
