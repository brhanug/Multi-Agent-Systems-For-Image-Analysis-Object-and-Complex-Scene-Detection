import re

tex_path = "/data/brhanu/thesis_project/presentation/thesis_v3_updated.tex"

with open(tex_path, 'r', encoding='utf-8') as f:
    text = f.read()

floats = re.findall(r'\\begin\{(figure|table)\}.*?\\label\{([^}]+)\}.*?\\end\{\1\}', text, re.DOTALL)

print(f"Total floats found: {len(floats)}")
for ftype, lbl in floats:
    ref_count = len(re.findall(r'\\(?:ref|pageref)\{' + re.escape(lbl) + r'\}', text))
    status = "✅ OK" if ref_count > 0 else "❌ UNREFERENCED"
    print(f"{ftype.capitalize()} label='{lbl}': references={ref_count} [{status}]")
