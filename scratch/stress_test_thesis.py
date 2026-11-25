import re

tex_path = "/data/brhanu/thesis_project/presentation/thesis_v3_updated.tex"

with open(tex_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

text = "".join(lines)

print("=== 1. CHECK UNREFERENCED FIGURES AND TABLES ===")
fig_labels = re.findall(r'\\begin\{figure\}.*?\\label\{([^}]+)\}.*?\\end\{figure\}', text, re.DOTALL)
tbl_labels = re.findall(r'\\begin\{table\}.*?\\label\{([^}]+)\}.*?\\end\{table\}', text, re.DOTALL)

all_labels = fig_labels + tbl_labels

for lbl in all_labels:
    # check if \ref{lbl} or \pageref{lbl} exists in text outside the label definition itself
    pattern = r'\\(?:ref|pageref|eqref)\{' + re.escape(lbl) + r'\}'
    matches = re.findall(pattern, text)
    if not matches:
        print(f"⚠️ UNREFERENCED FLOAT LABEL: '{lbl}' is defined but NEVER referenced with \\ref in the body text!")

print("\n=== 2. CHECK SAMPLE SIZE INCONSISTENCIES ===")
# Search for Europeana n values
europeana_n = re.findall(r'Europeana.*?(?:n\s*=\s*|sample\s*of\s*)(\d+)', text, re.IGNORECASE)
print(f"Europeana sample sizes mentioned: {europeana_n}")

# Search for Gold cohort n values
gold_n = re.findall(r'(?:Gold|Expert).*?(?:n\s*=\s*)(\d+)', text, re.IGNORECASE)
print(f"Gold cohort sample sizes mentioned: {set(gold_n)}")

# Search for Spatial Audit n values
spatial_n = re.findall(r'(?:Spatial Audit|annotator).*?(?:n\s*=\s*)(\d+)', text, re.IGNORECASE)
print(f"Spatial Audit sample sizes mentioned: {set(spatial_n)}")

print("\n=== 3. CHECK MODEL NAMES AND VERSION INCONSISTENCIES ===")
yolo_versions = set(re.findall(r'YOLOv?\d+', text, re.IGNORECASE))
print(f"YOLO model versions mentioned: {yolo_versions}")

llava_versions = set(re.findall(r'LLaVA[^\s\),]*', text, re.IGNORECASE))
print(f"LLaVA model variants mentioned: {llava_versions}")

system_names = set(re.findall(r'(?:DCV-MAC|Visual Historian-MAS|SAA)', text))
print(f"System names mentioned: {system_names}")

print("\n=== 4. CHECK LATEX ENVIRONMENTS AND MATH SYNTAX ===")
open_envs = re.findall(r'\\begin\{([^}]+)\}', text)
close_envs = re.findall(r'\\end\{([^}]+)\}', text)
if len(open_envs) != len(close_envs):
    print(f"⚠️ Mismatch in LaTeX environment counts: {len(open_envs)} begin vs {len(close_envs)} end")

print("\n=== 5. CHECK OVERFULL HBOX / WIDE TABLES ===")
for i, line in enumerate(lines):
    if "\\begin{tabular}" in line and "\\resizebox" not in lines[max(0, i-2)]:
        print(f"Line {i+1}: Tabular without resizebox (potential overfull hbox): {line.strip()[:60]}")
