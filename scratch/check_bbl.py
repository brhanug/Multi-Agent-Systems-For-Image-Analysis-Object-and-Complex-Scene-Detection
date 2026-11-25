import re
import os

with open("/data/brhanu/thesis_project/presentation/thesis_v3_updated.tex", "r") as f:
    tex = f.read()

cites = re.findall(r'\\cite\{([^}]+)\}', tex)
print(f"Total \\cite instances in thesis_v3_updated.tex: {len(cites)}")

# Check if bbl file was generated
bbl_path = "/data/brhanu/thesis_project/presentation/thesis_v3_updated.bbl"
if os.path.exists(bbl_path):
    with open(bbl_path, "r") as f:
        bbl_content = f.read()
    bitem_count = len(re.findall(r'\\bibitem', bbl_content))
    print(f"✅ Bibliography file generated successfully with {bitem_count} references!")
else:
    print("❌ thesis_v3_updated.bbl NOT found!")
