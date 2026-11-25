import re

with open("/data/brhanu/thesis_project/presentation/thesis_v3_updated.tex", "r") as f:
    tex = f.read()

with open("/data/brhanu/thesis_project/references.bib", "r") as f:
    bib = f.read()

cites = set()
for c in re.findall(r'\\cite\{([^}]+)\}', tex):
    for key in c.split(','):
        cites.add(key.strip())

all_keys = set(re.findall(r'@[a-zA-Z]+\s*\{\s*([^,\s]+),', bib))

uncited = all_keys - cites
print(f"Total keys in bib: {len(all_keys)}")
print(f"Cited keys in TeX: {len(cites)}")
print(f"Uncited keys ({len(uncited)}): {sorted(list(uncited))}")
