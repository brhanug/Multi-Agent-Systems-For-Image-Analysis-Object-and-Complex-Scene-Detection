from pathlib import Path

BASE_DIR = Path("/data/brhanu/thesis_project")

files_to_update = [
    BASE_DIR / "thesis_v3_updated.tex",
    BASE_DIR / "presentation/defense_slides.tex",
    BASE_DIR / "thesis_progress_report_2026_05_15.tex"
]

def update_file(file_path):
    print(f"Updating file precisely: {file_path}")
    if not file_path.exists():
        print(f"⚠️ File not found: {file_path}")
        return
        
    content = file_path.read_text(encoding='utf-8')
    updated = content
    
    # 1. Global replacements for unique numbers
    global_replacements = {
        "0.1316": "0.1462",
        "13.2%": "14.6%",
        "0.0570": "0.0639",
        "5.7%": "6.4%",
        "-0.8574": "-0.8428",
        
        # 200 to 300 images
        "200-Image Human-Verified": "300-Image Human-Verified",
        "200 real historic images": "300 real historic images",
        "200 historical archival images": "300 historical archival images",
        "153 of which contain objects and 47 are background-only": "244 of which contain objects and 56 are background-only",
        
        # Multi-class Kappa
        "0.1555": "0.2334",
    }
    
    for old, new in global_replacements.items():
        updated = updated.replace(old, new)
        
    # 2. Context-specific replacements for scene Kappas in thesis_v3_updated.tex
    context_replacements = {
        "\\item \\textbf{Landscape}: \\textbf{0.4971}": "\\item \\textbf{Landscape}: \\textbf{0.5667}",
        "\\item \\textbf{Family}: \\textbf{0.1853}": "\\item \\textbf{Family}: \\textbf{0.1174}",
        "\\item \\textbf{Drawing}: \\textbf{0.0415}": "\\item \\textbf{Drawing}: \\textbf{0.0855}",
        "\\item \\textbf{Teaching}: \\textbf{0.0000}": "\\item \\textbf{Teaching}: \\textbf{-0.0060}",
        "\\item \\textbf{Playing}: \\textbf{-0.0369}": "\\item \\textbf{Playing}: \\textbf{0.1084}",
        
        "\\item \\textbf{Person}: \\textbf{0.8245}": "\\item \\textbf{Person}: \\textbf{0.8201}",
        "\\item \\textbf{Building}: \\textbf{0.8190}": "\\item \\textbf{Building}: \\textbf{0.8666}",
        "\\item \\textbf{Horse}: \\textbf{0.7876}": "\\item \\textbf{Horse}: \\textbf{0.7326}",
        "\\item \\textbf{Child}: \\textbf{0.7812}": "\\item \\textbf{Child}: \\textbf{0.7932}",
        "\\item \\textbf{Mean Object Agreement}: \\textbf{0.8031}": "\\item \\textbf{Mean Object Agreement}: \\textbf{0.7893}",
        
        # Ambiguity Correlations
        "\\item \\textbf{Pearson Correlation Coefficient ($r$)}: \\textbf{-0.2244} ($p = 0.0248$)": "\\item \\textbf{Pearson Correlation Coefficient ($r$)}: \\textbf{-0.0277} ($p = 0.6328$)",
        "\\item \\textbf{Spearman Rank Correlation ($\\rho$)}: \\textbf{-0.2299} ($p = 0.0214$)": "\\item \\textbf{Spearman Rank Correlation ($\\rho$)}: \\textbf{-0.0715} ($p = 0.2169$)",
        
        # Object Agreement Section text
        "spatial object grounding exhibits high objective consistency among human annotators ($\\kappa = 0.8031$)" : "spatial object grounding exhibits high objective consistency among human annotators ($\\kappa = 0.7893$)",
        "high-level semantic scene categorization introduces massive subjective interpretation ($\\kappa = 0.1555$)" : "high-level semantic scene categorization introduces massive subjective interpretation ($\\kappa = 0.2334$)",
        
        # Slides replacements
        "Person & 0.8245": "Person & 0.8201",
        "Child & 0.7812": "Child & 0.7932",
        "Building & 0.8190": "Building & 0.8666",
        "Horse & 0.7876": "Horse & 0.7326",
        "\\textbf{Mean Agreement} & \\textbf{0.8031}": "\\textbf{Mean Agreement} & \\textbf{0.7893}",
        
        # Progress report replacements
        "real human mAP50 of \\textbf{0.1316}": "real human mAP50 of \\textbf{0.1462}",
        "mAP50 = 0.1316 (real human)": "mAP50 = 0.1462 (real human)"
    }
    
    for old, new in context_replacements.items():
        updated = updated.replace(old, new)
        
    if updated != content:
        file_path.write_text(updated, encoding='utf-8')
        print(f"✅ Successfully updated: {file_path.name}")
    else:
        print(f"ℹ️ No changes needed for: {file_path.name}")

def main():
    for f in files_to_update:
        update_file(f)

if __name__ == "__main__":
    main()
