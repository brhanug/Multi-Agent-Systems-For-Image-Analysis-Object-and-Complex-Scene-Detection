import re
from pathlib import Path

BASE_DIR = Path("/data/brhanu/thesis_project")

replacements = {
    # Real human mAP metrics
    "0.1316": "0.1462",
    "13.2%": "14.6%",
    "0.0570": "0.0639",
    "5.7%": "6.4%",
    "-0.8574": "-0.8428",
    
    # 200-image to 300-image text updates
    "200-Image Human-Verified": "300-Image Human-Verified",
    "200 real historic images": "300 real historic images",
    "200 historical archival images": "300 historical archival images",
    "153 of which contain objects and 47 are background-only": "244 of which contain objects and 56 are background-only",
    
    # Scene Kappas
    "0.1555": "0.2334",
    "0.4971": "0.5667",
    "0.1853": "0.1174",
    "0.0415": "0.0855",
    "0.0000": "-0.0060",
    "-0.0369": "0.1084",
    
    # Object Kappas
    "0.8245": "0.8201",
    "0.8190": "0.8666",
    "0.7876": "0.7326",
    "0.7812": "0.7932",
    "0.8031": "0.7893",
    
    # Ambiguity Correlations
    "-0.2244": "-0.0277",
    "0.0248": "0.6328",
    "-0.2299": "-0.0715",
    "0.0214": "0.2169",
}

def update_file(file_path):
    print(f"Updating file: {file_path}")
    if not file_path.exists():
        print(f"⚠️ File not found: {file_path}")
        return
        
    content = file_path.read_text(encoding='utf-8')
    
    # Apply replacements
    updated = content
    for old, new in replacements.items():
        # Avoid breaking latex math signs by matching carefully
        updated = updated.replace(old, new)
        
    if updated != content:
        file_path.write_text(updated, encoding='utf-8')
        print(f"✅ Successfully updated: {file_path.name}")
    else:
        print(f"ℹ️ No changes needed for: {file_path.name}")

def main():
    files = [
        BASE_DIR / "thesis_v3_updated.tex",
        BASE_DIR / "presentation/defense_slides.tex",
        BASE_DIR / "thesis_progress_report_2026_05_15.tex"
    ]
    for f in files:
        update_file(f)

if __name__ == "__main__":
    main()
