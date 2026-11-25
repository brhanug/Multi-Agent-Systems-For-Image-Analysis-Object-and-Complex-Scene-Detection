#!/usr/bin/env python3
"""
run_europeana_real_pilot_download.py
--------------------------------------
Genuine Europeana Open Culture API download for the cross-institutional
transfer pilot. Replaces the two prior "Europeana" attempts, both of which
were found during a full-project reproducibility audit to not constitute
real cross-institutional evidence:
  - run_external_archive_validation.py (n=85): downloaded real Europeana
    images, but layered fabricated VLM/scene confidence scores on top via
    np.random.uniform() -- no real VLM or SigLIP inference.
  - exp23_europeana_annotation.py (n=30): used NO Europeana data at all --
    sampled ordinary Colibri images (same PPN archive, verified against
    final_dataset/metadata/manifest.csv) and mislabeled them "Europeana."

This script does ONLY the download step, honestly: real Europeana Search
API v2 calls (public demo key, no registration required), real image
downloads, real per-image institutional metadata. No agent scores, no
ground truth, no simulated data of any kind. Downstream real Tier 1
inference (YOLO+SAM, SigLIP, Kosmos-2.5, LLaVA-OneVision) is a separate
script; real human ground-truth annotation must be done by a person and
cannot be generated here.

Outputs:
  results/external_archive/europeana_real_pilot/images/*.jpg
  results/external_archive/europeana_real_pilot/metadata.csv
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
OUT_DIR = BASE / "results" / "external_archive" / "europeana_real_pilot"
IMAGE_DIR = OUT_DIR / "images"
OUTPUT_METADATA = OUT_DIR / "metadata.csv"

EUROPEANA_API_KEY = "api2demo"  # public demo key, no registration required
TARGET_IMAGES = 40  # aim above 30 to allow for download failures / dedup

# Diverse queries chosen to span multiple institutions and match Colibri's
# domain (19th/early-20th-century illustrated books, educational/family/
# historical scenes) without constraining to any single provider.
SEARCH_QUERIES = [
    "children illustration 19th century",
    "historical family scene drawing",
    "school classroom historical illustration",
    "landscape historical engraving",
    "children playing drawing historical",
    "historical portrait engraving 19th century",
    "historical manuscript illustration Europe",
    "children reading book illustration",
    "historical village scene illustration",
    "antique botanical illustration engraving",
]
IMAGES_PER_QUERY = max(1, TARGET_IMAGES // len(SEARCH_QUERIES))


def fetch_europeana_page(query: str, rows: int = 50) -> list:
    url = (
        "https://api.europeana.eu/record/v2/search.json"
        f"?wskey={EUROPEANA_API_KEY}"
        f"&query={urllib.parse.quote(query)}"
        "&qf=TYPE:IMAGE"
        f"&rows={rows}"
        "&start=1"
        "&profile=minimal"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("items", [])
    except Exception as e:
        print(f"  ⚠ Europeana fetch failed for query '{query}': {e}")
        return []


def download_image(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        headers = {"User-Agent": "Mozilla/5.0 (DCV-MAC Research; academic use)"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        if len(data) < 1000:
            return False
        with open(dest, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    seen_ids = set()
    total_downloaded = 0

    print("=" * 70)
    print("REAL Europeana Open Culture API Download")
    print("=" * 70)

    for query in SEARCH_QUERIES:
        print(f"\n🔍 Query: '{query}'")
        items = fetch_europeana_page(query, rows=min(IMAGES_PER_QUERY * 4, 100))
        query_count = 0

        for item in items:
            if query_count >= IMAGES_PER_QUERY:
                break

            preview_urls = item.get("edmPreview", [])
            if not preview_urls:
                continue
            img_url = preview_urls[0]

            item_id = item.get("id", "").replace("/", "_").strip("_")
            if not item_id or item_id in seen_ids:
                continue

            img_path = IMAGE_DIR / f"{item_id}.jpg"
            if not download_image(img_url, img_path):
                continue

            seen_ids.add(item_id)
            records.append({
                "image_id": item_id,
                "image_path": str(img_path),
                "title": (item.get("title") or ["Unknown"])[0][:150],
                "data_provider": (item.get("dataProvider") or ["Unknown"])[0][:100],
                "country": (item.get("country") or ["Unknown"])[0],
                "year": (item.get("year") or ["Unknown"])[0],
                "rights": (item.get("rights") or ["Unknown"])[0][:100],
                "preview_url": img_url,
                "query": query,
                "europeana_record_id": item.get("id", ""),
            })
            query_count += 1
            total_downloaded += 1
            time.sleep(0.15)

        print(f"  ✅ Got {query_count} images for this query")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_METADATA, index=False)

    print(f"\n📦 Total REAL Europeana images downloaded: {total_downloaded}")
    print(f"   Unique data providers: {df['data_provider'].nunique() if len(df) else 0}")
    print(f"   Unique countries: {df['country'].nunique() if len(df) else 0}")
    print(f"✅ Metadata saved to {OUTPUT_METADATA}")


if __name__ == "__main__":
    main()
