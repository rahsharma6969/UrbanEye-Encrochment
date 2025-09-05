# src/stac/search_items_aws.py
import argparse
import json
import yaml
import sys
from pathlib import Path
from datetime import datetime
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

# ✅ Fixed URL (no trailing space)
AWS_STAC = "https://earth-search.aws.element84.com/v1/search"

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def aoi_union(aoi_dir: Path):
    files = list(aoi_dir.glob("*.geojson"))
    if not files:
        sys.exit(f"No AOI files in {aoi_dir}")
    gdfs = []
    for f in files:
        g = gpd.read_file(f)
        g = g[g.geometry.notna()]
        if not g.empty:
            gdfs.append(g.to_crs("EPSG:4326"))
    if not gdfs:
        sys.exit("AOI files contain no valid geometries.")
    merged = pd.concat(gdfs, ignore_index=True).explode(ignore_index=True)
    try:
        u = merged.geometry.union_all()
    except AttributeError:
        u = merged.geometry.unary_union
    return mapping(u)

def search_aws(collections, dt_start, dt_end, intersects, cloud_lt=None, limit=200):
    query = {"eo:cloud_cover": {"lt": cloud_lt}} if cloud_lt is not None else None

    body = {
        "collections": collections,
        "datetime": f"{dt_start}T00:00:00Z/{dt_end}T23:59:59Z",
        "limit": limit,
        "intersects": intersects,
    }
    if query:
        body["query"] = query

    items = []
    while True:
        r = requests.post(AWS_STAC, json=body, timeout=60)
        r.raise_for_status()
        data = r.json()
        feats = data.get("features", [])

        # ✅ Accept all scenes — no .tif filtering
        for it in feats:
            items.append({
                "collection": it.get("collection"),
                "item": it,
            })

        next_link = None
        for L in data.get("links", []):
            if L.get("rel") == "next" and L.get("method", "POST") == "POST":
                next_link = L
                break
        if not next_link:
            break
        body = next_link.get("body", body)
    return items


def main(config, start, end, collections, out, cloud_lt=None):
    cfg = load_cfg(config)
    aoi_dir = Path(cfg["paths"]["aoi_dir"])
    geom = aoi_union(aoi_dir)

    if cloud_lt is None:
        if "stac" in cfg and isinstance(cfg["stac"], dict):
            cloud_lt = cfg["stac"].get("cloud_lt", None)

    items = search_aws(collections, start, end, geom, cloud_lt=cloud_lt)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(items, indent=2))
    print(f"✅ Wrote {len(items)} STAC items → {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Search AWS Earth Search STAC API for Sentinel-2")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    ap.add_argument("--collections", nargs="+", required=True, help="STAC collections")
    ap.add_argument("--out", required=True, help="Output JSON file")
    ap.add_argument("--cloud_lt", type=int, help="Max cloud cover % (e.g., 30)")
    a = ap.parse_args()
    main(a.config, a.start, a.end, a.collections, a.out, cloud_lt=a.cloud_lt)
    
'''  
python -m src.stac.search_items_aws ^
  --config configs/config.yaml ^
  --start 2022-01-01 ^
  --end 2023-12-31 ^
  --collections sentinel-2-l2a ^
  --out outputs/stac_items_mumbai.json ^
  --cloud_lt 30

  
  '''
