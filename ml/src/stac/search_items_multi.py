# src/stac/search_items_multi.py
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
from typing import List, Dict, Any, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Multiple STAC endpoints for redundancy
STAC_ENDPOINTS = {
    "aws": {
        "url": "https://earth-search.aws.element84.com/v1/search",
        "priority": 1,
        "timeout": 60,
        "retry_attempts": 3
    },
    "usgs": {
        "url": "https://landsatlook.usgs.gov/stac-server/search",
        "priority": 3,
        "timeout": 30,
        "retry_attempts": 2
    }
}

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
    if u is None or u.is_empty:
        sys.exit("❌ AOI union resulted in empty geometry.")
    return mapping(u)

def search_single_endpoint(endpoint_name: str, endpoint_config: Dict[str, Any], 
                          collections: List[str], dt_start: str, dt_end: str, 
                          intersects: Dict[str, Any], cloud_lt: Optional[int] = None, 
                          limit: int = 200) -> List[Dict[str, Any]]:
    """Search a single STAC endpoint with retry logic"""
    
    # Skip USGS for Sentinel-2
    if endpoint_name == "usgs" and any("sentinel-2" in c for c in collections):
        logger.warning(f"⚠️ USGS does not support Sentinel-2. Skipping.")
        return []
    
    query = {"eo:cloud_cover": {"lt": cloud_lt}} if cloud_lt is not None else None
    
    body = {
        "collections": collections,
        "datetime": f"{dt_start}T00:00:00Z/{dt_end}T23:59:59Z",
        "limit": limit,
        "intersects": intersects,
    }
    if query:
        body["query"] = query
    
    headers = endpoint_config.get("headers", {})
    timeout = endpoint_config.get("timeout", 60)
    retry_attempts = endpoint_config.get("retry_attempts", 3)
    
    items = []
    
    for attempt in range(retry_attempts):
        try:
            logger.info(f"🔍 Searching {endpoint_name} (attempt {attempt + 1}/{retry_attempts})")
            
            while True:
                r = requests.post(
                    endpoint_config["url"], 
                    json=body, 
                    timeout=(10, timeout),  # connect, read
                    headers=headers
                )
                r.raise_for_status()
                data = r.json()
                feats = data.get("features", [])
                
                for it in feats:
                    items.append({
                        "collection": it.get("collection"),
                        "item": it,
                        "source_endpoint": endpoint_name
                    })
                
                next_link = None
                for L in data.get("links", []):
                    if L.get("rel") == "next" and L.get("method", "POST") == "POST":
                        next_link = L
                        break
                if not next_link:
                    break
                
                # Update body for pagination
                new_body = next_link.get("body", {})
                # Preserve critical keys if missing
                if "collections" not in new_body:
                    new_body["collections"] = collections
                if "datetime" not in new_body:
                    new_body["datetime"] = body["datetime"]
                if "intersects" not in new_body and intersects:
                    new_body["intersects"] = intersects
                if query and "query" not in new_body:
                    new_body["query"] = query
                body = new_body
            
            logger.info(f"✅ {endpoint_name}: Found {len(items)} items")
            return items
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ {endpoint_name} attempt {attempt + 1} failed: {e}")
            if attempt < retry_attempts - 1:
                wait_time = 2 ** attempt
                logger.info(f"⏳ Retrying {endpoint_name} in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ {endpoint_name} failed after {retry_attempts} attempts")
                return []
        
        except Exception as e:
            logger.error(f"❌ Unexpected error with {endpoint_name}: {e}")
            return []
    
    return []

def search_multi_source(collections: List[str], dt_start: str, dt_end: str, 
                       intersects: Dict[str, Any], cloud_lt: Optional[int] = None, 
                       limit: int = 200, parallel: bool = True) -> List[Dict[str, Any]]:
    """Search multiple STAC endpoints in parallel or sequentially"""
    
    all_items = []
    successful_sources = []
    
    if parallel:
        with ThreadPoolExecutor(max_workers=len(STAC_ENDPOINTS)) as executor:
            future_to_source = {}
            for source_name, config in STAC_ENDPOINTS.items():
                future = executor.submit(
                    search_single_endpoint, 
                    source_name, config, collections, dt_start, dt_end, 
                    intersects, cloud_lt, limit
                )
                future_to_source[future] = source_name
            
            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    items = future.result()
                    if items:
                        all_items.extend(items)
                        successful_sources.append(source_name)
                except Exception as e:
                    logger.error(f"❌ Error processing {source_name}: {e}")
    
    else:
        sorted_endpoints = sorted(STAC_ENDPOINTS.items(), key=lambda x: x[1]["priority"])
        for source_name, config in sorted_endpoints:
            items = search_single_endpoint(
                source_name, config, collections, dt_start, dt_end, 
                intersects, cloud_lt, limit
            )
            if items:
                all_items.extend(items)
                successful_sources.append(source_name)
    
    # Deduplicate by item ID, keep first occurrence (or add priority logic if needed)
    seen_ids = set()
    unique_items = []
    for item in all_items:
        item_id = item["item"].get("id")
        if item_id and item_id not in seen_ids:
            seen_ids.add(item_id)
            unique_items.append(item)
    
    logger.info(f"🎯 Total unique items: {len(unique_items)} from sources: {successful_sources}")
    
    return unique_items

def main(config: str, start: str, end: str, collections: List[str], 
         out: str, cloud_lt: Optional[int] = None, parallel: bool = True,
         sources: Optional[List[str]] = None):
    
    cfg = load_cfg(config)
    aoi_dir = Path(cfg["paths"]["aoi_dir"])
    geom = aoi_union(aoi_dir)
    
    if cloud_lt is None:
        if "stac" in cfg and isinstance(cfg["stac"], dict):
            cloud_lt = cfg["stac"].get("cloud_lt", None)
    
    global STAC_ENDPOINTS
    if sources:
        STAC_ENDPOINTS = {k: v for k, v in STAC_ENDPOINTS.items() if k in sources}
        logger.info(f"🔧 Using specific sources: {list(STAC_ENDPOINTS.keys())}")
    
    logger.info(f"🌍 Searching {collections} from {start} to {end}")
    logger.info(f"☁️ Cloud cover limit: {cloud_lt}%" if cloud_lt else "☁️ No cloud cover limit")
    
    items = search_multi_source(
        collections, start, end, geom, 
        cloud_lt=cloud_lt, parallel=parallel
    )
    
    if not items:
        logger.error("❌ No items found from any source!")
        sys.exit(1)
    
    output_data = {
        "search_metadata": {
            "collections": collections,
            "datetime_range": f"{start}/{end}",
            "cloud_cover_limit": cloud_lt,
            "sources_used": list(set(item["source_endpoint"] for item in items)),
            "total_items": len(items),
            "search_timestamp": datetime.now().isoformat()
        },
        "items": items
    }
    
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(output_data, indent=2))
    logger.info(f"✅ Wrote {len(items)} STAC items → {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Multi-source STAC search for Sentinel-2")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    ap.add_argument("--collections", nargs="+", required=True, help="STAC collections")
    ap.add_argument("--out", required=True, help="Output JSON file")
    ap.add_argument("--cloud_lt", type=int, help="Max cloud cover % (e.g., 30)")
    ap.add_argument("--parallel", action="store_true", default=True, help="Search sources in parallel")
    ap.add_argument("--sequential", dest="parallel", action="store_false", help="Search sources sequentially")
    ap.add_argument("--sources", nargs="+", choices=["aws", "usgs"], 
                help="Specific sources to use (default: all)")
    
    args = ap.parse_args()
    main(args.config, args.start, args.end, args.collections, args.out, 
         cloud_lt=args.cloud_lt, parallel=args.parallel, sources=args.sources)

'''
Usage examples:

# Search all sources in parallel (default)
python -m src.stac.search_items_multi ^
  --config configs/config.yaml ^
  --start 2022-01-01 ^
  --end 2023-12-31 ^
  --collections sentinel-2-l2a ^
  --out outputs/stac_items_mumbai_multi.json ^
  --cloud_lt 30




python -m src.stac.search_items_multi ^
  --config configs/config.yaml ^
  --start 2023-01-01 ^
  --end 2023-12-31 ^
  --collections sentinel-2-l2a ^
  --out data/stac/s2_items.json ^
  --cloud_lt 15 ^
  --parallel
  
  
  
'''
    

