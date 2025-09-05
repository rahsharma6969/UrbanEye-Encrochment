# scripts/make_chips_s2_pairs.py
import os
import argparse
import json
import yaml
import sys
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import rasterio
import pystac
from tqdm import tqdm

from src.preprocess.s2_prep import cloud_mask_from_scl, apply_mask

# ---------------- GDAL defaults (set once) ----------------
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.jp2,.json")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "200000000")  # Increased cache
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "6")  # More retries
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

# ---------------- utils ----------------
def load_cfg(p: str):
    p = Path(p)
    if not p.exists():
        sys.exit(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def item_time(it) -> datetime:
    dt_str = it["item"]["properties"]["datetime"].replace("Z", "")
    return datetime.fromisoformat(dt_str)

# Updated band aliases to match your STAC catalog naming
BAND_ALIASES = {
    "B02": ["blue", "B02", "blue-jp2"],
    "B03": ["green", "B03", "green-jp2"], 
    "B04": ["red", "B04", "red-jp2"],
    "B08": ["nir", "B08", "nir-jp2"],
    "B8A": ["nir08", "B8A", "nir08-jp2"],
    "SCL": ["scl", "SCL", "scl-jp2"],
}

def _choose_asset_href(stac_item: pystac.Item, key: str) -> str:
    """Choose the best asset href for a given band key"""
    key_up = key.upper()
    aliases = BAND_ALIASES.get(key_up, [key, key_up, key.lower()])
    
    print(f"  Looking for {key_up}, aliases: {aliases}")
    
    # First try .tif assets (preferred)
    for alias in aliases:
        if alias in stac_item.assets:
            href = stac_item.assets[alias].href.strip()
            if href.lower().endswith((".tif", ".tiff")):
                print(f"  ✅ Found TIF: {alias} -> {href[:100]}...")
                return href
    
    # Then try JP2 assets
    for alias in aliases:
        if alias in stac_item.assets:
            href = stac_item.assets[alias].href.strip()
            if href.lower().endswith(".jp2"):
                print(f"  ✅ Found JP2: {alias} -> {href[:100]}...")
                return href
    
    # Last resort - any matching asset
    for alias in aliases:
        if alias in stac_item.assets:
            href = stac_item.assets[alias].href.strip()
            print(f"  ⚠️ Using fallback: {alias} -> {href[:100]}...")
            return href
    
    available = list(stac_item.assets.keys())
    raise KeyError(f"Asset '{key}' not found. Aliases tried: {aliases}. Available: {available}")

def open_da_pick(item_dict: dict, asset_key_upper: str, max_retries=3):
    """Open data array with retry logic"""
    it = pystac.Item.from_dict(item_dict)
    href = _choose_asset_href(it, asset_key_upper)
    
    for attempt in range(max_retries):
        try:
            print(f"    Opening {asset_key_upper} (attempt {attempt+1}/{max_retries})")
            da = rxr.open_rasterio(href, masked=True, cache=False).squeeze()
            
            # Validate by reading a small sample
            test_val = da.isel(x=slice(0, min(10, da.sizes['x'])), 
                             y=slice(0, min(10, da.sizes['y']))).values
            
            if da.sizes['x'] == 0 or da.sizes['y'] == 0:
                raise ValueError(f"Empty raster for {asset_key_upper}")
                
            print(f"    ✅ Successfully opened {asset_key_upper}: {da.sizes}")
            return da
            
        except Exception as e:
            print(f"    ⚠️ Attempt {attempt+1} failed for {asset_key_upper}: {str(e)[:100]}")
            if attempt == max_retries - 1:
                raise Exception(f"Failed to open {asset_key_upper} after {max_retries} attempts: {e}")
            import time
            time.sleep(1 * (attempt + 1))  # Progressive backoff

def aoi_union(aoi_dir: Path) -> gpd.GeoDataFrame:
    aoi_dir = Path(aoi_dir)
    if not aoi_dir.exists():
        sys.exit(f"AOI directory not found: {aoi_dir}")
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
    u = merged.geometry.union_all() if hasattr(merged.geometry, "union_all") else merged.geometry.unary_union
    return gpd.GeoDataFrame(geometry=[u], crs="EPSG:4326")

def pair_nearest(t0_items, t1_items):
    if not t0_items or not t1_items: 
        return []
    t1t = [(item_time(i1), i1) for i1 in t1_items]
    out = []
    for i0 in t0_items:
        t0t = item_time(i0)
        best = min(t1t, key=lambda x: abs(x[0]-t0t))[1]
        out.append((i0, best))
    return out

# --------- worker ---------
def process_pair(args):
    # Set GDAL env in each worker
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.jp2,.json")
    os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
    os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "200000000")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "6")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

    i, i0, i1, aoi, ts, stride, res, chips_dir = args
    rows = []
    
    try:
        print(f"🔄 Processing pair {i}...")
        
        # Read t1 (reference) bands first
        print(f"  📥 Reading t1 bands...")
        b02_1 = open_da_pick(i1["item"], "B02")
        b03_1 = open_da_pick(i1["item"], "B03") 
        b04_1 = open_da_pick(i1["item"], "B04")
        b08_1 = open_da_pick(i1["item"], "B08")
        scl1 = open_da_pick(i1["item"], "SCL")
        
        # Reproject SCL to match B02 resolution
        scl1 = scl1.rio.reproject_match(b02_1, resampling=rasterio.enums.Resampling.nearest)
        
        print(f"  📥 Reading t0 bands...")
        # Read t0 bands and reproject to match t1
        b02_0 = open_da_pick(i0["item"], "B02").rio.reproject_match(b02_1)
        b03_0 = open_da_pick(i0["item"], "B03").rio.reproject_match(b03_1) 
        b04_0 = open_da_pick(i0["item"], "B04").rio.reproject_match(b04_1)
        b08_0 = open_da_pick(i0["item"], "B08").rio.reproject_match(b08_1)
        scl0 = open_da_pick(i0["item"], "SCL").rio.reproject_match(b02_1, resampling=rasterio.enums.Resampling.nearest)

        # Clip to AOI
        print(f"  ✂️ Clipping to AOI...")
        aoi_ref = aoi.to_crs(scl1.rio.crs)
        
        def safe_clip(*bands):
            clipped = []
            for b in bands:
                try:
                    clipped_b = b.rio.clip(aoi_ref.geometry, aoi_ref.crs, drop=True, from_disk=True)
                    if clipped_b.sizes['x'] == 0 or clipped_b.sizes['y'] == 0:
                        raise ValueError("Empty after clipping")
                    clipped.append(clipped_b)
                except Exception as e:
                    print(f"    ⚠️ Clipping failed, using original bounds: {e}")
                    clipped.append(b)
            return clipped
        
        b02_1, b03_1, b04_1, b08_1, scl1 = safe_clip(b02_1, b03_1, b04_1, b08_1, scl1)
        b02_0, b03_0, b04_0, b08_0, scl0 = safe_clip(b02_0, b03_0, b04_0, b08_0, scl0)

        # Stack bands
        print(f"  📊 Stacking bands...")
        t1_stack = np.stack([b02_1.values, b03_1.values, b04_1.values, b08_1.values])
        t0_stack = np.stack([b02_0.values, b03_0.values, b04_0.values, b08_0.values])

        # Apply cloud masks
        print(f"  ☁️ Applying cloud masks...")
        keep1 = cloud_mask_from_scl(scl1)
        keep1[np.isnan(scl1.values)] = True
        keep0 = cloud_mask_from_scl(scl0) 
        keep0[np.isnan(scl0.values)] = True
        
        t1_stack = apply_mask(t1_stack, keep1)
        t0_stack = apply_mask(t0_stack, keep0)

        H, W = t1_stack.shape[1:]
        if H < ts or W < ts:
            print(f"  ⚠️ Pair {i} too small ({H}x{W} < {ts}x{ts}), skipping")
            return []

        print(f"  📏 Pair {i}: dims {H}x{W}, generating chips...")
        transform = b02_1.rio.transform()
        chip_id = 0
        
        for y in range(0, H - ts + 1, stride):
            for x in range(0, W - ts + 1, stride):
                c1 = t1_stack[:, y:y+ts, x:x+ts]
                c0 = t0_stack[:, y:y+ts, x:x+ts]
                
                # Check data validity (require at least 5% valid pixels)
                valid_ratio_1 = np.isfinite(c1).mean()
                valid_ratio_0 = np.isfinite(c0).mean()
                
                if valid_ratio_1 < 0.05 or valid_ratio_0 < 0.05:
                    continue
                    
                # Save chips
                out0 = chips_dir / f"s2_t0_{i}_{chip_id}.npy"
                out1 = chips_dir / f"s2_t1_{i}_{chip_id}.npy"
                
                np.save(out0, c0.astype("float32"))
                np.save(out1, c1.astype("float32"))
                
                # Calculate geographic bounds
                x0_map, y0_map = transform * (x, y)
                x1_map, y1_map = transform * (x + ts, y + ts)
                xmin, xmax = sorted([x0_map, x1_map])
                ymin, ymax = sorted([y0_map, y1_map])
                
                rows.append(dict(
                    chip_id=f"s2_{i}_{chip_id}",
                    split="train",
                    t0_npy=str(out0.resolve()),
                    t1_npy=str(out1.resolve()),
                    xmin=float(xmin), ymin=float(ymin),
                    xmax=float(xmax), ymax=float(ymax),
                    width=ts, height=ts, res=res, crs=str(b02_1.rio.crs),
                    valid_ratio_t0=float(valid_ratio_0),
                    valid_ratio_t1=float(valid_ratio_1),
                ))
                chip_id += 1
        
        print(f"  ✅ Wrote {chip_id} chips for pair {i}")
        return rows
        
    except Exception as e:
        print(f"❌ Skipping pair {i}: {str(e)[:200]}")
        try:
            t0_item = pystac.Item.from_dict(i0['item'])
            t1_item = pystac.Item.from_dict(i1['item'])
            print(f"  t0 scene: {t0_item.id}")
            print(f"  t1 scene: {t1_item.id}")
            print(f"  t0 assets: {list(t0_item.assets.keys())[:10]}...")
            print(f"  t1 assets: {list(t1_item.assets.keys())[:10]}...")
        except Exception as inner_e:
            print(f"  Error inspecting items: {inner_e}")
        return []

# -------------- main --------------
def main(config, items_json, t0_start, t0_end, t1_start, t1_end, out_index, tile, stride, num_workers):
    cfg = load_cfg(config)
    items_json = Path(items_json)
    if not items_json.exists():
        sys.exit(f"Items JSON not found: {items_json}")
    items = json.loads(items_json.read_text())

    if not items:
        sys.exit("No STAC items; run search first.")

    s2 = [it for it in items if it.get("collection") == "sentinel-2-l2a"]
    print(f"Found {len(s2)} S2 items total")

    t0s, t0e = datetime.fromisoformat(t0_start), datetime.fromisoformat(t0_end)
    t1s, t1e = datetime.fromisoformat(t1_start), datetime.fromisoformat(t1_end)
    
    t0 = [it for it in s2 if t0s <= item_time(it) <= t0e and 
          it["item"]["properties"].get("eo:cloud_cover", 100) < 30]
    t1 = [it for it in s2 if t1s <= item_time(it) <= t1e and 
          it["item"]["properties"].get("eo:cloud_cover", 100) < 30]

    print(f"Filtered to {len(t0)} t0 items and {len(t1)} t1 items (cloud < 30%)")

    pairs = pair_nearest(t0, t1)
    print(f"Paired {len(pairs)} S2 t0/t1 scenes")

    if not pairs:
        sys.exit("No pairs found! Check your time ranges and cloud cover.")

    aoi = aoi_union(Path(cfg["paths"]["aoi_dir"]))
    chips_dir = Path(cfg["paths"]["chips_dir"]).resolve()
    chips_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Chips will be saved to: {chips_dir}")

    res = float(cfg["preprocess"].get("resolution", 10))
    ts = int(tile)

    args_list = [(i, i0, i1, aoi, ts, stride, res, chips_dir) for i, (i0, i1) in enumerate(pairs)]

    # Process pairs
    if num_workers > 1:
        print(f"🔄 Processing {len(args_list)} pairs with {num_workers} workers...")
        with Pool(num_workers) as p:
            results = list(tqdm(p.imap(process_pair, args_list), total=len(args_list), desc="Processing pairs"))
    else:
        print(f"🔄 Processing {len(args_list)} pairs sequentially...")
        results = [process_pair(a) for a in tqdm(args_list, desc="Processing pairs")]

    # Combine all results
    rows = [r for sub in results for r in sub]
    df = pd.DataFrame(rows)

    out_index = Path(out_index)
    out_index.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print("❌ No chips were created. Possible issues:")
        print("   - Network connectivity problems")  
        print("   - AOI doesn't overlap with imagery")
        print("   - All scenes too cloudy")
        print("   - Invalid STAC item URLs")
    else:
        df.to_parquet(out_index)
        print(f"✅ SUCCESS: Wrote {len(df)} paired S2 chips → {out_index}")
        print(f"   Average valid pixel ratio: {df['valid_ratio_t0'].mean():.2%} (t0), {df['valid_ratio_t1'].mean():.2%} (t1)")

# -------------- CLI --------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate paired S2 chips for change detection")
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--items", required=True, help="Path to STAC items JSON")
    ap.add_argument("--t0", nargs=2, required=True, help="t0 start/end (YYYY-MM-DD)")
    ap.add_argument("--t1", nargs=2, required=True, help="t1 start/end (YYYY-MM-DD)")
    ap.add_argument("--out_index", required=True, help="Output Parquet index path")
    ap.add_argument("--tile_size", type=int, default=256, help="Chip size")
    ap.add_argument("--stride", type=int, default=None, help="Stride for overlapping chips (default=tile_size)")
    ap.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers (default=1 for debugging)")

    a = ap.parse_args()
    stride = a.stride if a.stride else a.tile_size
    main(a.config, a.items, a.t0[0], a.t0[1], a.t1[0], a.t1[1], a.out_index, a.tile_size, stride, a.num_workers)
    
    
    
# python -m make_chips_s2_pairs.py ^ 
# --config config.yaml ^
# --items stac_items.json ^
# --t0 2023-01-01 2023-03-31 ^
# --t1 2023-04-01 2023-06-30 ^
# --out_index chips_index.parquet ^
# --num_workers 1


# python -m make_chips_s2_fast.py ^
# --config config.yaml ^
# --items stac_items.json ^
# --t0 2023-01-01 2023-03-31 ^
# --t1 2023-04-01 2023-06-30  ^
# --out_index test_chips.parquet ^
# --tile_size 128 ^
# --stride 256 ^
# --num_workers 4


