import os
import argparse
import json
import yaml
import sys
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
import urllib3

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import rasterio
import pystac
from tqdm import tqdm

urllib3.disable_warnings()

# Aggressive GDAL tuning
os.environ.update({
    "GDAL_CACHEMAX": "2048",
    "CPL_VSIL_CURL_CACHE_SIZE": "500000000",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "CPL_VSIL_CURL_USE_CACHE": "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.jp2",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_NUM_THREADS": "ALL_CPUS",
    "GDAL_TIFF_INTERNAL_MASK": "YES",
    "GDAL_HTTP_CONNECTTIMEOUT": "10",
    "GDAL_HTTP_TIMEOUT": "30",
    "GDAL_HTTP_UNSAFESSL": "YES",
})

def load_cfg(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def item_time(it) -> datetime:
    return datetime.fromisoformat(it["item"]["properties"]["datetime"].replace("Z", ""))

BAND_ALIASES = {
    "B02": ["blue", "B02", "blue-jp2"],
    "B03": ["green", "B03", "green-jp2"], 
    "B04": ["red", "B04", "red-jp2"],
    "B08": ["nir", "B08", "nir-jp2"],
    "SCL": ["SCL", "scl", "scene-classification"]
}

def _choose_asset_href(stac_item: pystac.Item, key: str) -> str:
    key_up = key.upper()
    aliases = BAND_ALIASES.get(key_up, [key, key_up, key.lower()])
    for alias in aliases:
        if alias in stac_item.assets:
            href = stac_item.assets[alias].href.strip()
            if href.lower().endswith((".tif", ".tiff", ".jp2")):
                return href
    raise KeyError(f"Asset '{key}' not found. Available: {list(stac_item.assets.keys())}")

def open_da_pick_fast(item_dict: dict, asset_key_upper: str):
    try:
        it = pystac.Item.from_dict(item_dict)
        href = _choose_asset_href(it, asset_key_upper)
        da = rxr.open_rasterio(href, masked=True, cache=False,
                            chunks={'x': 2048, 'y': 2048}, lock=False).squeeze()
        if da.sizes.get('x', 0) == 0 or da.sizes.get('y', 0) == 0:
            raise ValueError(f"Empty raster for {asset_key_upper}")
        return da
    except Exception as e:
        print(f"  ⚠️ Failed to open {asset_key_upper}: {e}")
        return None

def cloud_mask_from_scl(scl_array):
    """Simple cloud masking based on SCL values"""
    # Keep: vegetation(4), not-vegetated(5), water(6), snow/ice(11)
    # Remove: clouds(8,9,10), cloud shadows(3), saturated(1), dark(2)
    valid_classes = [4, 5, 6, 7, 11]  # Add bare soils(7) as valid
    return np.isin(scl_array.values if hasattr(scl_array, 'values') else scl_array, valid_classes)

def apply_mask(stack, mask):
    """Apply mask to multi-band stack"""
    if mask is None:
        return stack
    # Broadcast mask to all bands
    mask_3d = np.broadcast_to(mask, stack.shape)
    masked_stack = stack.copy()
    masked_stack[~mask_3d] = np.nan
    return masked_stack

def aoi_union(aoi_dir: Path) -> gpd.GeoDataFrame:
    files = list(Path(aoi_dir).glob("*.geojson"))
    if not files:
        print(f"⚠️ No geojson files found in {aoi_dir}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    
    gdfs = []
    for f in files:
        try:
            g = gpd.read_file(f)
            if not g.empty:
                gdfs.append(g.to_crs("EPSG:4326"))
        except Exception as e:
            print(f"⚠️ Failed to load {f}: {e}")
    
    if not gdfs:
        print("⚠️ No valid geojson files found")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
    merged = pd.concat(gdfs, ignore_index=True).explode(ignore_index=True)
    u = merged.geometry.unary_union
    return gpd.GeoDataFrame(geometry=[u], crs="EPSG:4326")

def pair_nearest(t0_items, t1_items):
    if not t1_items:
        return []
    t1t = [(item_time(i1), i1) for i1 in t1_items]
    out = []
    for i0 in t0_items:
        best = min(t1t, key=lambda x: abs(x[0]-item_time(i0)))[1]
        out.append((i0, best))
    return out

def process_pair(args):
    i, i0, i1, aoi, ts, stride, res, chips_dir = args
    rows = []
    try:
        print(f"⚡ Processing pair {i}...")
        
        # Load bands
        bands_t1 = {}
        bands_t0 = {}
        
        for band in ["B02", "B03", "B04", "B08"]:
            bands_t1[band] = open_da_pick_fast(i1["item"], band)
            bands_t0[band] = open_da_pick_fast(i0["item"], band)
        
        # Load SCL (optional)
        scl1 = open_da_pick_fast(i1["item"], "SCL")
        scl0 = open_da_pick_fast(i0["item"], "SCL")
        
        # Check if all required bands loaded
        if any(bands_t1[b] is None for b in ["B02", "B03", "B04", "B08"]) or \
           any(bands_t0[b] is None for b in ["B02", "B03", "B04", "B08"]):
            print(f"  ⚠️ Missing bands in pair {i}, skipping")
            return []

        # Reproject t0 to match t1
        reference_band = bands_t1["B02"]
        try:
            for band in ["B02", "B03", "B04", "B08"]:
                bands_t0[band] = bands_t0[band].rio.reproject_match(reference_band)
            
            if scl0 is not None and scl1 is not None:
                scl0 = scl0.rio.reproject_match(reference_band, resampling=rasterio.enums.Resampling.nearest)
                scl1 = scl1.rio.reproject_match(reference_band, resampling=rasterio.enums.Resampling.nearest)
                
        except Exception as e:
            print(f"  ⚠️ Reprojection failed for pair {i}: {e}")
            return []

        # Clip to AOI if provided
        if not aoi.empty:
            aoi_ref = aoi.to_crs(reference_band.rio.crs)
            try:
                for band in ["B02", "B03", "B04", "B08"]:
                    bands_t1[band] = bands_t1[band].rio.clip(aoi_ref.geometry, aoi_ref.crs, drop=True)
                    bands_t0[band] = bands_t0[band].rio.clip(aoi_ref.geometry, aoi_ref.crs, drop=True)
                
                if scl1 is not None:
                    scl1 = scl1.rio.clip(aoi_ref.geometry, aoi_ref.crs, drop=True)
                if scl0 is not None:
                    scl0 = scl0.rio.clip(aoi_ref.geometry, aoi_ref.crs, drop=True)
                    
            except Exception as e:
                print(f"  ⚠️ Clipping failed for pair {i}: {e}")

        # Stack bands
        t1_stack = np.stack([bands_t1[b].values for b in ["B02", "B03", "B04", "B08"]])
        t0_stack = np.stack([bands_t0[b].values for b in ["B02", "B03", "B04", "B08"]])

        # Apply cloud masking if SCL is available
        if scl1 is not None and scl0 is not None:
            try:
                keep1 = cloud_mask_from_scl(scl1)
                keep0 = cloud_mask_from_scl(scl0)
                t1_stack = apply_mask(t1_stack, keep1)
                t0_stack = apply_mask(t0_stack, keep0)
                print(f"  ✅ Applied cloud masking for pair {i}")
            except Exception as e:
                print(f"  ⚠️ Masking failed for pair {i}: {e}")

        H, W = t1_stack.shape[1:]
        print(f"  📐 Image dimensions: {H} x {W}")
        
        if H < ts or W < ts:
            print(f"  ⚠️ Image too small ({H}x{W}) for tile size {ts}")
            return []

        transform = reference_band.rio.transform()
        chip_id = 0
        
        # Generate chips
        for y in range(0, H - ts + 1, stride):
            for x in range(0, W - ts + 1, stride):
                c1 = t1_stack[:, y:y+ts, x:x+ts]
                c0 = t0_stack[:, y:y+ts, x:x+ts]
                
                # Quality check: skip if too many invalid pixels
                valid_ratio = np.isfinite(c1).mean()
                if valid_ratio < 0.1:
                    continue
                
                # Save chips
                out0 = chips_dir / f"s2_t0_{i}_{chip_id}.npy"
                out1 = chips_dir / f"s2_t1_{i}_{chip_id}.npy"
                
                np.save(out0, c0.astype("float32"))
                np.save(out1, c1.astype("float32"))
                
                # Calculate geographic bounds
                x_geo = transform[2] + x * transform[0]
                y_geo = transform[5] + y * transform[4]
                x_geo_max = transform[2] + (x + ts) * transform[0]
                y_geo_max = transform[5] + (y + ts) * transform[4]
                
                rows.append({
                    "chip_id": f"s2_{i}_{chip_id}", 
                    "split": "train",
                    "t0_npy": str(out0.resolve()), 
                    "t1_npy": str(out1.resolve()),
                    "xmin": float(x_geo),
                    "ymin": float(y_geo_max),  # Note: y flipped for geo coordinates
                    "xmax": float(x_geo_max),
                    "ymax": float(y_geo),     # Note: y flipped for geo coordinates
                    "width": ts, 
                    "height": ts, 
                    "res": res, 
                    "crs": str(reference_band.rio.crs),
                    "valid_pixel_ratio": float(valid_ratio)
                })
                chip_id += 1
                
        print(f"  ✅ Created {chip_id} chips for pair {i}")
        return rows
        
    except Exception as e:
        print(f"❌ Pair {i} failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def main(args):
    cfg = load_cfg(args.config)
    items = json.loads(Path(args.items).read_text())
    s2 = [it for it in items if it.get("collection") == "sentinel-2-l2a"]
    
    print(f"📡 Found {len(s2)} Sentinel-2 items")

    # Parse time ranges
    t0s, t0e = datetime.fromisoformat(args.t0[0]), datetime.fromisoformat(args.t0[1])
    t1s, t1e = datetime.fromisoformat(args.t1[0]), datetime.fromisoformat(args.t1[1])
    
    # Filter by time and cloud cover
    t0 = [it for it in s2 if t0s <= item_time(it) <= t0e and it["item"]["properties"].get("eo:cloud_cover", 100) < 15]
    t1 = [it for it in s2 if t1s <= item_time(it) <= t1e and it["item"]["properties"].get("eo:cloud_cover", 100) < 15]
    
    print(f"🌤️ After cloud filtering: {len(t0)} t0 items, {len(t1)} t1 items")
    
    # Create pairs
    pairs = pair_nearest(t0, t1)
    print(f"🔗 Created {len(pairs)} image pairs")
    
    if args.fast_mode:
        pairs = pairs[:10]
        args.tile_size, args.stride = 64, 64
        print("⚡ FAST MODE ENABLED - Processing 10 pairs with 64x64 tiles")

    # Load AOI
    aoi = aoi_union(Path(cfg["paths"]["aoi_dir"]))
    print(f"🗺️ AOI loaded: {len(aoi)} geometries")
    
    # Setup output directory
    chips_dir = Path(cfg["paths"]["chips_dir"]).resolve()
    chips_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Chips will be saved to: {chips_dir}")

    # Prepare arguments for multiprocessing
    args_list = [
        (i, i0, i1, aoi, args.tile_size, args.stride, cfg["preprocess"]["resolution"], chips_dir) 
        for i, (i0, i1) in enumerate(pairs)
    ]
    
    print(f"⚡ Processing {len(pairs)} pairs with {args.num_workers} workers...")
    
    # Process pairs
    if args.num_workers == 1:
        # Single-threaded for debugging
        results = []
        for arg_set in tqdm(args_list):
            results.append(process_pair(arg_set))
    else:
        # Multi-threaded
        with Pool(args.num_workers) as p:
            results = list(tqdm(p.imap(process_pair, args_list), total=len(args_list)))
    
    # Combine results
    all_rows = [r for sub in results for r in sub]
    df = pd.DataFrame(all_rows)
    
    if df.empty:
        print("❌ No chips created")
    else:
        df.to_parquet(args.out_index)
        print(f"✅ Created {len(df)} chips → {args.out_index}")
        print(f"📊 Quality stats: avg valid pixels = {df['valid_pixel_ratio'].mean():.2%}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config YAML file")
    ap.add_argument("--items", required=True, help="Path to STAC items JSON file")
    ap.add_argument("--t0", nargs=2, required=True, help="Time range for t0 images (start end)")
    ap.add_argument("--t1", nargs=2, required=True, help="Time range for t1 images (start end)")
    ap.add_argument("--out_index", required=True, help="Output parquet index file")
    ap.add_argument("--tile_size", type=int, default=128, help="Size of image chips")
    ap.add_argument("--stride", type=int, default=256, help="Stride for chip extraction")
    ap.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    ap.add_argument("--fast_mode", action="store_true", help="Fast mode for testing")
    args = ap.parse_args()
    main(args)




# python -m scripts.make_chips_s2_fast ^
#   --config configs/config.yaml ^
#   --items outputs/stac_items_navi.json ^
#   --t0 2022-01-01 2022-12-31 ^
#   --t1 2023-01-01 2023-12-31 ^
#   --out_index outputs/chips_index_test.parquet ^
#   --tile_size 128 ^
#   --stride 256 ^
#   --num_workers 4

'''
python -m make_chips_s2_fast.py ^
  --config configs/config.yaml ^
  --items outputs/stac_items_mumbai.json ^
  --t0 2022-01-01 2022-06-30 ^
  --t1 2022-07-01 2022-12-31 ^
  --out_index outputs/chips_index_s2.parquet ^
  --tile_size 128 ^
  --stride 64 ^
  --num_workers 8
  
  python -m scripts.make_chips_s2_fast ^
  --config configs/config.yaml ^
  --items outputs/stac_items_mumbai.json ^
  --t0 2022-01-01 2022-12-31 ^
  --t1 2023-01-01 2023-12-31 ^
  --out_index outputs/chips_index_mumbai.parquet ^
  --tile_size 256 ^
  --stride 128 ^
  --num_workers 8

python -m scripts.make_chips_s2_fast ^
  --config configs/config.yaml ^
  --items outputs/stac_items_mumbai.json ^
  --t0 2022-01-01 2022-06-30 ^
  --t1 2022-07-01 2022-12-31 ^
  --out_index outputs/chips_index_fast.parquet ^
  --fast_mode ^
  --num_workers 2


'''


