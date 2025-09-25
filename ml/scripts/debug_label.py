# scripts/debug_label.py
import os
import sys
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import box
from rasterio.warp import calculate_default_transform, reproject, Resampling

def print_unique(arr, name):
    u = np.unique(arr)
    print(f"  {name} unique values: {u}")

def load_parquet(parquet_path):
    if not os.path.exists(parquet_path):
        raise SystemExit(f"Parquet not found: {parquet_path}")
    return pd.read_parquet(parquet_path)

def find_row_for_chip(df, chip_id):
    row = df[df["chip_id"] == chip_id]
    if row.empty:
        raise SystemExit(f"chip_id {chip_id} not found in parquet")
    return row.iloc[0]

def sample_worldcover_at_chip(wc_tif, xmin, ymin, xmax, ymax, out_shape):
    if not os.path.exists(wc_tif):
        return None
    with rasterio.open(wc_tif) as src:
        # build transform and window that covers the chip bounds in wc CRS
        dst_crs = src.crs
        # if chip coords are already in same crs, do direct read; else we will reproject sample
        return src.read(1)  # fallback: full read (user can improve)

def raster_bounds_to_window_transform(xmin, ymin, xmax, ymax, width, height):
    # returns a rasterio Affine transform for a window with upper-left at (xmin,ymax)
    from affine import Affine
    resx = (xmax - xmin) / width
    resy = (ymin - ymax) / height
    tf = Affine.translation(xmin, ymax) * Affine.scale(resx, resy)
    return tf

def main():
    ap = argparse.ArgumentParser(description="Debug why a generated label is all 255")
    ap.add_argument("--labels_dir", default="data/labels/multiclass")
    ap.add_argument("--parquet", default="outputs/chips_index_fast.parquet")
    ap.add_argument("--preds_dir", default="outputs/preds")
    ap.add_argument("--osm_buildings", default="data/context/osm/osm_buildings.geojson")
    ap.add_argument("--osm_roads", default="data/context/osm/osm_roads.geojson")
    ap.add_argument("--worldcover_tif", default="data/context/worldcover.tif")
    ap.add_argument("--chip_id", required=True, help="chip_id without _label.npy (e.g. s2_0_50)")
    ap.add_argument("--change_prob_thresh", type=float, default=0.25)
    args = ap.parse_args()

    label_path = os.path.join(args.labels_dir, f"{args.chip_id}_label.npy")
    print("\n=== Debugging:", args.chip_id, "===\n")
    if not os.path.exists(label_path):
        print("Label not found:", label_path)
    else:
        lbl = np.load(label_path)
        print("Label path:", label_path)
        print("  Shape:", lbl.shape)
        print_unique(lbl, "Label")

    # load parquet and row
    df = load_parquet(args.parquet)
    row = find_row_for_chip(df, args.chip_id)
    print("\nChip index row values:")
    for c in ["t0_npy","t1_npy","xmin","ymin","xmax","ymax","width","height","crs"]:
        print(f"  {c}: {row.get(c)}")

    # load chip arrays
    for t in ["t0_npy","t1_npy"]:
        p = row[t]
        print(f"\nLoading {t}: {p}")
        if not os.path.exists(p):
            print("  File does not exist.")
            continue
        arr = np.load(p)
        print("  shape:", arr.shape, "dtype:", arr.dtype)
        print("  finite ratio:", float(np.isfinite(arr).sum())/arr.size)
        try:
            print("  min/max (ignoring NaN):", float(np.nanmin(arr)), float(np.nanmax(arr)))
        except Exception:
            pass

    # check pred tif
    pred_tif1 = os.path.join(args.preds_dir, f"{args.chip_id}_change.tif")
    pred_tif2 = os.path.join(args.preds_dir, f"{args.chip_id}_typed.tif")  # alternate naming
    pred = None
    for pt in (pred_tif1, pred_tif2):
        if os.path.exists(pt):
            pred = pt
            break
    print("\nPrediction file:", pred)
    if pred:
        try:
            with rasterio.open(pred) as src:
                p = src.read(1).astype("float32")
                print("  pred shape:", p.shape, "dtype:", p.dtype)
                print("  pred min/max:", p.min(), p.max())
                # change mask
                cm = (np.nan_to_num(p, nan=0.0) >= args.change_prob_thresh)
                print("  change pixels count (>=thresh):", int(cm.sum()), "/", cm.size)
        except Exception as e:
            print("  Failed to read pred tif:", e)
    else:
        print("  No prediction file found for chip.")

    # check OSM buildings / roads intersection
    chip_xmin, chip_ymin = float(row["xmin"]), float(row["ymin"])
    chip_xmax, chip_ymax = float(row["xmax"]), float(row["ymax"])
    chip_w, chip_h = int(row["width"]), int(row["height"])
    chip_crs = row["crs"]
    chip_geom = box(chip_xmin, chip_ymin, chip_xmax, chip_ymax)
    print("\nChip bbox:", chip_geom.bounds, "CRS:", chip_crs)

    # buildings
    if os.path.exists(args.osm_buildings):
        bld = gpd.read_file(args.osm_buildings)
        if bld.crs is None:
            bld.set_crs("EPSG:4326", inplace=True)
        try:
            bld = bld.to_crs(chip_crs)
        except Exception as e:
            print("  Cannot reproject buildings to chip CRS:", e)
        inter_b = bld[bld.intersects(chip_geom)]
        print("  Building features intersecting chip:", len(inter_b))
    else:
        print("  OSM buildings file not found:", args.osm_buildings)

    # roads
    if os.path.exists(args.osm_roads):
        roads = gpd.read_file(args.osm_roads)
        if roads.crs is None:
            roads.set_crs("EPSG:4326", inplace=True)
        try:
            roads = roads.to_crs(chip_crs)
        except Exception as e:
            print("  Cannot reproject roads to chip CRS:", e)
        inter_r = roads[roads.intersects(chip_geom)]
        print("  Road features intersecting chip:", len(inter_r))
        if len(inter_r)==0:
            print("  (Try increasing road buffer parameter when generating labels)")
    else:
        print("  OSM roads file not found:", args.osm_roads)

    # worldcover (simple check)
    if os.path.exists(args.worldcover_tif):
        try:
            with rasterio.open(args.worldcover_tif) as src:
                print("  Worldcover CRS:", src.crs, "shape:", src.width, src.height)
                # attempt to read small window corresponding to chip (best-effort)
                transform = raster_bounds_to_window_transform(chip_xmin, chip_ymin, chip_xmax, chip_ymax, chip_w, chip_h)
                # Reproject not implemented here — just note mismatch possibility
                print("  Note: worldcover read requires matching CRS. If CRS differs, samples won't align.")
        except Exception as e:
            print("  Failed to open worldcover:", e)
    else:
        print("  No worldcover tif provided:", args.worldcover_tif)

    # final diagnosis hints
    print("\n--- Diagnosis Hints ---")
    if not os.path.exists(label_path):
        print(" * No label file exists (label generator may have skipped).")
    else:
        if np.array_equal(np.unique(lbl), np.array([255])):
            print(" * Label is all IGNORE (255). Possible causes:")
            print("    - No OSM building/road intersects the chip area.")
            print("    - Prediction change mask is empty (no change predicted).")
            print("    - Worldcover didn't indicate vegetation/water at chip (or worldcover CRS mismatched).")
            print("    - Chip CRS / bounds mismatch: OSM/worldcover/pred must be in same CRS and aligned.")
    print("\nDone.")

if __name__ == "__main__":
    main()


'''
python -m scripts.debug_label ^
--chip_id s2_0_50 ^
--parquet outputs\mumbai_index.parquet ^
--labels_dir data\labels\multiclass  ^
--preds_dir outputs\preds  ^
--osm_buildings data\context\osm\osm_buildings.geojson ^
--osm_roads data\context\osm\osm_roads.geojson 

'''
