import argparse, os, json, numpy as np, rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union

CLASS_MAP = {
    1: "building/impervious",
    2: "road/linear",
    3: "vegetation_loss",
    4: "water_wetland_change",
}

def polygonize_tif(tif_path, min_pixels=50):
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        mask = arr > 0
        results = []
        for geom, val in shapes(arr, mask=mask, transform=src.transform, connectivity=8):
            cls = int(val)
            if cls == 0: 
                continue
            g = shape(geom)
            if g.is_empty:
                continue
            # filter tiny regions by pixel count
            if g.area < (min_pixels * src.transform.a * abs(src.transform.e)):
                continue
            results.append({"geometry": g, "class_id": cls, "change_type": CLASS_MAP.get(cls, "unknown")})
        gdf = gpd.GeoDataFrame(results, geometry="geometry", crs=src.crs)
        if not gdf.empty and gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
        return gdf

def main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    all_gdfs = []
    for name in os.listdir(a.preds_typed_dir):
        if not name.endswith("_typed.tif"):
            continue
        tif = os.path.join(a.preds_typed_dir, name)
        gdf = polygonize_tif(tif, min_pixels=a.min_pixels)
        if not gdf.empty:
            gdf["chip_id"] = name.replace("_typed.tif", "")
            all_gdfs.append(gdf)
    if not all_gdfs:
        print("No polygons found.")
        return
    merged = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), geometry="geometry", crs="EPSG:4326")
    # dissolve tiny slivers & fix invalids
    merged["geometry"] = merged.buffer(0)
    out_path = os.path.join(a.out_dir, "changes_typed.geojson")
    merged.to_file(out_path, driver="GeoJSON")
    print(f"Wrote typed polygons → {out_path}")
    # quick counts
    print(merged["change_type"].value_counts())

if __name__ == "__main__":
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_typed_dir", default="outputs/preds_typed")
    ap.add_argument("--out_dir", default="outputs/polys_typed")
    ap.add_argument("--min_pixels", type=int, default=30)
    args = ap.parse_args()
    main(args)
