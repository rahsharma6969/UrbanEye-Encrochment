import argparse, os, json, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import rasterize
from shapely.geometry import box

CLASSES = {
    "IGNORE": 255,
    "BUILDING": 1,
    "ROAD": 2,
    "VEG_LOSS": 3,
    "WATER_CHANGE": 4,
}

def load_parquet(p):
    df = pd.read_parquet(p)
    if "t0_npy" not in df.columns or "t1_npy" not in df.columns:
        raise SystemExit("parquet is missing t0_npy/t1_npy columns.")
    return df

def open_ref_window_transform(xmin, ymin, xmax, ymax, width, height, crs):
    """Build an Affine transform for a chip bbox + grid size."""
    from affine import Affine
    resx = (xmax - xmin) / width
    resy = (ymin - ymax) / height  # negative
    transform = Affine.translation(xmin, ymax) * Affine.scale(resx, resy)
    return transform

def read_vector(vpath, target_crs):
    if not vpath or not os.path.exists(vpath):
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    g = gpd.read_file(vpath)
    if g.crs is None:
        g.set_crs("EPSG:4326", inplace=True)
    g = g.to_crs(target_crs)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    return g

def worldcover_mask(worldcover_tif, chip_polygon, target_transform, width, height, cls_selector):
    """Rasterize a boolean mask from WorldCover classes for the chip window."""
    if not worldcover_tif or not os.path.exists(worldcover_tif):
        return np.zeros((height, width), dtype=np.uint8)
    with rasterio.open(worldcover_tif) as src:
        chip_geom_src = gpd.GeoSeries([chip_polygon], crs=chip_polygon.crs).to_crs(src.crs).iloc[0]
        out, _ = rasterio.mask.mask(src, [chip_geom_src.__geo_interface__], crop=True, nodata=0, filled=True)
        arr = out[0]  # class map
    # resample to chip grid (nearest)
    # build a temporary memory dataset to reproject to chip grid
    dst = np.zeros((height, width), dtype=np.uint8)
    src_profile = {
        "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
        "count": 1, "dtype": "uint8", "crs": src.crs, "transform": _
    }
    # quick-and-clean: use rasterio.warp.reproject
    from rasterio.warp import reproject, Resampling
    reproject(
        source=arr, destination=dst,
        src_transform=_, src_crs=src.crs,
        dst_transform=target_transform, dst_crs=chip_polygon.crs,
        resampling=Resampling.nearest, num_threads=1
    )
    return dst

def main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    df = load_parquet(a.parquet)
    # assume all chips share the same CRS in parquet (saved earlier)
    crs = df.iloc[0]["crs"]

    # Load vectors once
    buildings = read_vector(a.ms_buildings or a.osm_buildings, crs)  # prefer MS if provided
    roads = read_vector(a.osm_roads, crs)
    # small cleanups
    if not roads.empty and a.road_buffer_m > 0:
        roads_m = roads.to_crs(roads.estimate_utm_crs())
        roads = gpd.GeoDataFrame(geometry=roads_m.buffer(a.road_buffer_m), crs=roads_m.crs).to_crs(crs)

    labels_written = 0
    rows = []

    for _, r in df.iterrows():
        xmin, ymin, xmax, ymax = r["xmin"], r["ymin"], r["xmax"], r["ymax"]
        w, h = int(r["width"]), int(r["height"])
        transform = open_ref_window_transform(xmin, ymin, xmax, ymax, w, h, crs)

        # chip polygon in chip CRS
        chip_poly = gpd.GeoSeries([box(xmin, ymin, xmax, ymax)], crs=crs).iloc[0]

        # base mask: unknown
        lbl = np.full((h, w), CLASSES["IGNORE"], dtype=np.uint8)

        # WorldCover masks (optional; need the tif)
        veg_mask = np.zeros((h, w), dtype=np.uint8)
        water_mask = np.zeros((h, w), dtype=np.uint8)
        if a.worldcover_tif and os.path.exists(a.worldcover_tif):
            wc = worldcover_mask(a.worldcover_tif, chip_poly, transform, w, h, None)
            # WorldCover classes: 10/20/30 vegetation, 80 water, 90/95 wetlands
            veg_mask = np.isin(wc, np.array([10,20,30], dtype=np.uint8))
            water_mask = np.isin(wc, np.array([80,90,95], dtype=np.uint8))

        # Buildings → label as BUILDING
        if not buildings.empty:
            bld_shapes = [(geom, 1) for geom in buildings.intersection(chip_poly).dropna() if not geom.is_empty]
            if bld_shapes:
                bld_mask = rasterize(bld_shapes, out_shape=(h, w), transform=transform, fill=0, dtype="uint8")
                lbl[bld_mask == 1] = CLASSES["BUILDING"]

        # Roads → label as ROAD (where not already building)
        if not roads.empty:
            rd_shapes = [(geom, 1) for geom in roads.intersection(chip_poly).dropna() if not geom.is_empty]
            if rd_shapes:
                rd_mask = rasterize(rd_shapes, out_shape=(h, w), transform=transform, fill=0, dtype="uint8")
                lbl[(lbl == CLASSES["IGNORE"]) & (rd_mask == 1)] = CLASSES["ROAD"]

        # Vegetation & Water (contextual). We only mark **change** pixels later; here we just stage masks.
        veg_idx = (veg_mask == 1)
        wat_idx = (water_mask == 1)

        # Now restrict labels to **changed pixels only** using your model pred TIFF for this chip, if available
        # fallback: use mean-prob .tif or a simple threshold mask
        change_mask = None
        chip_id = r["chip_id"]
        pred_tif = os.path.join(a.preds_dir, f"{chip_id}_change.tif")
        if os.path.exists(pred_tif):
            with rasterio.open(pred_tif) as src:
                arr = src.read(1).astype("float32")
            arr = np.nan_to_num(arr, nan=0.0)
            change_mask = (arr >= a.change_prob_thresh)
        else:
            # if no pred tif, skip labeling this chip
            continue

        # Apply contextual labels to only change pixels that still IGNORE
        # priority: building > road > vegetation > water
        # vegetation loss: mark changed pixels that were vegetation in WC
        lbl[(lbl == CLASSES["IGNORE"]) & change_mask & veg_idx] = CLASSES["VEG_LOSS"]
        # water change: mark changed pixels that were water/wetland in WC
        lbl[(lbl == CLASSES["IGNORE"]) & change_mask & wat_idx] = CLASSES["WATER_CHANGE"]

        # keep IGNORE for anything else
        out_npy = os.path.join(a.out_dir, f"{chip_id}_labels.npy")
        np.save(out_npy, lbl)
        labels_written += 1

        rows.append({
            "chip_id": chip_id,
            "label_npy": out_npy,
        })

    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "labels_index.csv"), index=False)
    print(f"Wrote labels for {labels_written} chips -> {a.out_dir}")

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore", category=UserWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="outputs/chips_index_s2.parquet")
    ap.add_argument("--preds_dir", default="outputs/preds")
    ap.add_argument("--ms_buildings", default="")      # preferred (or set --osm_buildings)
    ap.add_argument("--osm_buildings", default="")
    ap.add_argument("--osm_roads", default="")
    ap.add_argument("--road_buffer_m", type=float, default=6.0)
    ap.add_argument("--worldcover_tif", default="")    # optional but recommended
    ap.add_argument("--change_prob_thresh", type=float, default=0.25)
    ap.add_argument("--out_dir", default="data/labels/multiclass")
    args = ap.parse_args()
    main(args)
