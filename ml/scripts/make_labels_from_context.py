import argparse, os, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from skimage.transform import resize

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
    from affine import Affine
    resx = (xmax - xmin) / width
    resy = (ymin - ymax) / height
    return Affine.translation(xmin, ymax) * Affine.scale(resx, resy)

def read_vector(vpath, target_crs):
    if not vpath or not os.path.exists(vpath):
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    g = gpd.read_file(vpath)
    if g.crs is None:
        g.set_crs("EPSG:4326", inplace=True)
    g = g.to_crs(target_crs)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    return g

def main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    df = load_parquet(a.parquet)
    crs = df.iloc[0]["crs"]

    # load vectors
    buildings = read_vector(a.ms_buildings or a.osm_buildings, crs)
    roads = read_vector(a.osm_roads, crs)
    if not roads.empty and a.road_buffer_m > 0:
        roads_m = roads.to_crs(roads.estimate_utm_crs())
        roads = gpd.GeoDataFrame(geometry=roads_m.buffer(a.road_buffer_m), crs=roads_m.crs).to_crs(crs)

    labels_written, rows = 0, []

    for _, r in df.iterrows():
        xmin, ymin, xmax, ymax = r["xmin"], r["ymin"], r["xmax"], r["ymax"]
        w, h = int(r["width"]), int(r["height"])
        transform = open_ref_window_transform(xmin, ymin, xmax, ymax, w, h, crs)

        chip_poly = gpd.GeoSeries([box(xmin, ymin, xmax, ymax)], crs=crs).iloc[0]
        lbl = np.full((h, w), CLASSES["IGNORE"], dtype=np.uint8)

        # buildings
        if not buildings.empty:
            bld_shapes = [(geom, 1) for geom in buildings.intersection(chip_poly).dropna() if not geom.is_empty]
            if bld_shapes:
                bld_mask = rasterize(bld_shapes, out_shape=(h, w), transform=transform, fill=0, dtype="uint8")
                lbl[bld_mask == 1] = CLASSES["BUILDING"]

        # roads
        if not roads.empty:
            rd_shapes = [(geom, 1) for geom in roads.intersection(chip_poly).dropna() if not geom.is_empty]
            if rd_shapes:
                rd_mask = rasterize(rd_shapes, out_shape=(h, w), transform=transform, fill=0, dtype="uint8")
                lbl[(lbl == CLASSES["IGNORE"]) & (rd_mask == 1)] = CLASSES["ROAD"]

        # load model pred
        chip_id = r["chip_id"]
        pred_tif = os.path.join(a.preds_dir, f"{chip_id}_change.tif")
        if not os.path.exists(pred_tif):
            continue
        with rasterio.open(pred_tif) as src:
            arr = src.read(1).astype("float32")
        change_mask = (np.nan_to_num(arr, nan=0.0) >= a.change_prob_thresh)

        # vegetation / water (if worldcover available)
        if a.worldcover_tif and os.path.exists(a.worldcover_tif):
            with rasterio.open(a.worldcover_tif) as wc_src:
                wc = wc_src.read(1)
                veg_mask = np.isin(wc, [10, 20, 30])
                water_mask = np.isin(wc, [80, 90, 95])
        else:
            veg_mask = np.zeros((h, w), bool)
            water_mask = np.zeros((h, w), bool)

        lbl[(lbl == CLASSES["IGNORE"]) & change_mask & veg_mask] = CLASSES["VEG_LOSS"]
        lbl[(lbl == CLASSES["IGNORE"]) & change_mask & water_mask] = CLASSES["WATER_CHANGE"]

        # === FIX 1: force uniform size (256×256) ===
        if (h, w) != (256, 256):
            lbl = resize(lbl, (256, 256), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

        # === FIX 2: save with correct naming ===
        out_npy = os.path.join(a.out_dir, f"{chip_id}_label.npy")
        np.save(out_npy, lbl)
        rows.append({"chip_id": chip_id, "label_npy": out_npy})
        labels_written += 1

    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "labels_index.csv"), index=False)
    print(f"Wrote labels for {labels_written} chips -> {a.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="outputs/chips_index_s2.parquet")
    ap.add_argument("--preds_dir", default="outputs/preds")
    ap.add_argument("--ms_buildings", default="")
    ap.add_argument("--osm_buildings", default="")
    ap.add_argument("--osm_roads", default="")
    ap.add_argument("--road_buffer_m", type=float, default=6.0)
    ap.add_argument("--worldcover_tif", default="")
    ap.add_argument("--change_prob_thresh", type=float, default=0.25)
    ap.add_argument("--out_dir", default="data/labels/multiclass")
    args = ap.parse_args()
    main(args)




# python -m src.train.train_multiclass ^
#   --parquet outputs/chips_index_s2.parquet ^
#   --labels_dir data/labels/multiclass ^
#   --epochs 20 ^
#   --batch_size 8 ^
#   --out_dir outputs/models_multi
