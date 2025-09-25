import argparse, os, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from skimage.transform import resize
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

CLASSES = {
    "IGNORE": 255,
    "ROAD": 0,
    "BUILDING": 1,
    "OTHER": 2,  # vegetation loss + water change collapsed here
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

# === Worker function for multiprocessing ===
def process_chip(args):
    r, buildings, roads, wc, a = args
    xmin, ymin, xmax, ymax = r["xmin"], r["ymin"], r["xmax"], r["ymax"]
    w, h = int(r["width"]), int(r["height"])
    crs = r["crs"]
    transform = open_ref_window_transform(xmin, ymin, xmax, ymax, w, h, crs)

    # Simplified chip polygon
    chip_poly = box(xmin, ymin, xmax, ymax)
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

    # vegetation / water (if worldcover available)
    if wc is not None:
        try:
            wc_win = wc.read(
                1,
                window=rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, wc.transform),
                out_shape=(h, w),
                boundless=True,  # allows reading outside bounds
                fill_value=0  # default class
            )
            veg_mask = np.isin(wc_win, [10, 20, 30])  # Trees, Shrub, Grass
            water_mask = np.isin(wc_win, [80, 90, 95])  # Water bodies
            lbl[(lbl == CLASSES["IGNORE"]) & (veg_mask | water_mask)] = CLASSES["OTHER"]
        except Exception as e:
            print(f"⚠️ WorldCover read failed for chip {r['chip_id']}: {e}")

    # Always resize to target (e.g., 64x64) for consistency
    TARGET_SIZE = (64, 64)
    if lbl.shape != TARGET_SIZE:
        lbl = resize(lbl, TARGET_SIZE, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

    # save label
    chip_id = r["chip_id"]
    out_npy = os.path.join(a.out_dir, f"{chip_id}_label.npy")
    np.save(out_npy, lbl)

    return {"chip_id": chip_id, "label_npy": out_npy}

def main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    df = load_parquet(a.parquet)
    crs = df.iloc[0]["crs"]

    # load vectors once
    buildings = read_vector(a.osm_buildings, crs)
    roads = read_vector(a.osm_roads, crs)
    wc = rasterio.open(a.worldcover_tif) if a.worldcover_tif and os.path.exists(a.worldcover_tif) else None

    # prepare args for workers
    args_list = [(r, buildings, roads, wc, a) for _, r in df.iterrows()]

    # parallel execution
    with Pool(processes=a.num_workers) as pool:
        results = list(tqdm(pool.imap(process_chip, args_list), total=len(args_list)))

    pd.DataFrame(results).to_csv(os.path.join(a.out_dir, "labels_index.csv"), index=False)
    print(f"Wrote labels for {len(results)} chips -> {a.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="outputs/chips_index_s2.parquet")
    ap.add_argument("--preds_dir", default="outputs/preds")  # currently unused
    ap.add_argument("--osm_buildings", default="")
    ap.add_argument("--osm_roads", default="")
    ap.add_argument("--worldcover_tif", default="")
    ap.add_argument("--out_dir", default="data/labels/multiclass")
    ap.add_argument("--num_workers", type=int, default=min(4, cpu_count()))  # NEW
    args = ap.parse_args()
    main(args)


'''

  
 python -m scripts.make_labels_from_context ^
  --parquet outputs/chips_index_fast.parquet ^
  --preds_dir outputs/preds ^
  --osm_buildings data/context/osm/osm_buildings.geojson ^
  --osm_roads data/context/osm/osm_roads.geojson ^
  --out_dir data/labels/multiclass
  
  python -m scripts.make_labels_from_context ^
  --parquet outputs/chips_index_fast.parquet ^
  --osm_buildings data/context/osm/osm_buildings.geojson ^
  --osm_roads data/context/osm/osm_roads.geojson ^
  --out_dir data/labels/multiclass ^
  --num_workers 4



'''



