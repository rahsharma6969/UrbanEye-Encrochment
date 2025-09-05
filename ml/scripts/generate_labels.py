# ml/scripts/generate_labels.py

import argparse
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from skimage.transform import resize
from scipy.ndimage import binary_opening, binary_closing
from pathlib import Path


# === Classes (4-class system) ===
CLASSES = {
    "IGNORE": 255,
    "BACKGROUND": 0,
    "BUILDING": 1,
    "ROAD": 2,
    "CHANGE": 3,
}


def load_parquet(p):
    """Load and validate parquet file."""
    if not os.path.exists(p):
        raise FileNotFoundError(f"Parquet file not found: {p}")
    df = pd.read_parquet(p)
    required_cols = ["chip_id", "xmin", "ymin", "xmax", "ymax", "width", "height", "crs"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Parquet is missing columns: {missing}")
    return df


def open_ref_window_transform(xmin, ymin, xmax, ymax, width, height, crs):
    """Create affine transform for rasterization."""
    from affine import Affine
    resx = (xmax - xmin) / width
    resy = (ymax - ymin) / height  # Note: top-down, so positive
    return Affine.translation(xmin, ymin) * Affine.scale(resx, resy)


def read_vector(vpath, target_crs):
    """Read vector file and reproject to target CRS."""
    if not vpath or not os.path.exists(vpath):
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    try:
        g = gpd.read_file(vpath)
    except Exception as e:
        raise RuntimeError(f"Failed to read vector file: {vpath}\n{e}")

    if g.crs is None:
        print(f"⚠️  No CRS in {vpath}, assuming EPSG:4326")
        g.set_crs("EPSG:4326", inplace=True)
    g = g.to_crs(target_crs)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    return g


def clean_labels(lbl):
    """Apply morphological cleanup to masks."""
    cleaned = lbl.copy()
    structure_2x2 = np.ones((2, 2))
    structure_3x3 = np.ones((3, 3))

    for class_id in [CLASSES["BUILDING"], CLASSES["ROAD"], CLASSES["CHANGE"]]:
        if np.sum(lbl == class_id) == 0:
            continue
        mask = (lbl == class_id)
        mask_clean = binary_opening(mask, structure=structure_2x2)
        mask_clean = binary_closing(mask_clean, structure=structure_3x3)
        # Restore original background
        cleaned[mask] = CLASSES["BACKGROUND"]
        cleaned[mask_clean] = class_id
    return cleaned


def main(a):
    print("🚀 Starting label generation...")

    # Validate and create output directory
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load chip index
    df = load_parquet(a.parquet)
    print(f"Loaded {len(df)} chips from parquet.")

    # Use first chip's CRS
    crs = df.iloc[0]["crs"]
    if not crs:
        raise ValueError("CRS not found in parquet file.")

    # Load context layers
    print("📂 Loading context data...")
    buildings = read_vector(a.ms_buildings, crs)
    roads = read_vector(a.osm_roads, crs)

    # --- 🔥 Critical: Warn or fail if required data is missing ---
    if a.require_context:
        if a.ms_buildings and not os.path.exists(a.ms_buildings):
            raise FileNotFoundError(f"--ms_buildings not found: {a.ms_buildings}")
        if a.osm_roads and not os.path.exists(a.osm_roads):
            raise FileNotFoundError(f"--osm_roads not found: {a.osm_roads}")

    if buildings.empty:
        print("⚠️  No buildings loaded. Buildings will not be labeled.")
    else:
        print(f"✅ Loaded {len(buildings)} building geometries.")

    if roads.empty:
        print("⚠️  No roads loaded. Roads will not be labeled.")
    else:
        print(f"✅ Loaded {len(roads)} road geometries.")
    # -----------------------------------------------------------

    # Buffer roads if requested
    if not roads.empty and a.road_buffer_m > 0:
        print(f"🛣️  Buffering roads by {a.road_buffer_m}m...")
        utm_crs = roads.estimate_utm_crs()
        roads_utm = roads.to_crs(utm_crs)
        roads_buffered = roads_utm.buffer(a.road_buffer_m)
        roads = gpd.GeoDataFrame(geometry=roads_buffered, crs=utm_crs).to_crs(crs)

    # Setup augmentations
    aug_types = ["original"]
    if a.augment:
        aug_types.extend(["rot90", "rot180", "rot270", "fliph", "flipv"])

    print(f"⚙️  Processing {len(df)} chips × {len(aug_types)} aug types...")

    rows = []
    labels_written = 0

    for _, r in df.iterrows():
        chip_id = r["chip_id"]
        xmin, ymin, xmax, ymax = r["xmin"], r["ymin"], r["xmax"], r["ymax"]
        w, h = int(r["width"]), int(r["height"])
        transform = open_ref_window_transform(xmin, ymin, xmax, ymax, w, h, crs)

        # Initialize label array
        lbl = np.full((h, w), CLASSES["BACKGROUND"], dtype=np.uint8)

        # === Add Buildings ===
        if not buildings.empty:
            bbox = box(xmin, ymin, xmax, ymax)
            intersecting = buildings.intersects(bbox)
            bld_geoms = buildings[intersecting].geometry.dropna()
            if not bld_geoms.empty:
                shapes = ((geom, CLASSES["BUILDING"]) for geom in bld_geoms if not geom.is_empty)
                mask = rasterize(shapes, out_shape=(h, w), transform=transform, fill=0, dtype="uint8")
                lbl[mask == CLASSES["BUILDING"]] = CLASSES["BUILDING"]

        # === Add Roads ===
        if not roads.empty:
            bbox = box(xmin, ymin, xmax, ymax)
            intersecting = roads.intersects(bbox)
            rd_geoms = roads[intersecting].geometry.dropna()
            if not rd_geoms.empty:
                shapes = ((geom, CLASSES["ROAD"]) for geom in rd_geoms if not geom.is_empty)
                mask = rasterize(shapes, out_shape=(h, w), transform=transform, fill=0, dtype="uint8")
                # Only where background remains
                lbl[(lbl == CLASSES["BACKGROUND"]) & (mask == CLASSES["ROAD"])] = CLASSES["ROAD"]

        # === Add Change Mask ===
        pred_tif = os.path.join(a.preds_dir, f"{chip_id}_change.tif")
        if not os.path.exists(pred_tif):
            pred_tif = os.path.join(a.preds_dir, f"{chip_id}_typed.tif")
        if not os.path.exists(pred_tif):
            print(f"⚠️  Skipping {chip_id}: no change prediction found.")
            continue

        try:
            with rasterio.open(pred_tif) as src:
                arr = src.read(1).astype("float32")
        except Exception as e:
            print(f"❌ Failed to read {pred_tif}: {e}")
            continue

        if arr.shape != (h, w):
            arr = resize(arr, (h, w), order=1, preserve_range=True, anti_aliasing=True)

        change_mask = (np.nan_to_num(arr, nan=0.0) >= a.change_prob_thresh)
        lbl[(lbl == CLASSES["BACKGROUND"]) & change_mask] = CLASSES["CHANGE"]

        # === Clean labels ===
        if a.clean_labels:
            lbl = clean_labels(lbl)

        # === Save augmented versions ===
        for suffix in aug_types:
            out_lbl = lbl.copy()

            if suffix != "original":
                if suffix == "rot90": out_lbl = np.rot90(lbl, 1)
                elif suffix == "rot180": out_lbl = np.rot90(lbl, 2)
                elif suffix == "rot270": out_lbl = np.rot90(lbl, 3)
                elif suffix == "fliph": out_lbl = np.fliplr(lbl)
                elif suffix == "flipv": out_lbl = np.flipud(lbl)
                else:
                    continue  # skip unsupported

            # Save .npy
            out_npy = out_dir / f"{chip_id}_{suffix}_label.npy"
            np.save(out_npy, out_lbl)

            # Record in index
            rows.append({"chip_id": f"{chip_id}_{suffix}", "label_npy": str(out_npy)})
            labels_written += 1

        # --- Debug: Print class stats ---
        if a.debug:
            u, c = np.unique(lbl, return_counts=True)
            stats = {CLASSES[k]: v for k, v in CLASSES.items() if k != "IGNORE"}
            for cls_id in u:
                stats[cls_id] = stats.get(cls_id, 0) + np.sum(lbl == cls_id)
            print(f"{chip_id}: {stats}")

    # Save index CSV
    index_csv = out_dir / "labels_index.csv"
    pd.DataFrame(rows).to_csv(index_csv, index=False)
    print(f"\n✅ Wrote {labels_written} labels to {a.out_dir}")
    print(f"📄 Label index saved: {index_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate 4-class semantic labels for change + context (buildings, roads)"
    )
    ap.add_argument("--parquet", required=True, help="Path to chips index parquet (with crs, bounds)")
    ap.add_argument("--preds_dir", required=True, help="Directory with change probability TIFs")
    ap.add_argument("--out_dir", required=True, help="Output directory for .npy labels")
    ap.add_argument("--ms_buildings", default="", help="Path to Microsoft Buildings GeoJSON (optional)")
    ap.add_argument("--osm_roads", default="", help="Path to OSM Roads GeoJSON (optional)")
    ap.add_argument("--road_buffer_m", type=float, default=6.0, help="Buffer distance for roads (meters)")
    ap.add_argument("--augment", action="store_true", help="Generate rotated/flipped versions")
    ap.add_argument("--clean_labels", action="store_true", help="Apply morphological cleanup")
    ap.add_argument("--change_prob_thresh", type=float, default=0.02, help="Threshold for change detection")
    ap.add_argument("--debug", action="store_true", help="Print label stats per chip")
    ap.add_argument("--require_context", action="store_true",
                    help="Fail if --ms_buildings or --osm_roads are missing")

    args = ap.parse_args()
    main(args)




# python -m scripts.generate_labels ^
#   --parquet outputs/chips_index_s2.parquet ^
#   --preds_dir outputs/preds ^
#   --out_dir data/labels/multiclass_4class ^
#   --augment ^
#   --clean_labels ^
#   --change_prob_thresh 0.02


# python -m scripts.generate_labels ^
#   --parquet outputs/chips_index_s2.parquet ^
#   --preds_dir outputs/preds_typed ^
#   --ms_buildings data/context/ms_buildings.geojson ^
#   --osm_roads data/context/osm_roads.geojson ^
#   --out_dir data/labels/multiclass_4class ^
#   --augment ^
#   --clean_labels ^
#   --change_prob_thresh 0.02
