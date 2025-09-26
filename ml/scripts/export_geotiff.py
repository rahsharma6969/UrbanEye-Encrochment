


# ml/scripts/export_geotiff.py
"""
Export change detection predictions to GeoTIFF with georeferencing.
Now handles: {base_name}.npy_th0.30_ma20.png → correct format
"""

import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
import rasterio
from rasterio.transform import from_bounds

# --- CONFIG ---
INDEX_PARQUET = r"C:\Users\Rahul\CollegeProject\UrbanEyeML\ml\data\LEVIR_CD\chips_256\index.parquet"
PREDICTION_DIR = r"C:\Users\Rahul\CollegeProject\UrbanEyeML\ml\outputs\preds\thresholded"
OUTPUT_DIR = r"C:\Users\Rahul\CollegeProject\UrbanEyeML\ml\outputs\preds\geotiff"

# Match exactly what was used in inference
THRESHOLD_FORMAT = "0.30"   # ← Because your files are named _th0.30_
MIN_AREA = 20

def main():
    print("🔍 Loading validation dataset...")
    df = pd.read_parquet(INDEX_PARQUET)
    val_df = df[df.split == 'val'].reset_index(drop=True)

    if val_df.empty:
        print("❌ No validation samples found.")
        return

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"✅ Found {len(val_df)} validation chips → exporting to GeoTIFF")

    for idx, row in val_df.iterrows():
        mask_npy = row['mask_npy']
        base_name = Path(mask_npy).stem  # e.g., 'train_162_256_256'

        # 🔧 FIXED: Use the EXACT format from your debugging: {base_name}.npy_th0.30_ma20.png
        pred_filename = f"{base_name}.npy_th{THRESHOLD_FORMAT}_ma{MIN_AREA}.png"
        pred_png = Path(PREDICTION_DIR) / pred_filename

        if not pred_png.exists():
            print(f"⚠️ Missing prediction: {pred_filename}")
            continue

        # Load binary mask (already thresholded and post-processed)
        try:
            img = Image.open(pred_png)
            bin_mask = np.array(img).astype(np.uint8)
            bin_mask = (bin_mask > 0).astype(np.uint8) * 255  # Ensure clean 0/255
        except Exception as e:
            print(f"❌ Failed to load {pred_png}: {e}")
            continue

        # Extract coordinates from name
        # Handle different patterns: train_4_0_256 or train_162_256_768
        parts = base_name.split('_')
        if len(parts) < 3:
            print(f"⚠️ Cannot parse coords from {base_name}")
            continue

        try:
            # For patterns like train_4_0_256 → x=0, y=256
            # For patterns like train_162_256_768 → x=256, y=768
            if len(parts) == 4:  # train_162_256_768
                x_start = int(parts[-2])  # second-to-last number
                y_start = int(parts[-1])   # last number
            elif len(parts) == 3:  # train_4_0_256 (assuming x=0, y=last)
                x_start = int(parts[-2])  # second-to-last number (0)
                y_start = int(parts[-1])   # last number (256)
            else:
                print(f"⚠️ Unexpected format in {base_name}")
                continue
        except ValueError:
            print(f"⚠️ Invalid coords in {base_name}")
            continue

        # 🛰️ Find original T0 image for georeferencing
        t0_tif_path = Path(row['t0_npy']).parent.parent.parent / "train" / "A" / f"{Path(row['t0_npy']).stem}.tif"

        if not t0_tif_path.exists():
            t0_tif_path = t0_tif_path.with_suffix('.png')

        if not t0_tif_path.exists():
            print(f"⚠️ Missing source image: {t0_tif_path}")
            continue

        # Read geospatial metadata
        with rasterio.open(t0_tif_path) as src:
            crs = src.crs
            transform = src.transform
            pixel_size_x = transform.a
            pixel_size_y = abs(transform.e)
            bounds = src.bounds

        # Calculate top-left corner of this chip
        west = bounds.left + x_start * pixel_size_x
        north = bounds.top - y_start * pixel_size_y
        south = north - 256 * pixel_size_y
        east = west + 256 * pixel_size_x

        chip_transform = from_bounds(west, south, east, north, 256, 256)

        # Save GeoTIFF
        out_tif = output_path / f"{base_name}_change.tif"
        with rasterio.open(
            out_tif,
            'w',
            driver='GTiff',
            height=256,
            width=256,
            count=1,
            dtype=rasterio.uint8,
            crs=crs,
            transform=chip_transform,
            nodata=0,
        ) as dst:
            dst.write(bin_mask, 1)

        print(f"✅ Saved: {out_tif}")

    print("\n🎉 All available predictions exported to GeoTIFF!")


if __name__ == "__main__":
    main()