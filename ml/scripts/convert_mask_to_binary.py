# scripts/convert_masks_to_binary.py
"""
Convert mask files referenced by an index.parquet to binary masks.
- Any pixel == 255 -> stays 255 (ignore)
- Any pixel > 0 and !=255 -> becomes 1
- Any pixel == 0 -> stays 0

Writes new mask files under a sibling directory 'binary' and produces a new parquet:
    <original_stem>_binary.parquet

Usage:
python -m scripts.convert_masks_to_binary --index data/chips_256/npy/index.parquet --out_dir data/chips_256/npy/binary
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def convert_mask_array(arr):
    # handle HxW or HxWx1
    if arr.ndim == 3:
        arr = arr[..., 0]
    out = np.array(arr, copy=True)
    IGNORE = 255
    # keep IGNORE as-is
    mask_ignore = (out == IGNORE)
    # positive = any non-zero and not ignore
    pos = (out != 0) & (~mask_ignore)
    out[:] = 0
    out[pos] = 1
    out[mask_ignore] = IGNORE
    return out.astype(np.uint8)

def main(index_path, out_dir, overwrite=False):
    index_path = Path(index_path)
    df = pd.read_parquet(index_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    new_rows = []
    for row in tqdm(df.itertuples(), total=len(df)):
        old_mask = Path(row.mask_npy)
        old_mask_arr = np.load(old_mask)
        new_arr = convert_mask_array(old_mask_arr)
        # build new filename
        new_name = f"{old_mask.stem}_bin.npy"
        new_path = out_dir / new_name
        # if overwrite False don't overwrite existing new file
        np.save(new_path, new_arr)
        # produce new row copying other columns
        new_row = {
            "t0_npy": row.t0_npy,
            "t1_npy": row.t1_npy,
            "mask_npy": str(new_path),
            "split": row.split if "split" in row._fields else "train"
        }
        new_rows.append(new_row)

    out_df = pd.DataFrame(new_rows)
    out_index_path = index_path.with_name(index_path.stem + "_binary.parquet")
    out_df.to_parquet(out_index_path, index=False)
    print("Wrote converted index:", out_index_path)
    print("New mask dir:", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out_dir", required=True, help="Directory to save binary mask .npy files")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    main(args.index, args.out_dir, args.overwrite)
