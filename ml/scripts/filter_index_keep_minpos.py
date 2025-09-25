# scripts/filter_index_keep_minpos.py
"""
Filter index.parquet to keep only chips with >= min_pos positive pixels.
Usage:
    python -m scripts.filter_index_keep_minpos --index data/chips_256/npy/index.parquet --min_pos 20
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def main(index_path, min_pos):
    df = pd.read_parquet(index_path)
    kept = []
    for _, r in df.iterrows():
        mpath = Path(r.mask_npy)
        arr = np.load(mpath)
        if arr.ndim == 3:  # handle (H,W,1) or (H,W,3)
            arr = arr[..., 0]
        pos = int((arr == 1).sum())
        if pos >= min_pos:
            kept.append(r)

    out_df = pd.DataFrame(kept)
    out_path = Path(index_path).with_name(
        Path(index_path).stem + f"_filtered_minpos_{min_pos}.parquet"
    )
    out_df.to_parquet(out_path, index=False)

    print(f"Wrote {out_path} kept {len(out_df)} from {len(df)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Path to input index.parquet")
    ap.add_argument("--min_pos", type=int, required=True, help="Minimum number of positive pixels to keep a chip")
    args = ap.parse_args()
    main(args.index, args.min_pos)
'''
python -m scripts.filter_index_keep_minpos data/chips_256/npy/index.parquet 20

'''