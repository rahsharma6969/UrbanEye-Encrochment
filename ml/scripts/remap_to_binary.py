# ml/scripts/remap_to_binary.py
from pathlib import Path
import numpy as np, pandas as pd

# index pointing to npy masks
index_path = Path('data/chips_256/npy/index.parquet')
if not index_path.exists():
    raise SystemExit(f"Index not found: {index_path}")

df = pd.read_parquet(index_path)
print("Rows in index:", len(df))

# Remap in-place: 0 -> 0 (background), non-zero -> 1 (change), preserve 255 as ignore
cnt = 0
for p in df['mask_npy']:
    arr = np.load(p)
    out = np.full_like(arr, 255, dtype=np.uint8)   # default to ignore
    out[arr == 0] = 0                              # background
    nonzero_mask = (arr != 0) & (arr != 255)
    out[nonzero_mask] = 1                          # change
    np.save(p, out)
    cnt += 1
print(f"Remapped {cnt} masks to binary (0 background, 1 change, 255 ignore).")
