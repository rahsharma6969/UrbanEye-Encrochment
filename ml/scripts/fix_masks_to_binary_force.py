# ml/scripts/fix_masks_to_binary_force.py
from pathlib import Path
import numpy as np
import pandas as pd

index_path = Path("data/chips_256/npy/index.parquet")
if not index_path.exists():
    raise SystemExit("Index parquet not found: " + str(index_path))

df = pd.read_parquet(index_path)
print("Rows:", len(df))

cnt_fixed = 0
bad_samples = []
for p in df["mask_npy"]:
    arr = np.load(p)
    unique = np.unique(arr)
    # if any value not in {0,1,255}, remap ALL non-zero (and not 255) -> 1
    bad = [v for v in unique if v not in (0,1,255)]
    if bad:
        out = np.full_like(arr, 255, dtype=np.uint8)
        out[arr == 0] = 0
        mask_nonzero = (arr != 0) & (arr != 255)
        out[mask_nonzero] = 1
        np.save(p, out)
        cnt_fixed += 1
        bad_samples.append((p, unique.tolist()))
print(f"Fixed {cnt_fixed} mask files (remapped non-zero -> 1).")
if cnt_fixed:
    print("Examples of files fixed and their original uniques (first 10):")
    for t in bad_samples[:10]:
        print(t)
