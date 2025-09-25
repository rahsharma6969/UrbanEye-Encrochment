# scripts/inspect_index_labels.py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--sample", type=int, default=10)
args = ap.parse_args()

df = pd.read_parquet(args.index)
counter = Counter()
bad_samples = []
for i, row in enumerate(df.itertuples()):
    p = Path(row.mask_npy)
    arr = np.load(p)
    if arr.ndim == 3:
        arr = arr[..., 0]
    unique = np.unique(arr)
    for v in unique:
        counter[int(v)] += 1
    if len([v for v in unique if v not in (0,1,255)])>0 and len(bad_samples) < args.sample:
        bad_samples.append((str(p), unique.tolist()))
print("Top label counts:", counter.most_common(20))
print("Examples with unexpected labels (first {}):".format(len(bad_samples)))
for p, u in bad_samples:
    print(p, u)
