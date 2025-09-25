# inspect_labels_and_chips.py
import os, numpy as np, pandas as pd
from pathlib import Path

LABEL_DIR = Path("data/labels/multiclass")
CHIPS_DIR = Path("data/chips_256")   # change if needed
PARQUET = Path("outputs/mumbai_index.parquet")  # or your parquet

# labels
lbl_files = sorted(LABEL_DIR.glob("*_label.npy"))
print(f"Found {len(lbl_files)} label files in {LABEL_DIR}")

val_counts = {}
shapes = {}
all_uniques = set()
for p in lbl_files:
    a = np.load(p)
    shapes.setdefault(a.shape, 0)
    shapes[a.shape] += 1
    uniq = np.unique(a)
    for u in uniq: all_uniques.add(int(u))
    for u in uniq:
        val_counts[int(u)] = val_counts.get(int(u), 0) + 1

print("Label shapes distribution:", shapes)
print("Label unique values (across files):", sorted(all_uniques))
print("Per-value counts (approx, files containing value):", val_counts)

# check parquet -> chip existence
if PARQUET.exists():
    df = pd.read_parquet(PARQUET)
    n = len(df)
    missing = 0
    for _, r in df.iterrows():
        if not Path(r.t0_npy).exists() or not Path(r.t1_npy).exists():
            missing += 1
    print(f"Parquet {PARQUET} rows: {n}, missing chip files: {missing}")
else:
    print(f"Parquet file {PARQUET} not found")

# sample a few label files to inspect visually (counts)
print("\nSample label file previews (first 5):")
for p in lbl_files[:5]:
    a = np.load(p)
    print(p.name, "shape", a.shape, "unique", np.unique(a))
