# ml/scripts/create_stratified_val.py
from pathlib import Path
import numpy as np
import pandas as pd
import random

INDEX = Path("data/chips_256/npy/index.parquet")
if not INDEX.exists():
    raise SystemExit(f"Index not found: {INDEX}")

df = pd.read_parquet(INDEX)
print("Total rows:", len(df))
print("Original split counts:")
print(df['split'].value_counts(dropna=False))

# Ensure split column exists
if 'split' not in df.columns:
    df['split'] = 'train'

# Identify positive and negative samples within TRAIN
train_df = df[df.split == 'train'].reset_index()
if train_df.empty:
    raise SystemExit("No train rows found to sample from. Aborting.")

pos_idx = []
neg_idx = []
for i, row in train_df.iterrows():
    mask_p = Path(row['mask_npy'])
    try:
        arr = np.load(mask_p)
    except Exception as e:
        print("Failed to load mask:", mask_p, e)
        continue
    # positive if any 1 (and not ignore 255)
    if ((arr == 1) & (arr != 255)).any():
        pos_idx.append(row['index'])
    else:
        neg_idx.append(row['index'])

print("Found positive chips in train:", len(pos_idx))
print("Found negative chips in train:", len(neg_idx))

# Parameters: how many positives & negatives to put in val
N_POS_VAL = 50   # change as you want
N_NEG_VAL = 50   # change as you want (or keep small)

if len(pos_idx) == 0:
    raise SystemExit("No positive chips found in train. Can't build stratified val. Consider adjusting remap / index.")

# sample (clip if not enough)
npos = min(len(pos_idx), N_POS_VAL)
npos_samples = random.Random(42).sample(pos_idx, npos)

nneg = min(len(neg_idx), N_NEG_VAL)
nneg_samples = random.Random(42).sample(neg_idx, nneg)

val_indices = set(npos_samples + nneg_samples)
print(f"Selecting {len(npos_samples)} positive + {len(nneg_samples)} negative rows -> val ({len(val_indices)} total)")

# Move these rows to val in df (df index equals original parquet index)
df.loc[df.index.isin(val_indices), 'split'] = 'val'

# Save updated parquet (overwrite)
df.to_parquet(INDEX, index=False)
print("Updated index written to:", INDEX)
print("New split counts:")
print(df['split'].value_counts())
