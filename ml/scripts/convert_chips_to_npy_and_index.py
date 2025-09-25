# ml/scripts/convert_chips_to_npy_and_index.py
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import shutil

CHIPS_ROOT = Path('C:/Users/Rahul/CollegeProject/UrbanEyeML/ml/data/chips_256')   # input PNG chips: split/{A,B,label}
OUT_ROOT = CHIPS_ROOT / 'npy'           # output npy files here
OUT_ROOT.mkdir(parents=True, exist_ok=True)

rows = []
for split in ['train','val','test']:
    a_dir = CHIPS_ROOT / split / 'A'
    b_dir = CHIPS_ROOT / split / 'B'
    l_dir = CHIPS_ROOT / split / 'label'
    if not a_dir.exists():
        continue
    (OUT_ROOT / split / 'A').mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / split / 'B').mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / split / 'label').mkdir(parents=True, exist_ok=True)

    for a in sorted(a_dir.glob('*')):
        b = b_dir / a.name
        l = l_dir / a.name
        if not b.exists() or not l.exists():
            print("Skipping incomplete:", a)
            continue

        # load A
        A_img = np.array(Image.open(a))
        # ensure HWC
        if A_img.ndim == 2:
            A_img = A_img[..., None]
        # convert to float32 and scale to 0..1 (if you trained normalized differently adjust)
        A_arr = A_img.astype(np.float32) / 255.0
        # transpose HWC -> CHW
        A_arr = np.transpose(A_arr, (2,0,1))

        # load B
        B_img = np.array(Image.open(b))
        if B_img.ndim == 2:
            B_img = B_img[..., None]
        B_arr = B_img.astype(np.float32) / 255.0
        B_arr = np.transpose(B_arr, (2,0,1))

        # load label (single channel)
        L_img = np.array(Image.open(l))
        if L_img.ndim == 3:
            # assume R==G==B -> take first channel
            L_img = L_img[...,0]
        L_arr = L_img.astype(np.uint8)  # keep int class ids

        # save .npy files
        stem = a.stem  # name without extension
        A_path = OUT_ROOT / split / 'A' / f"{stem}.npy"
        B_path = OUT_ROOT / split / 'B' / f"{stem}.npy"
        L_path = OUT_ROOT / split / 'label' / f"{stem}.npy"
        np.save(str(A_path), A_arr)
        np.save(str(B_path), B_arr)
        np.save(str(L_path), L_arr)

        rows.append({'t0_npy': str(A_path), 't1_npy': str(B_path), 'mask_npy': str(L_path), 'split': split})

# write parquet index in the same npy folder
if rows:
    df = pd.DataFrame(rows)
    index_path = OUT_ROOT / 'index.parquet'
    df.to_parquet(index_path, index=False)
    print("Wrote index:", index_path, "rows:", len(df))
else:
    print("No samples converted.")
