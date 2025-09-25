# ml/scripts/convert_labels_to_classids.py
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

ROOT = Path('C:/Users/Rahul/CollegeProject/UrbanEyeML/ml/data/SVCD')    # original big masks location (train/val/test/label)
OUT_NPY = Path('C:/Users/Rahul/CollegeProject/UrbanEyeML/ml/data/chips_256/npy')  # where npy mask files will go (same layout as before)
OUT_NPY.mkdir(parents=True, exist_ok=True)

# 1) Collect all unique RGB colors across all label images
colors = {}
for split in ['train','val','test']:
    lbl_dir = ROOT / split / 'label'
    if not lbl_dir.exists():
        continue
    for p in sorted(lbl_dir.glob('*')):
        img = Image.open(p).convert('RGB')
        arr = np.array(img)
        # collect unique colors in this image
        resh = arr.reshape(-1, 3)
        for rgb in np.unique(resh, axis=0):
            colors[tuple(rgb)] = colors.get(tuple(rgb), 0)  # placeholder

# sort colors to make deterministic mapping
sorted_colors = sorted(colors.keys())
color2id = {c: i for i, c in enumerate(sorted_colors)}
print("Found", len(color2id), "unique colors -> classes")

# Save mapping for reference
pd.Series({str(k): v for k, v in color2id.items()}).to_csv(OUT_NPY / 'color2id.csv')

# 2) Convert and save as npy (and optionally png)
rows = []
for split in ['train','val','test']:
    a_dir = Path('C:/Users/Rahul/CollegeProject/UrbanEyeML/ml/data/chips_256') / split / 'A'
    b_dir = Path('C:/Users/Rahul/CollegeProject/UrbanEyeML/ml/data/chips_256') / split / 'B'
    l_dir = Path('C:/Users/Rahul/CollegeProject/UrbanEyeML/ml/data/SVCD') / split / 'label'  # original label big images, or chips if already tiled
    # We'll try to find label chips under chips_256 first; fallback to big originals under SVCD
    if (Path('C:/Users/Rahul/CollegeProject/UrbanEyeML/ml/data/chips_256') / split / 'label').exists():
        l_dir = Path('C:/Users/Rahul/CollegeProject/UrbanEyeML/ml/data/chips_256') / split / 'label'
    out_a = OUT_NPY / split / 'A'; out_b = OUT_NPY / split / 'B'; out_l = OUT_NPY / split / 'label'
    out_a.mkdir(parents=True, exist_ok=True)
    out_b.mkdir(parents=True, exist_ok=True)
    out_l.mkdir(parents=True, exist_ok=True)

    # iterate A chips and find matching B and label by filename
    a_chips = sorted((Path('C:/Users/Rahul/CollegeProject/UrbanEyeML/ml/data/chips_256') / split / 'A').glob('*')) if (Path('ml/data/chips_256') / split / 'A').exists() else []
    for a in a_chips:
        b = Path('C:/Users/Rahul/CollegeProject/UrbanEyeML/ml/data/chips_256') / split / 'B' / a.name
        # try label in chips folder first, else in SVCD original folder (matching base name)
        l_candidates = [
            Path('C:/Users/Rahul/CollegeProject/UrbanEyeML/ml/data/chips_256') / split / 'label' / a.name,
            Path('C:/Users/Rahul/CollegeProject/UrbanEyeML/ml/data/SVCD') / split / 'label' / a.name,
        ]
        l = None
        for cand in l_candidates:
            if cand.exists(): 
                l = cand
                break
        if not b.exists() or not l:
            print("Skipping incomplete:", a)
            continue

        # load A/B and save numpy arrays
        A_img = np.array(Image.open(a)).astype(np.float32) / 255.0
        B_img = np.array(Image.open(b)).astype(np.float32) / 255.0
        # transpose to CHW
        A_arr = np.transpose(A_img, (2,0,1))
        B_arr = np.transpose(B_img, (2,0,1))

        # load label RGB and convert colors to ids
        lab_img = np.array(Image.open(l).convert('RGB'))
        h,w = lab_img.shape[:2]
        lab_ids = np.zeros((h,w), dtype=np.uint8)
        for rgb, idx in color2id.items():
            mask = (lab_img[...,0]==rgb[0]) & (lab_img[...,1]==rgb[1]) & (lab_img[...,2]==rgb[2])
            lab_ids[mask] = idx

        # save npy
        stem = a.stem
        np.save(out_a / f"{stem}.npy", A_arr)
        np.save(out_b / f"{stem}.npy", B_arr)
        np.save(out_l / f"{stem}.npy", lab_ids)

        rows.append({'t0_npy': str(out_a / f"{stem}.npy"), 't1_npy': str(out_b / f"{stem}.npy"), 'mask_npy': str(out_l / f"{stem}.npy"), 'split': split})

# 3) write parquet index
import pandas as pd
if rows:
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_NPY / 'index.parquet', index=False)
    print("Wrote index:", OUT_NPY / 'index.parquet', "rows:", len(df))
else:
    print("No rows written - check paths")
