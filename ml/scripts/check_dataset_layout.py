# ml/scripts/check_dataset_layout.py
from pathlib import Path
from PIL import Image
import numpy as np

def check_split(split_path):
    p = Path(split_path)
    print(f"\nChecking {p}")
    a = sorted((p/'A').glob('*'))
    b = sorted((p/'B').glob('*'))
    lab = sorted((p/'label').glob('*'))
    print(" counts: A, B, label ->", len(a), len(b), len(lab))
    if a:
        ai = np.array(Image.open(a[0]))
        bi = np.array(Image.open(b[0])) if (p/'B').exists() else None
        li = np.array(Image.open(lab[0])) if lab else None
        print(" example files:", a[0].name, b[0].name if b else None, lab[0].name if lab else None)
        print(" shapes: A", ai.shape, "B", bi.shape if bi is not None else None, "LABEL", li.shape if li is not None else None)
        print(" label unique values (first 10):", np.unique(li)[:10] if li is not None else None)

root = Path('C:/Users/Rahul/CollegeProject/UrbanEyeML/ml/data/SVCD')  # adjust if your path differs
for split in ['train','val','test']:
    p = root / split
    if p.exists():
        check_split(p)
    else:
        print(f"Missing split: {p}")
