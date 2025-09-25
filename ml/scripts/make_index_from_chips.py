# ml/scripts/make_index_from_chips.py
from pathlib import Path
import pandas as pd

CHIPS_ROOT = Path('C:/Users/Rahul/CollegeProject/UrbanEyeML/ml/data/chips_256')
rows = []
for split in ['train','val','test']:
    a_dir = CHIPS_ROOT / split / 'A'
    b_dir = CHIPS_ROOT / split / 'B'
    l_dir = CHIPS_ROOT / split / 'label'
    if not a_dir.exists():
        continue
    for a in sorted(a_dir.glob('*')):
        b = b_dir / a.name
        l = l_dir / a.name
        if b.exists() and l.exists():
            rows.append({'t0_npy': str(a), 't1_npy': str(b), 'mask_npy': str(l), 'split': split})
        else:
            print("Skipping incomplete:", a)
df = pd.DataFrame(rows)
out = CHIPS_ROOT / 'index.parquet'
df.to_parquet(out, index=False)
print("Index saved:", out, "rows:", len(df))
