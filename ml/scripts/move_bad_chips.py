# scripts/move_bad_chips.py
import os, shutil, pandas as pd
from pathlib import Path

SUMMARY = Path("outputs/chips_inspect_summary.csv")   # produced earlier
CHIPS_DIR = Path("data/chips_256")
BAD_DIR = CHIPS_DIR.parent / "chips_bad"
BAD_DIR.mkdir(exist_ok=True)

df = pd.read_csv(SUMMARY)
# choose threshold (you can change 0.5 -> 0.3 etc)
threshold = 0.5

bad = df[(df['valid_ratio'] < threshold) | (df['finite_count'] == 0)]
print("Bad chip files:", len(bad))

for p in bad['path'].tolist():
    src = Path(p)
    if src.exists():
        dst = BAD_DIR / src.name
        shutil.move(str(src), str(dst))
print("Moved bad chips to:", BAD_DIR)
