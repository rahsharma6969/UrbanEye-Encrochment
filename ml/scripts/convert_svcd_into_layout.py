# ml/scripts/convert_svcd_into_layout.py
from pathlib import Path
import shutil

# Source base where your _A/_B/_OUT files are nested
SRC_BASE = Path('ml/data/SVCD/Real/original')
DST_ROOT = Path('ml/data/SVCD/train')

DST_A = DST_ROOT / 'A'
DST_B = DST_ROOT / 'B'
DST_L = DST_ROOT / 'label'
for p in (DST_A, DST_B, DST_L):
    p.mkdir(parents=True, exist_ok=True)

count = 0
# look into all nested dirs under SRC_BASE
for sub in sorted(SRC_BASE.rglob('*')):
    if not sub.is_dir():
        continue

# find all *_A.* anywhere under SRC_BASE
for a in sorted(SRC_BASE.rglob('*_A.*')):
    prefix = a.name.rsplit('_A', 1)[0]  # e.g. "1" from "1_A.bmp"
    ext = a.suffix                         # .bmp
    b = a.with_name(f"{prefix}_B{ext}")
    out = a.with_name(f"{prefix}_OUT{ext}")

    # If not present in same dir, try sibling dirs (safer)
    if not b.exists():
        # try same parent with other naming / case variants
        parent = a.parent
        b = None
        for cand in parent.glob(f"{prefix}_B*"):
            b = cand
            break
    if not out.exists():
        parent = a.parent
        out = None
        for cand in parent.glob(f"{prefix}_OUT*"):
            out = cand
            break

    if b is None or out is None:
        print("Skipping (missing pair):", a)
        continue

    # copy as prefix.ext (remove _A suffix) so filenames match across folders
    dst_a = DST_A / f"{prefix}{ext}"
    dst_b = DST_B / f"{prefix}{ext}"
    dst_l = DST_L / f"{prefix}{ext}"
    shutil.copy2(a, dst_a)
    shutil.copy2(b, dst_b)
    shutil.copy2(out, dst_l)
    count += 1

print(f"Done. Copied {count} samples to {DST_ROOT}")
