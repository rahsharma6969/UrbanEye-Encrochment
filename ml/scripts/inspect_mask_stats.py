# scripts/inspect_mask_stats.py
import sys, numpy as np
from pathlib import Path
p = Path(sys.argv[1])
files = list(p.rglob("*.npy")) + list(p.rglob("*.png")) + list(p.rglob("*.bmp"))
out=[]
for f in files:
    try:
        if f.suffix=='.npy':
            arr = np.load(f)
        else:
            from PIL import Image
            arr = np.array(Image.open(f))
        if arr.ndim==3: arr = arr[...,0]
        total = arr.size
        npos = int((arr==1).sum())
        nign = int((arr==255).sum())
        out.append((str(f), total, npos, nign))
    except Exception as e:
        out.append((str(f), "ERR", str(e)))
print("file, total, npos, nignore")
for r in out:
    print(*r)
