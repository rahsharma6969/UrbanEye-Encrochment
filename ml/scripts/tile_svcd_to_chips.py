# ml/scripts/tile_svcd_to_chips.py
from pathlib import Path
from PIL import Image
import numpy as np

SRC_BASE = Path('ml/data/SVCD')   # has train/val/test
OUT_BASE = Path('ml/data/chips_256')  # target chips dir
PATCH = 256
STRIDE = 256  # set <PATCH for overlap, use 192 for 25% overlap

for split in ['train','val','test']:
    for sub in ['A','B','label']:
        (OUT_BASE/split/sub).mkdir(parents=True, exist_ok=True)

    src_split = SRC_BASE / split
    if not src_split.exists(): 
        continue

    a_list = sorted((src_split/'A').glob('*'))
    for a_path in a_list:
        prefix = a_path.stem  # e.g., "1"
        b_path = src_split/'B'/a_path.name
        l_path = src_split/'label'/a_path.name
        if not b_path.exists() or not l_path.exists():
            print("Skipping incomplete:", a_path)
            continue

        a_img = np.array(Image.open(a_path))
        b_img = np.array(Image.open(b_path))
        l_img = np.array(Image.open(l_path))

        H, W = a_img.shape[:2]
        idx=0
        for y in range(0, H-PATCH+1, STRIDE):
            for x in range(0, W-PATCH+1, STRIDE):
                a_chip = a_img[y:y+PATCH, x:x+PATCH]
                b_chip = b_img[y:y+PATCH, x:x+PATCH]
                l_chip = l_img[y:y+PATCH, x:x+PATCH]
                name = f"{prefix}_{y}_{x}.png"
                Image.fromarray(a_chip).save(OUT_BASE/split/'A'/name)
                Image.fromarray(b_chip).save(OUT_BASE/split/'B'/name)
                Image.fromarray(l_chip).save(OUT_BASE/split/'label'/name)
                idx += 1
        # handle right/bottom edge (pad or skip). For simplicity, skip partial patches now.
        print(f"Tiled {a_path.name} -> {idx} chips")
print("Done.")
