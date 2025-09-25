# scripts/mask_to_binary_backup.py
import numpy as np
from pathlib import Path
import shutil

def process_mask_file(p: Path, ignore_val=255):
    arr = np.load(p)
    # keep 255 untouched, map others >0 to 1, zeros remain 0
    out = np.where(arr == ignore_val, ignore_val, (arr > 0).astype(np.uint8))
    if not np.array_equal(arr, out):
        bak = p.with_suffix(p.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(p, bak)
        np.save(p, out)
        return True
    return False

def main(root="data/chips_256/npy"):
    root = Path(root)
    paths = list(root.rglob("*.npy"))
    print(f"Found {len(paths)} .npy files")
    changed = 0
    for p in paths:
        # only process label folders typically under .../label/ or .../train/label
        if "/label/" in str(p).replace("\\", "/") or p.name.lower().startswith("label") or "label" in p.parent.name.lower():
            if process_mask_file(p):
                print("Updated:", p)
                changed += 1
    print("Done. changed:", changed)

if __name__ == "__main__":
    main()
