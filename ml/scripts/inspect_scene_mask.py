# scripts/inspect_scene_mask_recursive.py
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import csv

def load_arr(p: Path):
    # support .png/.bmp/.jpg/.npy
    if p.suffix.lower() == ".npy":
        return np.load(p)
    else:
        im = Image.open(p)
        arr = np.array(im)
        return arr

def analyze_folder(folder: Path, out_csv: Path=None):
    imgs = sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".png",".bmp",".jpg",".jpeg",".npy"}])
    if not imgs:
        print("No mask/image files found under:", folder)
        return

    total = 0
    with_change = 0
    with_ignore = 0
    empty_files = []
    counts = []

    for p in imgs:
        total += 1
        try:
            arr = load_arr(p)
        except Exception as e:
            print(f"Failed to load {p}: {e}")
            continue

        # normalize shapes: if HxWx3 pick first channel if needed
        if arr.ndim == 3:
            if arr.shape[2] == 3 or arr.shape[2] == 4:
                m = arr[...,0]
            elif arr.shape[2] == 1:
                m = arr[...,0]
            else:
                m = arr[...,0]
        else:
            m = arr

        # ensure integer-like
        m = np.asarray(m)

        n_pos = int((m == 1).sum())
        n_ignore = int((m == 255).sum())
        n_valid = int((m != 255).sum())
        has_pos = n_pos > 0

        counts.append({"path": str(p), "pos_pixels": n_pos, "ignore_pixels": n_ignore, "valid_pixels": n_valid, "has_pos": has_pos})
        if has_pos:
            with_change += 1
        if n_ignore > 0:
            with_ignore += 1
        if n_pos == 0:
            empty_files.append(str(p))

    # summary
    print("Folder:", folder)
    print("Total files inspected:", total)
    print("Files with any change (value==1):", with_change)
    print("Files containing ignore value 255:", with_ignore)
    print("Empty (no positive pixels) count:", len(empty_files))
    if len(empty_files) > 0:
        print("First 10 empty files:")
        for f in empty_files[:10]:
            print("  ", f)

    # optional CSV
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["path","pos_pixels","ignore_pixels","valid_pixels","has_pos"])
            writer.writeheader()
            for r in counts:
                writer.writerow(r)
        print("Wrote CSV:", out_csv)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_scene_mask_recursive.py <folder> [out.csv]")
        sys.exit(1)
    folder = Path(sys.argv[1])
    out_csv = Path(sys.argv[2]) if len(sys.argv) >= 3 else None
    if not folder.exists():
        print("Folder not found:", folder)
        sys.exit(1)
    analyze_folder(folder, out_csv)

'''
python scripts/inspect_scene_mask.py ^
  data/chips_256/original/scene1_OUT.png ^
  data/chips_256/original/scene2_OUT.png



python -m scripts.inspect_scene_mask data/SVCD/Real/original/with_add_object outputs/logs/svcd_with_add_object_mask_report.csv
'''