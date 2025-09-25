import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import random

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

def load_img(path: Path) -> np.ndarray:
    """Load an image and convert to RGB numpy array."""
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))

def load_mask(path: Path) -> np.ndarray:
    """Load a mask and convert to single channel numpy array."""
    with Image.open(path) as im:
        if im.mode != "L":
            im = im.convert("L")
        return np.array(im)

def save_npy(arr: np.ndarray, filepath: Path):
    """Save numpy array to .npy file, creating parent dirs if necessary."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(filepath), arr)

def tile_and_save_pair(a_img: np.ndarray, b_img: np.ndarray, mask: np.ndarray, prefix: str,
                       chip_size: int, stride: int, out_dir: Path, split: str):
    """
    Tiles large images and mask into chips, saves as .npy files,
    and returns list of dict rows with file paths and split info.
    """
    height, width, _ = a_img.shape
    rows = []
    for y in range(0, height - chip_size + 1, stride):
        for x in range(0, width - chip_size + 1, stride):
            a_chip = a_img[y:y+chip_size, x:x+chip_size, :]
            b_chip = b_img[y:y+chip_size, x:x+chip_size, :]
            mask_chip = mask[y:y+chip_size, x:x+chip_size]

            # Filter chips with very few positive pixels in the mask
            if np.sum(mask_chip > 0) < 5:
                continue

            base_filename = f"{prefix}_{y}_{x}.npy"
            a_fp = out_dir / split / "A" / base_filename
            b_fp = out_dir / split / "B" / base_filename
            label_fp = out_dir / split / "label" / base_filename

            # Save chips in CHW format needed for training (channels first)
            save_npy(np.transpose(a_chip, (2, 0, 1)), a_fp)
            save_npy(np.transpose(b_chip, (2, 0, 1)), b_fp)
            save_npy(mask_chip, label_fp)

            rows.append({
                "t0_npy": str(a_fp),
                "t1_npy": str(b_fp),
                "mask_npy": str(label_fp),
                "split": split
            })
    return rows

def main():
    parser = argparse.ArgumentParser(description="Tile paired images and masks into chips for change detection.")
    parser.add_argument("--a_dir", type=Path, required=True, help="Path to before images directory")
    parser.add_argument("--b_dir", type=Path, required=True, help="Path to after images directory")
    parser.add_argument("--label_dir", type=Path, required=True, help="Path to mask images directory")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for chips and index")
    parser.add_argument("--chip_size", type=int, default=256, help="Chip (tile) size")
    parser.add_argument("--stride", type=int, default=256, help="Stride for tiling")
    parser.add_argument("--val_frac", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--test_frac", type=float, default=0.15, help="Test fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    a_files = sorted([p for p in args.a_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    b_files = sorted([p for p in args.b_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    label_files = sorted([p for p in args.label_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

    assert len(a_files) == len(b_files) == len(label_files), "Mismatch in number of A, B, or label files."

    # Pair files and shuffle for split
    paired_files = list(zip(a_files, b_files, label_files))
    random.seed(args.seed)
    random.shuffle(paired_files)

    n = len(paired_files)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    n_train = n - n_val - n_test

    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

    all_rows = []
    for (a_fp, b_fp, label_fp), split in zip(paired_files, splits):
        prefix = a_fp.stem
        a_img = load_img(a_fp)
        b_img = load_img(b_fp)
        mask = load_mask(label_fp)
        rows = tile_and_save_pair(a_img, b_img, mask, prefix, args.chip_size, args.stride, args.out_dir, split)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    index_fp = args.out_dir / "index.parquet"
    df.to_parquet(str(index_fp), index=False)

    print(f"Saved {len(all_rows)} chips metadata rows to {index_fp}")

if __name__ == "__main__":
    main()




'''

  
  
  python scripts/make_chips_pair.py ^
  --a_dir data/LEVIR_CD/train/A ^
  --b_dir data/LEVIR_CD/train/B ^
  --label_dir data/LEVIR_CD/train/label ^
  --out_dir data/LEVIR_CD/chips_256 ^
  --chip_size 256 ^
  --stride 256



python -m scripts.refined_chip_generator ^
  --a_dir data\LEVIR_CD\train\A ^
  --b_dir data\LEVIR_CD\train\B ^
  --label_dir data\LEVIR_CD\train\label ^
  --out_dir data\LEVIR_CD\chips_256 ^
  --chip_size 256 ^
  --stride 256 ^
  --val_frac 0.15 ^
  --test_frac 0.15 ^
  --seed 42
  
  
  2. 
  python -m scripts.refined_chip_generator ^
  --a_dir data\LEVIR_CD\train\A ^
  --b_dir data\LEVIR_CD\train\B ^
  --label_dir data\LEVIR_CD\train\label ^
  --out_dir data\LEVIR_CD\chips_256 ^
  --chip_size 256 ^
  --stride 256 ^
  --val_frac 0.15 ^
  --test_frac 0.15 ^
  --seed 42
'''