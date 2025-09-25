import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import random

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

def load_img(path: Path) -> np.ndarray:
    """Load an image and convert to RGB numpy array [0,1]."""
    with Image.open(path) as im:
        return np.array(im.convert("RGB"), dtype=np.float32) / 255.0

def load_mask(path: Path) -> np.ndarray:
    """Load mask and convert to binary {0,1} (255 → 1, 0 → 0)."""
    with Image.open(path) as im:
        if im.mode != "L":
            im = im.convert("L")
        mask = np.array(im, dtype=np.uint8)
        
        # Debug: Print unique values for first few masks
        unique_vals = np.unique(mask)
        if len(unique_vals) > 2:
            print(f"⚠️ Mask {path.name} has values: {unique_vals} - converting non-zero to 1")
        
        # Convert any non-zero pixel to 1 (change)
        return (mask > 0).astype(np.uint8)

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
    chips_saved = 0
    chips_skipped = 0
    
    for y in range(0, height - chip_size + 1, stride):
        for x in range(0, width - chip_size + 1, stride):
            a_chip = a_img[y:y+chip_size, x:x+chip_size, :]
            b_chip = b_img[y:y+chip_size, x:x+chip_size, :]
            mask_chip = mask[y:y+chip_size, x:x+chip_size]

            # Filter chips with too few change pixels
            n_change = np.sum(mask_chip == 1)
            if n_change < 5:
                chips_skipped += 1
                continue

            # Filter chips with too much change (likely mislabeled)
            if n_change > chip_size * chip_size * 0.9:
                chips_skipped += 1
                continue

            base_filename = f"{prefix}_{y}_{x}.npy"
            a_fp = out_dir / split / "A" / base_filename
            b_fp = out_dir / split / "B" / base_filename
            label_fp = out_dir / split / "label" / base_filename

            # Save in CHW format: (C, H, W)
            save_npy(np.transpose(a_chip, (2, 0, 1)), a_fp)
            save_npy(np.transpose(b_chip, (2, 0, 1)), b_fp)
            save_npy(mask_chip, label_fp)

            # Use relative paths from out_dir (simpler approach)
            a_rel = Path(split) / "A" / base_filename
            b_rel = Path(split) / "B" / base_filename
            label_rel = Path(split) / "label" / base_filename

            rows.append({
                "t0_npy": str(a_rel),
                "t1_npy": str(b_rel),
                "mask_npy": str(label_rel),
                "split": split
            })
            chips_saved += 1
    
    print(f"   💾 Saved: {chips_saved} chips, ⏭️ Skipped: {chips_skipped} chips")
    return rows

def main():
    parser = argparse.ArgumentParser(description="Tile paired images and masks into chips for change detection.")
    parser.add_argument("--a_dir", type=Path, required=True, help="Path to before images directory (T0)")
    parser.add_argument("--b_dir", type=Path, required=True, help="Path to after images directory (T1)")
    parser.add_argument("--label_dir", type=Path, required=True, help="Path to mask images directory (0/255)")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for chips and index.parquet")
    parser.add_argument("--chip_size", type=int, default=256, help="Chip (tile) size")
    parser.add_argument("--stride", type=int, default=256, help="Stride for tiling (non-overlapping)")
    parser.add_argument("--val_frac", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--test_frac", type=float, default=0.15, help="Test fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    # Validate input directories
    if not args.a_dir.exists() or not args.b_dir.exists() or not args.label_dir.exists():
        raise FileNotFoundError(f"Input directories missing:\n"
                              f"A: {args.a_dir} exists={args.a_dir.exists()}\n"
                              f"B: {args.b_dir} exists={args.b_dir.exists()}\n"
                              f"Label: {args.label_dir} exists={args.label_dir.exists()}")

    # Get all image files
    a_files = sorted([p for p in args.a_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    b_files = sorted([p for p in args.b_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    label_files = sorted([p for p in args.label_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

    # Debug file counts
    print(f"📁 A files: {len(a_files)}")
    print(f"📁 B files: {len(b_files)}")
    print(f"📁 Label files: {len(label_files)}")

    assert len(a_files) == len(b_files) == len(label_files), \
        f"Mismatch in number of files: A={len(a_files)}, B={len(b_files)}, Label={len(label_files)}"

    print(f"✅ Found {len(a_files)} paired image-mask sets.")

    # Pair files and shuffle for split
    paired_files = list(zip(a_files, b_files, label_files))
    random.seed(args.seed)
    random.shuffle(paired_files)

    n = len(paired_files)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    n_train = n - n_val - n_test

    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

    print(f"📊 Split: train={n_train}, val={n_val}, test={n_test}")

    all_rows = []
    total_chips = 0

    for i, ((a_fp, b_fp, label_fp), split) in enumerate(zip(paired_files, splits)):
        prefix = a_fp.stem
        print(f"[{i+1}/{n}] Processing {prefix} ({split})...")

        try:
            a_img = load_img(a_fp)
            b_img = load_img(b_fp)
            mask = load_mask(label_fp)  # ✅ Now returns {0,1}

            # Validate shapes
            if a_img.shape[:2] != b_img.shape[:2] or a_img.shape[:2] != mask.shape:
                print(f"⚠️ Shape mismatch in {prefix}: A={a_img.shape}, B={b_img.shape}, Mask={mask.shape}")
                continue

            # Check if mask has any change pixels
            n_change_total = np.sum(mask == 1)
            if n_change_total == 0:
                print(f"⚠️ No change pixels in {prefix}, skipping...")
                continue

            rows = tile_and_save_pair(a_img, b_img, mask, prefix, args.chip_size, args.stride, 
                                     args.out_dir, split)
            all_rows.extend(rows)
            total_chips += len(rows)

        except Exception as e:
            print(f"❌ Error processing {prefix}: {e}")
            continue

    # Save index
    if all_rows:
        df = pd.DataFrame(all_rows)
        index_fp = args.out_dir / "index.parquet"
        df.to_parquet(index_fp, index=False)

        print(f"\n🎉 COMPLETE!")
        print(f"✅ Total valid chips generated: {total_chips:,}")
        print(f"📊 Saved metadata to: {index_fp}")
        print(f"📁 Output directory: {args.out_dir}")

        # Print split summary
        split_counts = df['split'].value_counts()
        print("\n--- SPLIT SUMMARY ---")
        for split, count in split_counts.items():
            pct = count / len(df) * 100
            print(f"  {split:>5}: {count:>6} chips ({pct:>5.1f}%)")
    else:
        print("❌ No valid chips generated! Check your input data.")

if __name__ == "__main__":
    main()