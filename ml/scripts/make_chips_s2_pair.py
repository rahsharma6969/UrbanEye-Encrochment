#!/usr/bin/env python3
"""
make_chips_s2_pairs.py

Tile paired before/after images + mask into chips, filter empty/poisoned chips,
and produce an index.parquet suitable for the training pipeline.

Usage examples:
  python scripts/make_chips_s2_pair.py ^
  --pairs_dir data/SVCD/REAL ^
  --out_dir data/chips_256 ^
  --chip_size 256 ^
  --stride 128 ^
  --min_change_pixels 50 ^
  --max_ignore_ratio 0.2 ^
  --val_frac 0.10 --test_frac 0.10 ^
  --force_clean


  python ml/scripts/make_chips_s2_pairs.py \
    --t0_dir data/SVCD/A --t1_dir data/SVCD/B --mask_dir data/SVCD/OUT \
    --out_dir data/chips_256 --chip_size 256 --min_change_pixels 10
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import math
import random
import pandas as pd
import sys

# supported image extensions
IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"]

def img_load_as_array(p: Path):
    """Load image or numpy .npy -> returns H,W,C or H,W (for mask)"""
    if p.suffix.lower() == ".npy":
        arr = np.load(p)
        return arr
    else:
        # PIL: convert to RGB for multichannel images
        im = Image.open(p)
        # leave single-channel images as-is
        if im.mode == "L" or im.mode == "I":
            return np.array(im)
        else:
            return np.array(im.convert("RGB"))

def save_npy_chw(img_hwc: np.ndarray, outp: Path):
    """Convert HWC (uint8/float) -> CHW float32 in 0..1 and save .npy"""
    arr = img_hwc
    if arr.ndim == 2:  # single channel
        arr_chw = arr[np.newaxis, :, :].astype(np.float32)
    else:
        # ensure channels last
        arr_chw = np.transpose(arr, (2, 0, 1)).astype(np.float32)
    # if values appear to be 0..255 scale to 0..1
    if arr_chw.max() > 2.0:
        arr_chw = arr_chw / 255.0
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(outp), arr_chw.astype(np.float32))

def save_mask(mask_hw: np.ndarray, outp: Path):
    """Ensure mask is 2D integer and save .npy"""
    arr = mask_hw
    if arr.ndim == 3:
        # if single-channel in third dim, squeeze, else take first channel (best to convert beforehand)
        if arr.shape[2] == 1:
            arr = arr[..., 0]
        else:
            arr = arr[..., 0]
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(outp), arr.astype(np.int32))

def get_prefix(fname: str):
    # remove suffix like _A, _B, _OUT, _LABEL etc
    base = Path(fname).stem
    # try a few common separators
    for tag in ["_A", "_B", "_OUT", "_OUTLINE", "_OUTT", "_LABEL", "_mask", "_MASK", "_OUTER"]:
        if base.endswith(tag):
            return base[: -len(tag)]
    # fallback: split on last underscore and return left part
    if "_" in base:
        return "_".join(base.split("_")[:-1])
    return base

def find_pairs_in_dir(pairs_dir: Path):
    """
    Find triplets by scanning for files with suffixes _A / _B / _OUT (case-insensitive).
    Returns list of tuples (t0_path, t1_path, mask_path, prefix)
    """
    files = list(pairs_dir.rglob("*"))
    # group by prefix heuristics
    by_prefix = {}
    for f in files:
        if not f.is_file(): 
            continue
        if f.suffix.lower() not in IMG_EXTS:
            continue
        name = f.name
        # try to detect tag
        lowered = name.lower()
        if "_a" in lowered:
            tag = "_A"
            prefix = name.rsplit("_", 1)[0]
        elif "_b" in lowered:
            tag = "_B"
            prefix = name.rsplit("_", 1)[0]
        elif "_out" in lowered or "_label" in lowered or "mask" in lowered:
            tag = "_OUT"
            # try prefix
            # strip the trailing part after last underscore
            prefix = name.rsplit("_", 1)[0]
        else:
            # fallback: group by base without final numeric tile offset if possible
            prefix = get_prefix(name)
            tag = None
        by_prefix.setdefault(prefix, []).append((tag, f))
    triplets = []
    for prefix, lst in by_prefix.items():
        t0 = t1 = mask = None
        for tag, f in lst:
            low = f.name.lower()
            if "_a" in low and t0 is None:
                t0 = f
            elif "_b" in low and t1 is None:
                t1 = f
            elif ("_out" in low or "mask" in low or "label" in low) and mask is None:
                mask = f
        # fallback: if no tags found, try patterns with prefix_A.* etc
        if t0 is None:
            cand = next((x for (_, x) in lst if str(x).lower().endswith(("_a.png", "_a.jpg", "_a.bmp", "_a.tif", "_a.npy"))), None)
            if cand: t0 = cand
        if t1 is None:
            cand = next((x for (_, x) in lst if str(x).lower().endswith(("_b.png", "_b.jpg", "_b.bmp", "_b.tif", "_b.npy"))), None)
            if cand: t1 = cand
        if mask is None:
            cand = next((x for (_, x) in lst if ("out" in x.name.lower() or "mask" in x.name.lower() or "label" in x.name.lower())), None)
            if cand: mask = cand
        if t0 and t1 and mask:
            triplets.append((t0, t1, mask, prefix))
    return triplets

def tile_and_save_pair(t0_arr, t1_arr, mask_arr, out_root: Path, prefix: str, chip_size=256, stride=256,
                       min_change_pixels=1, max_ignore_ratio=0.2, split='train'):
    """
    tile arrays and save to out_root/{split}/A/, B/, label/
    returns list of rows
    """
    H, W = t0_arr.shape[0], t0_arr.shape[1]
    rows = []

    # If images have channel dim last -> H,W,C
    def ensure_hwc(a):
        if a.ndim == 3 and a.shape[2] not in (1,3,4):
            # maybe it's CHW; convert to HWC
            return np.transpose(a, (1,2,0))
        return a

    t0_arr = ensure_hwc(t0_arr)
    t1_arr = ensure_hwc(t1_arr)
    # if masks are CHW -> convert
    if mask_arr.ndim == 3 and mask_arr.shape[2] not in (1,):
        # take first channel
        mask_arr = mask_arr[..., 0]
    # tile grid
    ys = list(range(0, H - chip_size + 1, stride))
    xs = list(range(0, W - chip_size + 1, stride))
    # also include last tile aligning right/bottom if remainder exists
    if len(ys) == 0 and H >= chip_size:
        ys = [0]
    if len(xs) == 0 and W >= chip_size:
        xs = [0]
    if (ys and ys[-1] + chip_size < H):
        ys.append(H - chip_size)
    if (xs and xs[-1] + chip_size < W):
        xs.append(W - chip_size)

    cnt_saved = 0
    for yi in ys:
        for xi in xs:
            t0_crop = t0_arr[yi:yi+chip_size, xi:xi+chip_size].copy()
            t1_crop = t1_arr[yi:yi+chip_size, xi:xi+chip_size].copy()
            mask_crop = mask_arr[yi:yi+chip_size, xi:xi+chip_size].copy()

            # compute positive and ignore stats
            # positive considered value == 1
            pos = int((mask_crop == 1).sum())
            ignore = int((mask_crop == 255).sum()) if mask_crop.dtype != np.bool_ else 0
            valid = mask_crop.size - ignore
            # skip if too many ignore pixels
            if valid <= 0:
                continue
            if (ignore / mask_crop.size) > max_ignore_ratio:
                continue
            if pos < min_change_pixels:
                # skip background-only chips
                continue

            # prepare names
            tile_name = f"{prefix}_{yi}_{xi}.npy"
            out_a = out_root / split / "A" / tile_name
            out_b = out_root / split / "B" / tile_name
            out_label = out_root / split / "label" / tile_name

            save_npy_chw(t0_crop, out_a)
            save_npy_chw(t1_crop, out_b)
            save_mask(mask_crop, out_label)

            rows.append({"t0_npy": str(out_a), "t1_npy": str(out_b), "mask_npy": str(out_label), "split": split})
            cnt_saved += 1

    return rows

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", type=str, default=None,
                    help="Directory containing paired files named like prefix_A.png, prefix_B.png, prefix_OUT.png etc.")
    ap.add_argument("--t0_dir", type=str, default=None, help="Directory with t0 images (before)")
    ap.add_argument("--t1_dir", type=str, default=None, help="Directory with t1 images (after)")
    ap.add_argument("--mask_dir", type=str, default=None, help="Directory with mask images")
    ap.add_argument("--out_dir", type=str, required=True, help="Output root for chips (will create A/ B/ label under splits)")
    ap.add_argument("--chip_size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--min_change_pixels", type=int, default=1,
                    help="Minimum positive pixels (mask==1) required to keep a chip")
    ap.add_argument("--max_ignore_ratio", type=float, default=0.2,
                    help="Max fraction of pixels that can be ignore-index (255) before discarding chip")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_clean", action="store_true",
                    help="If set, remove out_dir first (be careful!)")
    args = ap.parse_args(argv)

    out_root = Path(args.out_dir).resolve()
    if args.force_clean:
        if out_root.exists():
            print("Removing existing out_dir:", out_root)
            import shutil
            shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # find triplets
    triplets = []
    if args.pairs_dir:
        pairs_dir = Path(args.pairs_dir).resolve()
        if not pairs_dir.exists():
            print("pairs_dir not found:", pairs_dir); sys.exit(1)
        triplets = find_pairs_in_dir(pairs_dir)
    else:
        if not (args.t0_dir and args.t1_dir and args.mask_dir):
            print("Either --pairs_dir or all of --t0_dir --t1_dir --mask_dir are required."); sys.exit(1)
        t0_files = {p.name: p for p in Path(args.t0_dir).rglob("*") if p.suffix.lower() in IMG_EXTS}
        t1_files = {p.name: p for p in Path(args.t1_dir).rglob("*") if p.suffix.lower() in IMG_EXTS}
        mask_files = {p.name: p for p in Path(args.mask_dir).rglob("*") if p.suffix.lower() in IMG_EXTS}
        # match by name base prefix heuristics
        # produce pairs where a matching file exists in each folder by prefix (lhs of last underscore)
        for fname, p0 in t0_files.items():
            base = get_prefix(fname)
            # look for t1 and mask with same base
            cand_t1 = next((v for k,v in t1_files.items() if get_prefix(k) == base), None)
            cand_mask = next((v for k,v in mask_files.items() if get_prefix(k) == base), None)
            if cand_t1 and cand_mask:
                triplets.append((p0, cand_t1, cand_mask, base))

    print(f"Found {len(triplets)} paired scenes to chip.")

    # prepare splits
    random.seed(args.seed)
    scenes = triplets.copy()
    random.shuffle(scenes)
    n = len(scenes)
    n_test = int(round(n * args.test_frac))
    n_val = int(round(n * args.val_frac))
    n_train = n - n_val - n_test
    splits = (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test)
    # align length
    if len(splits) < n:
        splits += ["train"] * (n - len(splits))

    all_rows = []
    for (t0p, t1p, maskp, prefix), split in zip(scenes, splits):
        print(f"Processing scene {prefix} -> split {split}")
        t0_arr = img_load_as_array(t0p)
        t1_arr = img_load_as_array(t1p)
        mask_arr = img_load_as_array(maskp)
        rows = tile_and_save_pair(t0_arr, t1_arr, mask_arr, out_root, prefix,
                                  chip_size=args.chip_size, stride=args.stride,
                                  min_change_pixels=args.min_change_pixels,
                                  max_ignore_ratio=args.max_ignore_ratio,
                                  split=split)
        print(f"  saved {len(rows)} chips from scene {prefix}")
        all_rows.extend(rows)

    print(f"Total saved chips: {len(all_rows)}")

    # write parquet index
    if len(all_rows) > 0:
        df = pd.DataFrame(all_rows)
        index_path = out_root / "npy" / "index.parquet"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(index_path, index=False)
        print("Wrote index:", index_path)
    else:
        print("No chips saved - nothing to write.")

if __name__ == "__main__":
    main()


'''
python -m scripts.make_chips_s2_pair --t0_dir data/SVCD/Real/original ^
                                    --t1_dir data/SVCD/Real/changed ^
                                    --mask_dir data/SVCD/Real/original ^
                                    --out_dir data/chips_256 ^
                                    --chip_size 256 ^
                                    --stride 128 ^
                                    --format npy ^
                                    --min_change_pixels 1 ^
                                    --val_frac 0.15 ^
                                    --test_frac 0.15 ^
                                    --force_clean
'''