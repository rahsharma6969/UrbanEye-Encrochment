"""
Make tiles / chips from SVCD-style scenes (pair A,B and OUT mask).
Saves per-chip .npy arrays:
  t0: CHW float32 (3,H,W) scaled 0..1
  t1: CHW float32 (3,H,W) scaled 0..1
  mask: HxW uint8 with values {0,1,255} (255 ignored)
Writes index parquet at out_dir/npy/index.parquet listing t0_npy,t1_npy,mask_npy,split
Usage:
  python -m scripts.make_chips_from_scenes --scenes data/SVCD/Real/original --out data/chips_256/npy --tile 256 --stride 256 --val_frac 0.1 --test_frac 0.1
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import math
import random
import os

def find_scene_files(scenes_dir: Path):
    # Accept scenes where there exists scene_OUT + scene_A + scene_B in the same folder or subfolders.
    scenes = []
    for p in scenes_dir.rglob("*_OUT.*"):
        stem = p.name.rsplit("_OUT",1)[0]
        parent = p.parent
        a = parent / (stem + "_A" + p.suffix)
        b = parent / (stem + "_B" + p.suffix)
        if a.exists() and b.exists():
            scenes.append({"a": a, "b": b, "out": p})
    return scenes

def load_image_as_array(p: Path):
    im = Image.open(p)
    arr = np.array(im)
    # Accept HxW or HxWx3 — convert to HxWx3 if grayscale
    if arr.ndim == 2:
        arr = np.stack([arr,arr,arr], axis=-1)
    return arr

def save_chip(out_dir: Path, split: str, prefix: str, i, t0_arr, t1_arr, mask_arr):
    # Create folder structure out_dir/{split}/{A,B,label}
    base = out_dir / split
    a_dir = base / "A"
    b_dir = base / "B"
    l_dir = base / "label"
    a_dir.mkdir(parents=True, exist_ok=True)
    b_dir.mkdir(parents=True, exist_ok=True)
    l_dir.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}_{i}.npy"
    # Save as CHW float32 (0..1)
    t0_chw = np.transpose(t0_arr.astype(np.float32) / 255.0, (2,0,1))
    t1_chw = np.transpose(t1_arr.astype(np.float32) / 255.0, (2,0,1))
    # Mask: keep values 0,1,255. If mask image contains other colors, caller should pre-map them.
    mask_save = mask_arr.astype(np.uint8)
    np.save(a_dir / name, t0_chw)
    np.save(b_dir / name, t1_chw)
    np.save(l_dir / name, mask_save)
    return str(a_dir / name), str(b_dir / name), str(l_dir / name)

def crop_tiles(img_arr, tile_size, stride):
    H, W = img_arr.shape[:2]
    tiles = []
    if H < tile_size or W < tile_size:
        # pad to tile size if scene small
        pad_h = max(0, tile_size - H)
        pad_w = max(0, tile_size - W)
        img_arr = np.pad(img_arr, ((0,pad_h),(0,pad_w),(0,0)), constant_values=0)
        H, W = img_arr.shape[:2]
    y_steps = list(range(0, H - tile_size + 1, stride))
    x_steps = list(range(0, W - tile_size + 1, stride))
    # ensure last tile touches right/bottom edges
    if (H - tile_size) % stride != 0:
        y_steps.append(H - tile_size)
    if (W - tile_size) % stride != 0:
        x_steps.append(W - tile_size)
    for y in y_steps:
        for x in x_steps:
            tiles.append(img_arr[y:y+tile_size, x:x+tile_size])
    return tiles

def remap_mask(mask_arr):
    # If mask colors are RGB-coded, user should map colors -> {0,1,255} prior.
    # Here we try to handle common SVCD: white background 0, changed objects maybe non-zero,
    # and 255 used as ignore. We'll map >0 to 1.
    if mask_arr.ndim == 3:
        # If RGB, convert to gray first then threshold
        mask_gray = mask_arr[...,0]
    else:
        mask_gray = mask_arr
    # If mask has 255 values, keep them
    out = np.array(mask_gray, copy=True)
    # Map any pixel >1 (like 2..255) appropriately: keep 255 if equals 255 else map >0 ->1
    out = np.where(out == 255, 255, np.where(out > 0, 1, 0)).astype(np.uint8)
    return out

def main(args):
    scenes_dir = Path(args.scenes)
    out_dir = Path(args.out)
    tile = int(args.tile)
    stride = int(args.stride)
    val_frac = float(args.val_frac)
    test_frac = float(args.test_frac)
    random_seed = int(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = find_scene_files(scenes_dir)
    print("Found scenes:", len(scenes))
    rows = []
    random.seed(random_seed)
    for idx, s in enumerate(scenes):
        print("Scene", idx, "->", s["a"].parent)
        # Load arrays
        a_arr = load_image_as_array(s["a"])
        b_arr = load_image_as_array(s["b"])
        out_arr = load_image_as_array(s["out"])
        # Map mask to 0/1/255
        mask_mapped = remap_mask(out_arr)
        # tile both a,b,mask same way
        t0_tiles = crop_tiles(a_arr, tile, stride)
        t1_tiles = crop_tiles(b_arr, tile, stride)
        m_tiles  = crop_tiles(mask_mapped, tile, stride)
        assert len(t0_tiles) == len(t1_tiles) == len(m_tiles)
        # assign split for scene
        r = random.random()
        if r < test_frac:
            split = "test"
        elif r < test_frac + val_frac:
            split = "val"
        else:
            split = "train"
        saved = 0
        for i, (t0,t1,m) in enumerate(zip(t0_tiles, t1_tiles, m_tiles)):
            m2 = remap_mask(m)
            # Skip tiles that are fully ignored (all 255) or empty (no valid pixels)
            valid_mask_pixels = (m2 != 255).sum()
            if valid_mask_pixels == 0:
                # if no valid region, skip
                continue
            # Optionally skip tiles where all valid are 0 (no positive). Keep many negatives? You can change.
            # If args.keep_all_neg False, skip pure-negative tiles to boost positives. For now we keep all.
            a_p, b_p, l_p = save_chip(out_dir, split, f"{idx}", saved, t0, t1, m2)
            rows.append({"t0_npy": a_p, "t1_npy": b_p, "mask_npy": l_p, "split": split})
            saved += 1
        print("  saved", saved, "chips from scene", idx, "split", split)
    df = pd.DataFrame(rows)
    # shuffle rows deterministically
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    idx_path = out_dir / "index.parquet"
    df.to_parquet(idx_path)
    print("Total saved chips:", len(df))
    print("Wrote index:", idx_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", default=256)
    ap.add_argument("--stride", default=256)
    ap.add_argument("--val_frac", default=0.1)
    ap.add_argument("--test_frac", default=0.1)
    ap.add_argument("--seed", default=42)
    args = ap.parse_args()
    main(args)
