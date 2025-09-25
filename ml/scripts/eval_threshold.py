"""
Evaluate a trained checkpoint by sweeping thresholds on validation set.

Usage (from ml/):
python -m scripts.eval_threshold --config configs/svcd_train_improved.yaml --index data/chips_256/npy/index.parquet --ckpt outputs/checkpoints/best.pth --device cpu

Outputs:
 - prints best threshold (by F1) and its metrics
 - saves CSV: outputs/logs/threshold_sweep.csv
 - saves one example predicted mask (best threshold) to outputs/preds/best_thresh_sample.png
"""
import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

# adapt to repo imports
from src.dataset.change_dataset import ChangeDataset
from torch.utils.data import DataLoader

# 👇 NEW IMPORT — required for smp.Unet
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

import os, csv

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_model_from_ckpt(ckpt_path, in_ch, base_ch, num_classes, device):
    """
    Load model using smp.Unet (must match training architecture).
    Assumes model was trained with smp.Unet(encoder=resnet34, in_channels=6, classes=2)
    """
    if not SMP_AVAILABLE:
        raise RuntimeError(
            "segmentation_models_pytorch (smp) is required for evaluation. "
            "Install via: pip install segmentation-models-pytorch"
        )

    print("Loading model as smp.Unet (encoder=resnet34)...")

    # Build same model as during training
    model = smp.Unet(
        encoder_name="resnet34",      # MUST MATCH TRAINING CONFIG
        encoder_weights=None,         # Not used in inference
        in_channels=in_ch,
        classes=num_classes,
    ).to(device)

    ck = torch.load(ckpt_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(ck, dict):
        if 'model_state' in ck:
            state = ck['model_state']
        elif 'model_state_dict' in ck:
            state = ck['model_state_dict']
        elif 'state_dict' in ck:
            state = ck['state_dict']
        elif 'model' in ck:
            state = ck['model']
        else:
            state = ck
    else:
        state = ck

    # Remove 'module.' prefix if present (DataParallel)
    new_state = {}
    for k, v in state.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        new_state[nk] = v

    # Load weights
    model.load_state_dict(new_state)
    model.eval()
    print(f"✅ Successfully loaded checkpoint from {ckpt_path}")
    return model

def flatten_valid_pixels(probs, gts, ignore_index=255):
    # probs: (N,H,W), gts: (N,H,W)
    mask = (gts != ignore_index)
    if mask.sum() == 0:
        return np.array([]), np.array([])
    probs_flat = probs[mask]
    gts_flat = gts[mask]
    return probs_flat, gts_flat

def evaluate_probs_vs_thresholds(probs_flat, gts_flat, thresholds):
    rows = []
    for th in thresholds:
        preds = (probs_flat >= th).astype(np.uint8)
        tp = int(((preds == 1) & (gts_flat == 1)).sum())
        fp = int(((preds == 1) & (gts_flat == 0)).sum())
        fn = int(((preds == 0) & (gts_flat == 1)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        iou = tp / (tp + fp + fn + 1e-9)
        rows.append((th, tp, fp, fn, prec, rec, f1, iou))
    return rows

def main(args):
    cfg = load_cfg(args.config)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Eval device:", device)

    index_path = Path(args.index)
    if not index_path.exists():
        raise SystemExit(f"Index not found: {index_path}")

    df = pd.read_parquet(index_path)
    if 'split' not in df.columns:
        raise SystemExit("Index must contain 'split' column. Run create_stratified_val or ensure val present.")

    val_df = df[df.split == 'val'].reset_index(drop=True)
    if val_df.empty:
        raise SystemExit("No validation rows found in index.")

    dataset = ChangeDataset(val_df)
    dl = DataLoader(dataset, batch_size=args.batch_size or 4, shuffle=False, num_workers=0)

    in_ch = cfg["model"].get("in_channels", 6)
    base_ch = cfg["model"].get("base_channels", 32)  # unused now, but kept for compatibility
    num_classes = cfg["model"].get("num_classes", 2)

    print("Loading model from:", args.ckpt)
    model = load_model_from_ckpt(args.ckpt, in_ch, base_ch, num_classes, device)

    # accumulate probs and gts
    probs_list = []
    gts_list = []
    IGNORE = 255
    with torch.no_grad():
        for xb, yb in tqdm(dl, desc="Running val"):
            xb = xb.to(device).half() if device.type == 'cuda' else xb.to(device).float()
            if device.type == 'cuda':
                logits = model(xb.half())
            else:
                logits = model(xb)
            if isinstance(logits, dict):
                logits = logits.get("out", list(logits.values())[0])
            probs = torch.softmax(logits, dim=1)[:,1,:,:].cpu().numpy()  # (N,H,W)
            gts = yb.cpu().numpy()
            if gts.ndim == 4:
                gts = gts[:,0,:,:]
            probs_list.append(probs)
            gts_list.append(gts)

    probs = np.concatenate(probs_list, axis=0)  # (N,H,W)
    gts = np.concatenate(gts_list, axis=0).astype(np.uint8)

    probs_flat, gts_flat = flatten_valid_pixels(probs, gts, ignore_index=IGNORE)
    if probs_flat.size == 0:
        raise SystemExit("No valid pixels in validation (all ignored).")

    # Sweep thresholds from 0.01 to 0.99 (99 points)
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = evaluate_probs_vs_thresholds(probs_flat, gts_flat, thresholds)

    # Save CSV
    out_dir = Path("outputs/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "threshold_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold","tp","fp","fn","precision","recall","f1","iou"])
        writer.writerows(rows)
    print("Saved threshold sweep to:", csv_path)

    # Find best by F1
    best = max(rows, key=lambda r: r[6])  # index 6 = F1
    best_th, best_tp, best_fp, best_fn, best_prec, best_rec, best_f1, best_iou = best
    print("Best threshold by F1:", best_th)
    print(f"TP={best_tp}, FP={best_fp}, FN={best_fn}, prec={best_prec:.4f}, rec={best_rec:.4f}, f1={best_f1:.4f}, iou={best_iou:.4f}")

    # Save one example prediction (first val row) binarized with best threshold
    sample_probs = probs[0]  # H,W
    sample_gt = gts[0]
    bin_mask = (sample_probs >= best_th).astype(np.uint8) * 255
    out_pred = Path("outputs/preds/best_thresh_sample.png")
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(bin_mask).save(out_pred)
    print("Saved sample binarized pred to:", out_pred)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default=None, help="cpu or cuda")
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()
    main(args)
    
    
    '''
    python -m scripts.eval_threshold ^
  --config configs/svcd_train_improved.yaml ^
  --index data/chips_256/npy/index.parquet ^
  --ckpt outputs/checkpoints/best.pth
  
 python -m scripts.apply_threshold_postprocess_eval ^
  --config configs/svcd_train_improved.yaml ^
  --index data/chips_256/npy/index.parquet ^
  --ckpt outputs/checkpoints/best.pth ^
  --threshold 0.35 ^
  --min_area 20 ^
  --device cpu
  '''