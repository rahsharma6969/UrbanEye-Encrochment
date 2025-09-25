#!/usr/bin/env python3
"""
evaluate_matrix.py

Run pixel-wise inference on a split, compute confusion matrix, per-class metrics,
and save visualizations and CSVs.

Usage:
python -m src.train.evaluate ^
  --config configs/svcd_train.yaml ^
  --checkpoint outputs/checkpoints/best.pth ^
  --split val ^
  --out-dir outputs/eval_val
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet
from src.dataset.change_dataset import ChangeDataset
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from datetime import datetime
from pathlib import Path
import json
import matplotlib.pyplot as plt

# -------------------------
# Helper functions
# -------------------------
def flatten_valid_pixels(pred_mask, true_mask, ignore_index=255):
    """Return flattened 1D arrays of predicted and true pixels where true != ignore_index"""
    mask = (true_mask != ignore_index)
    if mask.sum() == 0:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.uint8)
    return pred_mask[mask].ravel().astype(np.int64), true_mask[mask].ravel().astype(np.int64)

def iou_score_np(pred_mask, true_mask, class_id=1, ignore_index=255, eps=1e-8):
    """IoU for a single class (binary) in masks where values are class ids."""
    valid = (true_mask != ignore_index)
    if valid.sum() == 0:
        return float('nan')
    pred_bin = (pred_mask == class_id)[valid]
    true_bin = (true_mask == class_id)[valid]
    inter = np.logical_and(pred_bin, true_bin).sum()
    union = np.logical_or(pred_bin, true_bin).sum()
    if union == 0:
        return float('nan')
    return inter / (union + eps)

def plot_confusion_matrix(cm, class_names, out_path, title="Confusion matrix"):
    """Simple confusion matrix plotting using matplotlib (no external styles)."""
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm)  # default colormap
    ax.set_title(title)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", color="w" if cm[i,j] > cm.max()/2 else "k")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

# -------------------------
# Main evaluation
# -------------------------
def evaluate_matrix(cfg, checkpoint_path, split='val', device=None, num_workers=4, batch_size=None, out_dir="eval_outputs", n_save_samples=10):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_dir, f"{split}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "examples"), exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # load index and pick split
    df = pd.read_parquet(cfg['train_index'])
    if 'split' in df.columns:
        df_split = df[df.split == split].reset_index(drop=True)
    else:
        df_split = df

    ds = ChangeDataset(df_split)
    bs = batch_size or cfg.get('batch_size', 8)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)

    # recreate model
    model = Unet(
        encoder_name=cfg.get('encoder', 'resnet34'),
        in_channels=cfg.get('in_channels', 3),
        classes=cfg.get('classes', 2),
        encoder_weights=cfg.get('encoder_weights', 'imagenet')
    )
    model.to(device)

    # load checkpoint (supports both plain state_dict and wrapper with 'model_state')
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    all_preds = []
    all_trues = []
    per_sample_records = []
    sample_idx = 0
    saved_examples = []

    with torch.no_grad():
        pbar = tqdm(dl, desc="Evaluating")
        for batch in pbar:
            # Accept (x, y) or (x, y, extras)
            if len(batch) == 3:
                x, y, extras = batch
            elif len(batch) == 2:
                x, y = batch
                extras = None
            else:
                x, y = batch[0], batch[1]
                extras = None

            x = x.to(device)
            y_np = y.cpu().numpy().astype(np.int64)  # B,H,W

            logits = model(x)
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits).cpu().numpy()  # B,1,H,W
                preds = (probs >= 0.5).astype(np.uint8)[:,0]  # B,H,W with values 0/1
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()  # B,C,H,W
                preds = np.argmax(probs, axis=1).astype(np.int64)  # B,H,W

            B = preds.shape[0]
            for i in range(B):
                pred_i = preds[i]
                true_i = y_np[i]

                # flatten valid pixels
                pred_flat, true_flat = flatten_valid_pixels(pred_i, true_i, ignore_index=cfg.get('ignore_index', 255))
                # append to global lists (for confusion matrix)
                if pred_flat.size > 0:
                    all_preds.append(pred_flat)
                    all_trues.append(true_flat)

                # per-sample metrics (precision/recall/f1/iou)
                rec = {}
                if pred_flat.size == 0:
                    rec['precision'] = float('nan')
                    rec['recall'] = float('nan')
                    rec['f1'] = float('nan')
                    rec['iou'] = float('nan')
                else:
                    # compute binary dice if binary, or use sklearn supports for multi-class
                    # We'll compute per-sample overall F1 (micro)
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    rec['precision'] = float(precision_score(true_flat, pred_flat, average='binary' if cfg.get('classes',2)==2 else 'macro', zero_division=0))
                    rec['recall'] = float(recall_score(true_flat, pred_flat, average='binary' if cfg.get('classes',2)==2 else 'macro', zero_division=0))
                    rec['f1'] = float(f1_score(true_flat, pred_flat, average='binary' if cfg.get('classes',2)==2 else 'macro', zero_division=0))
                    # IoU: compute for positive class (1) if binary; else mean IoU across classes
                    if cfg.get('classes',2) == 2:
                        rec['iou'] = float(iou_score_np(pred_i, true_i, class_id=1, ignore_index=cfg.get('ignore_index',255)))
                    else:
                        # mean IoU across classes (excluding ignore)
                        ious = []
                        for cls in range(cfg.get('classes',2)):
                            iou_c = iou_score_np(pred_i, true_i, class_id=cls, ignore_index=cfg.get('ignore_index',255))
                            if not np.isnan(iou_c):
                                ious.append(iou_c)
                        rec['iou'] = float(np.mean(ious)) if ious else float('nan')

                per_sample_records.append({
                    'sample_idx': sample_idx,
                    'precision': rec['precision'],
                    'recall': rec['recall'],
                    'f1': rec['f1'],
                    'iou': rec['iou']
                })

                # save example overlays for first N samples
                if sample_idx < n_save_samples:
                    # attempt to extract T1 image for visualization
                    t1_img = None
                    try:
                        inp_np = x[i].cpu().numpy()
                        C = inp_np.shape[0]
                        if C >= 6:
                            t1_img = inp_np[3:6].transpose(1,2,0)
                        else:
                            t1_img = inp_np[:3].transpose(1,2,0)
                        # normalize to 0-255 for saving
                        if t1_img.max() <= 1.1:
                            t1_img_vis = (np.clip(t1_img,0,1) * 255).astype(np.uint8)
                        else:
                            t1_img_vis = t1_img.astype(np.uint8)
                        # create overlays: GT green, Pred red
                        vis = t1_img_vis.copy()
                        alpha = 0.5
                        gt_mask = (true_i == 1)
                        pred_mask_bool = (pred_i == 1)
                        vis[gt_mask] = (vis[gt_mask] * (1-alpha) + np.array([0,255,0])*alpha).astype(np.uint8)
                        vis[pred_mask_bool] = (vis[pred_mask_bool] * (1-alpha) + np.array([255,0,0])*alpha).astype(np.uint8)
                        outfile = os.path.join(out_dir, "examples", f"sample_{sample_idx:04d}.png")
                        # save with matplotlib
                        fig = plt.figure(figsize=(4,4))
                        plt.imshow(vis); plt.axis('off'); plt.tight_layout()
                        fig.savefig(outfile, bbox_inches='tight', pad_inches=0.0, dpi=150)
                        plt.close(fig)
                        saved_examples.append(outfile)
                    except Exception:
                        pass

                sample_idx += 1

    # join all flattened arrays
    if len(all_preds) == 0:
        raise RuntimeError("No valid pixels found in dataset (maybe ignore_index covers all pixels).")

    all_preds_flat = np.concatenate(all_preds).ravel()
    all_trues_flat = np.concatenate(all_trues).ravel()

    # determine class labels
    n_classes = cfg.get('classes', 2)
    labels = list(range(n_classes))
    class_names = [str(i) for i in labels]

    # confusion matrix (rows=true, cols=pred)
    cm = confusion_matrix(all_trues_flat, all_preds_flat, labels=labels)
    # save cm CSV
    cm_csv_path = os.path.join(out_dir, f"confusion_matrix_{split}_{ts}.csv")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_csv_path)

    # plot and save cm image
    cm_img_path = os.path.join(out_dir, f"confusion_matrix_{split}_{ts}.png")
    plot_confusion_matrix(cm, class_names, cm_img_path, title=f"Confusion matrix ({split})")

    # classification report (per-class precision/recall/f1)
    # We use sklearn's report
    report = classification_report(all_trues_flat, all_preds_flat, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    # add IoU per class
    for cls in labels:
        report[str(cls)]['iou'] = iou_score_np((all_preds_flat == cls).astype(np.uint8).reshape(-1), (all_trues_flat == cls).astype(np.uint8).reshape(-1), class_id=1, ignore_index=-1) if n_classes==2 and cls==1 else None
        # For multiclass, compute per-class IoU:
        if n_classes > 2:
            # compute IoU for cls using global arrays
            pred_bin = (all_preds_flat == cls).astype(np.uint8)
            true_bin = (all_trues_flat == cls).astype(np.uint8)
            inter = np.logical_and(pred_bin, true_bin).sum()
            union = np.logical_or(pred_bin, true_bin).sum()
            report[str(cls)]['iou'] = float(inter / union) if union > 0 else float('nan')

    # save classification report JSON
    report_path = os.path.join(out_dir, f"classification_report_{split}_{ts}.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2, default=lambda x: None)

    # save per-sample metrics CSV
    per_sample_df = pd.DataFrame(per_sample_records)
    per_sample_csv = os.path.join(out_dir, f"metrics_{split}_{ts}.csv")
    per_sample_df.to_csv(per_sample_csv, index=False)

    # print summary
    overall = precision_recall_fscore_support(all_trues_flat, all_preds_flat, average='binary' if n_classes==2 else 'macro', zero_division=0)
    overall_prec, overall_rec, overall_f1, _ = overall
    # compute mean IoU (for binary, IoU of class1; for multiclass, mean IoU)
    if n_classes == 2:
        mean_iou = iou_score_np(all_preds_flat.reshape(-1), all_trues_flat.reshape(-1), class_id=1, ignore_index=-1)
    else:
        ious = []
        for cls in labels:
            pred_bin = (all_preds_flat == cls)
            true_bin = (all_trues_flat == cls)
            union = np.logical_or(pred_bin, true_bin).sum()
            inter = np.logical_and(pred_bin, true_bin).sum()
            if union > 0:
                ious.append(inter / union)
        mean_iou = float(np.mean(ious)) if ious else float('nan')

    summary = {
        'split': split,
        'n_samples_in_split': len(df_split),
        'n_valid_pixels': int(all_trues_flat.size),
        'overall_precision': float(overall_prec),
        'overall_recall': float(overall_rec),
        'overall_f1': float(overall_f1),
        'mean_iou': float(mean_iou),
        'confusion_matrix_csv': cm_csv_path,
        'confusion_matrix_png': cm_img_path,
        'classification_report_json': report_path,
        'per_sample_csv': per_sample_csv,
        'saved_example_images': saved_examples
    }

    # save summary JSON
    with open(os.path.join(out_dir, f"summary_{split}_{ts}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("EVAL SUMMARY")
    print(json.dumps(summary, indent=2))
    return summary, report, per_sample_df

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="path to yaml config")
    parser.add_argument("--checkpoint", "-ckpt", required=True, help="path to model checkpoint (.pth)")
    parser.add_argument("--split", default="val", choices=["train","val","test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default="eval_outputs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-save-samples", type=int, default=10)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    evaluate_matrix(cfg, args.checkpoint, split=args.split, device=args.device,
                    num_workers=args.num_workers, batch_size=args.batch_size,
                    out_dir=args.out_dir, n_save_samples=args.n_save_samples)
