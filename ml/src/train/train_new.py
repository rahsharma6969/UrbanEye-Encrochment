#!/usr/bin/env python3
# import argparse
# import yaml
# import torch
# from torch.utils.data import DataLoader
# from segmentation_models_pytorch import Unet
# from src.dataset.change_dataset import ChangeDataset
# import os
# import pandas as pd
# import random
# import numpy as np
# from datetime import datetime
# from tqdm import tqdm
# import logging
# import json
# import sys
# import csv
# import torch.nn.functional as F
# from typing import Optional

# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# def make_logger(log_path):
#     logger = logging.getLogger("train")
#     logger.setLevel(logging.DEBUG)
#     fmt = logging.Formatter("%(asctime)s | %(levelname)8s | %(message)s", "%Y-%m-%d %H:%M:%S")

#     # console
#     if not logger.handlers:
#         ch = logging.StreamHandler(sys.stdout)
#         ch.setLevel(logging.INFO)
#         ch.setFormatter(fmt)
#         logger.addHandler(ch)

#     # file
#     fh = logging.FileHandler(log_path)
#     fh.setLevel(logging.DEBUG)
#     fh.setFormatter(fmt)
#     logger.addHandler(fh)

#     return logger

# def focal_loss(logits, targets, alpha=1.0, gamma=2.0, ignore_index=255, reduction='mean'):
#     """
#     Focal Loss for multi-class classification.
#     logits: (N, C, H, W)
#     targets: (N, H, W) with class indices, ignore_index=255
#     """
#     ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_index)
#     pt = torch.exp(-ce_loss)
#     focal_weight = alpha * ((1 - pt) ** gamma)
#     loss = focal_weight * ce_loss

#     # Mask out ignored pixels
#     valid_mask = (targets != ignore_index)
#     loss = loss * valid_mask.float()

#     if reduction == 'mean':
#         return loss.sum() / (valid_mask.sum() + 1e-8)
#     elif reduction == 'sum':
#         return loss.sum()
#     else:
#         return loss

# def dice_loss_binary(pred_logits, target, ignore_index=255, eps=1e-6):
#     """
#     Binary Dice Loss on logits (for change detection).
#     pred_logits: (N, 2, H, W) — output of Unet
#     target: (N, H, W) — binary {0, 1} or {0, 255}
#     """
#     # Get probability of positive class
#     probs = torch.softmax(pred_logits, dim=1)[:, 1, :, :]  # (N, H, W)
#     # Convert target: 255 -> ignore, 1 -> 1, 0 -> 0
#     mask_valid = (target != ignore_index)
#     if not mask_valid.any():
#         return torch.tensor(0.0, device=pred_logits.device)

#     tgt = (target[mask_valid] == 1).float()  # shape: (N*H*W,)
#     prob = probs[mask_valid]

#     inter = (prob * tgt).sum()
#     denom = prob.sum() + tgt.sum()
#     dice = (2.0 * inter + eps) / (denom + eps)
#     return 1.0 - dice

# def compute_metrics(preds, targets, ignore_index=255):
#     """Compute TP, FP, FN, TN, precision, recall, F1, IoU, accuracy from logits and targets"""
#     preds = preds.argmax(dim=1)  # (N,H,W)
#     valid = (targets != ignore_index)
#     if not valid.any():
#         return 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0

#     p = preds[valid].cpu().numpy().ravel()
#     g = targets[valid].cpu().numpy().ravel()

#     tp = int(((p == 1) & (g == 1)).sum())
#     fp = int(((p == 1) & (g == 0)).sum())
#     fn = int(((p == 0) & (g == 1)).sum())
#     tn = int(((p == 0) & (g == 0)).sum())

#     eps = 1e-9
#     prec = tp / (tp + fp + eps)
#     rec = tp / (tp + fn + eps)
#     f1 = 2 * prec * rec / (prec + rec + eps)
#     iou = tp / (tp + fp + fn + eps)
#     acc = (tp + tn) / (tp + tn + fp + fn + eps)

#     return tp, fp, fn, tn, prec, rec, f1, iou, acc

# def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None, amp=False, logger=None):
#     model.train()
#     total_loss = 0.0
#     it = 0
#     pbar = tqdm(dataloader, desc="Train", leave=False)
#     for x, y in pbar:
#         x, y = x.to(device), y.to(device)
#         optimizer.zero_grad()
#         if amp and scaler is not None:
#             with torch.cuda.amp.autocast():
#                 pred = model(x)
#                 loss = criterion(pred, y)
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             pred = model(x)
#             loss = criterion(pred, y)
#             loss.backward()
#             optimizer.step()

#         batch_loss = loss.item()
#         total_loss += batch_loss
#         it += 1
#         pbar.set_postfix(train_loss=f"{total_loss/it:.4f}")

#     avg = total_loss / max(1, it)
#     if logger: logger.info(f"Train loss: {avg:.4f}")
#     return avg

# def validate(model, dataloader, criterion, device, logger=None, ignore_index=255):
#     model.eval()
#     total_loss = 0.0
#     it = 0
#     total_tp = total_fp = total_fn = total_tn = 0
#     pbar = tqdm(dataloader, desc="Val", leave=False)
#     with torch.no_grad():
#         for x, y in pbar:
#             x, y = x.to(device), y.to(device)
#             pred = model(x)
#             loss = criterion(pred, y)
#             total_loss += loss.item()
#             it += 1
#             pbar.set_postfix(val_loss=f"{total_loss/it:.4f}")

#             # Compute metrics
#             tp, fp, fn, tn, _, _, _, _, _ = compute_metrics(pred, y, ignore_index)
#             total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn

#     avg_loss = total_loss / max(1, it)

#     eps = 1e-9
#     prec = total_tp / (total_tp + total_fp + eps)
#     rec = total_tp / (total_tp + total_fn + eps)
#     f1 = 2 * prec * rec / (prec + rec + eps)
#     iou = total_tp / (total_tp + total_fp + total_fn + eps)
#     acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + eps)

#     if logger:
#         logger.info(f"Val loss: {avg_loss:.4f}")
#         logger.info(f"Val metrics: F1={f1:.4f}, IoU={iou:.4f}, Recall={rec:.4f}, Precision={prec:.4f}")

#     return avg_loss, prec, rec, f1, iou, acc

# def save_checkpoint(state, path):
#     torch.save(state, path)

# def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device=None):
#     ckpt = torch.load(path, map_location=device)
#     model.load_state_dict(ckpt['model_state'])
#     if optimizer is not None and 'optimizer_state' in ckpt:
#         optimizer.load_state_dict(ckpt['optimizer_state'])
#     if scheduler is not None and 'scheduler_state' in ckpt:
#         scheduler.load_state_dict(ckpt['scheduler_state'])
#     if scaler is not None and 'scaler_state' in ckpt:
#         scaler.load_state_dict(ckpt['scaler_state'])
#     start_epoch = ckpt.get('epoch', 0) + 1
#     best_val = ckpt.get('best_val_loss', float('inf'))
#     return start_epoch, best_val

# def main(config_path, args):
#     # Load config
#     with open(config_path) as f:
#         cfg = yaml.safe_load(f)

#     # Merge args -> cfg (allow CLI overrides)
#     if args.batch_size: cfg['batch_size'] = args.batch_size
#     if args.epochs: cfg['epochs'] = args.epochs
#     if args.lr: cfg['lr'] = args.lr
#     if args.index: cfg['train_index'] = args.index
#     if args.val_index: cfg['val_index'] = args.val_index

#     # output dir
#     out_dir = args.output_dir or cfg.get('output_dir', 'outputs')
#     os.makedirs(out_dir, exist_ok=True)
#     os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

#     # Logging
#     log_path = os.path.join(out_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
#     logger = make_logger(log_path)
#     logger.info(f"Using config: {config_path}")
#     logger.debug(json.dumps(cfg, indent=2))

#     # Save copy of config used
#     with open(os.path.join(out_dir, "used_config.yaml"), "w") as f:
#         yaml.safe_dump(cfg, f)

#     # device
#     device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
#     logger.info(f"Device: {device}")

#     # seed
#     seed = cfg.get('seed', 42)
#     set_seed(seed)
#     logger.info(f"Seed: {seed}")

#     # read index parquet
#     df = pd.read_parquet(cfg['train_index'])
#     if 'split' in df.columns:
#         train_df = df[df.split == 'train'].reset_index(drop=True)
#         val_df = df[df.split == 'val'].reset_index(drop=True)
#     else:
#         train_df = df
#         val_df = df

#     # datasets
#     train_ds = ChangeDataset(train_df)
#     val_ds = ChangeDataset(val_df)

#     train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg.get('num_workers', 4), pin_memory=True)
#     val_dl = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg.get('num_workers', 4), pin_memory=True)
    
#     def init_encoder_for_6ch(model):
#         """
#         Fix the first convolutional layer of smp.Unet when in_channels=6.
#         If the model was initialized with encoder_weights='imagenet' and in_channels=6,
#         smp duplicates the first 3 channels → we fix it by averaging them for T0 and T1.
#         """
#         if hasattr(model.encoder, 'conv1'):
#             conv1 = model.encoder.conv1
#             if conv1.in_channels == 6:
#             # Get the current 6-channel weights (already duplicated by smp)
#                 orig_weights = conv1.weight.data  # shape: [64, 6, 7, 7]
            
#             # Extract the original 3-channel weights (first half)
#                 orig_3ch = orig_weights[:, :3, :, :].clone()  # shape: [64, 3, 7, 7]
            
#             # Set both halves to half the original weights
#                 orig_weights[:, :3, :, :] = orig_3ch / 2.0   # T0 half
#                 orig_weights[:, 3:, :, :] = orig_3ch / 2.0   # T1 half
            
#                 print("✅ Successfully re-initialized encoder.conv1 for 6-channel input.")
#             else:
#                 print("⚠️ Encoder conv1 has unexpected in_channels:", conv1.in_channels)
#         else:
#             print("⚠️ Could not find encoder.conv1 to re-initialize.")
#     # model
#     model = Unet(
#     encoder_name=cfg.get('encoder', 'resnet34'),
#     in_channels=cfg.get('in_channels', 3),
#     classes=cfg.get('classes', 2),
#     encoder_weights=cfg.get('encoder_weights', 'imagenet')
#     ).to(device)

# # 👇 CALL IT HERE — right after model creation
#     init_encoder_for_6ch(model)

#     # optimizer
#     lr = float(cfg.get('lr', 1e-3))
#     weight_decay = float(cfg.get('weight_decay', 0.0))
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

#     # scheduler
#     scheduler = None
#     sched_cfg = cfg.get('scheduler', {})
#     if sched_cfg.get('name'):
#         params = sched_cfg.get('params', {})
#         if sched_cfg['name'] == 'reduce_on_plateau':
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
#         else:
#             try:
#                 scheduler_class = getattr(torch.optim.lr_scheduler, sched_cfg['name'])
#                 scheduler = scheduler_class(optimizer, **params)
#             except Exception as e:
#                 logger.warning(f"Could not create scheduler {sched_cfg['name']}: {e}")

#     # criterion
#     crit_name = cfg.get('criterion', 'CrossEntropyLoss')
#     ignore_index = cfg.get('ignore_index', 255)

#     if crit_name == 'focal_dice_hybrid':
#         focal_alpha = cfg.get('focal_alpha', [1.0, 1.0])
#         focal_gamma = cfg.get('focal_gamma', 2.0)
#         ce_weight = cfg.get('ce_weight', 0.5)
#         dice_weight = cfg.get('dice_weight', 0.5)

#         def criterion(pred, target):
#             focal_l = focal_loss(pred, target, alpha=focal_alpha[1], gamma=focal_gamma, ignore_index=ignore_index)
#             dice_l = dice_loss_binary(pred, target, ignore_index=ignore_index)
#             return ce_weight * focal_l + dice_weight * dice_l

#     elif crit_name == 'focal_loss':
#         focal_alpha = cfg.get('focal_alpha', [1.0, 1.0])
#         focal_gamma = cfg.get('focal_gamma', 2.0)
#         def criterion(pred, target):
#             return focal_loss(pred, target, alpha=focal_alpha[1], gamma=focal_gamma, ignore_index=ignore_index)

#     elif crit_name == 'dice_loss':
#         def criterion(pred, target):
#             return dice_loss_binary(pred, target, ignore_index=ignore_index)

#     else:
#         # fallback: standard CrossEntropy
#         criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

#     # amp scaler
#     scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == 'cuda') else None

#     start_epoch = 1
#     best_val_loss = float('inf')

#     # resume
#     if args.resume:
#         ckpt_path = args.resume
#         if os.path.exists(ckpt_path):
#             logger.info(f"Resuming from checkpoint: {ckpt_path}")
#             start_epoch, best_val_loss = load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, device)
#             logger.info(f"Resumed at epoch {start_epoch}, best_val={best_val_loss:.4f}")
#         else:
#             logger.warning(f"Resume path not found: {ckpt_path}")

#     # metrics CSV
#     metrics_file = os.path.join(out_dir, "metrics.csv")
#     if not os.path.exists(metrics_file):
#         with open(metrics_file, "w", newline='') as mf:
#             writer = csv.writer(mf)
#             writer.writerow([
#                 "epoch", "train_loss", "val_loss",
#                 "lr", "tp", "fp", "fn", "tn",
#                 "precision", "recall", "f1", "iou", "acc"
#             ])

#     try:
#         for epoch in range(start_epoch, cfg.get('epochs', 10) + 1):
#             logger.info(f"Epoch {epoch}/{cfg.get('epochs')}")
#             train_loss = train_one_epoch(model, train_dl, optimizer, criterion, device, scaler=scaler, amp=(args.amp and device.type == 'cuda'), logger=logger)
#             val_loss, prec, rec, f1, iou, acc = validate(model, val_dl, criterion, device, logger=logger, ignore_index=ignore_index)

#             # scheduler step
#             if scheduler is not None:
#                 if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#                     scheduler.step(val_loss)
#                 else:
#                     scheduler.step()

#             # current lr
#             current_lr = optimizer.param_groups[0]['lr']

#             # write metrics to CSV
#             with open(metrics_file, "a", newline='') as mf:
#                 writer = csv.writer(mf)
#                 writer.writerow([
#                     epoch, train_loss, val_loss,
#                     current_lr,
#                     0, 0, 0, 0,  # tp, fp, fn, tn (we'll recompute in validate)
#                     prec, rec, f1, iou, acc
#                 ])

#             # save best
#             ckpt = {
#                 'epoch': epoch,
#                 'model_state': model.state_dict(),
#                 'optimizer_state': optimizer.state_dict(),
#                 'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
#                 'best_val_loss': best_val_loss,
#             }
#             if scaler is not None:
#                 ckpt['scaler_state'] = scaler.state_dict()

#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 best_path = os.path.join(out_dir, "checkpoints", "best.pth")
#                 ckpt['best_val_loss'] = best_val_loss
#                 save_checkpoint(ckpt, best_path)
#                 logger.info(f"New best model (val={best_val_loss:.4f}) saved to {best_path}")

#             # periodic save
#             if epoch % cfg.get('save_interval', 5) == 0 or epoch == cfg.get('epochs'):
#                 epoch_path = os.path.join(out_dir, "checkpoints", f"epoch_{epoch:03d}.pth")
#                 ckpt['best_val_loss'] = best_val_loss
#                 save_checkpoint(ckpt, epoch_path)
#                 logger.info(f"Checkpoint saved: {epoch_path}")

#     except KeyboardInterrupt:
#         logger.warning("Training interrupted by user. Saving last checkpoint...")
#         save_checkpoint({
#             'epoch': epoch,
#             'model_state': model.state_dict(),
#             'optimizer_state': optimizer.state_dict(),
#             'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
#             'best_val_loss': best_val_loss,
#             'interrupted': True,
#             'timestamp': datetime.now().isoformat()
#         }, os.path.join(out_dir, "checkpoints", "interrupted.pth"))
#     except Exception as e:
#         logger.exception(f"Unhandled exception during training: {e}")
#         # attempt to save checkpoint
#         try:
#             save_checkpoint({
#                 'epoch': epoch if 'epoch' in locals() else -1,
#                 'model_state': model.state_dict(),
#                 'optimizer_state': optimizer.state_dict(),
#                 'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
#                 'best_val_loss': best_val_loss,
#                 'error': str(e),
#                 'timestamp': datetime.now().isoformat()
#             }, os.path.join(out_dir, "checkpoints", "error_dump.pth"))
#             logger.info("Saved error_dump checkpoint.")
#         except Exception as e2:
#             logger.exception(f"Failed to save error dump: {e2}")
#         raise

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Training entrypoint")
#     parser.add_argument("--config", "-c", required=True, help="Path to yaml config")
#     parser.add_argument("--resume", "-r", default=None, help="Path to checkpoint to resume from")
#     parser.add_argument("--output-dir", default=None, help="Directory to save outputs/checkpoints")
#     parser.add_argument("--device", default=None, help="Device to use, e.g. cpu or cuda:0")
#     parser.add_argument("--amp", action='store_true', help="Use mixed precision training (only with CUDA)")
#     parser.add_argument("--batch-size", type=int, default=None, help="Override batch size in config")
#     parser.add_argument("--epochs", type=int, default=None, help="Override epochs in config")
#     parser.add_argument("--lr", type=float, default=None, help="Override LR in config")
#     parser.add_argument("--index", default=None, help="Override train_index path in config")
#     parser.add_argument("--val-index", default=None, help="Override val_index path in config")
#     parsed = parser.parse_args()

#     main(parsed.config, parsed)
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet
from src.dataset.change_dataset import ChangeDataset
import os
import pandas as pd
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
import logging
import json
import sys
import csv
import torch.nn.functional as F
from typing import Optional
from pathlib import Path

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_logger(log_path):
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)8s | %(message)s", "%Y-%m-%d %H:%M:%S")

    # console
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # file
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

def focal_loss(logits, targets, alpha=1.0, gamma=2.0, ignore_index=255, reduction='mean'):
    """
    Focal Loss for multi-class classification.
    logits: (N, C, H, W)
    targets: (N, H, W) with class indices, ignore_index=255
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_index)
    pt = torch.exp(-ce_loss)
    focal_weight = alpha * ((1 - pt) ** gamma)
    loss = focal_weight * ce_loss

    # Mask out ignored pixels
    valid_mask = (targets != ignore_index)
    loss = loss * valid_mask.float()

    if reduction == 'mean':
        return loss.sum() / (valid_mask.sum() + 1e-8)
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def dice_loss_binary(pred_logits, target, ignore_index=255, eps=1e-6):
    """
    Binary Dice Loss on logits (for change detection).
    pred_logits: (N, 2, H, W) — output of Unet
    target: (N, H, W) — binary {0, 1} or {0, 255}
    """
    # Get probability of positive class
    probs = torch.softmax(pred_logits, dim=1)[:, 1, :, :]  # (N, H, W)
    # Convert target: 255 -> ignore, 1 -> 1, 0 -> 0
    mask_valid = (target != ignore_index)
    if not mask_valid.any():
        return torch.tensor(0.0, device=pred_logits.device)

    tgt = (target[mask_valid] == 1).float()  # shape: (N*H*W,)
    prob = probs[mask_valid]

    inter = (prob * tgt).sum()
    denom = prob.sum() + tgt.sum()
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice

def compute_metrics(preds, targets, ignore_index=255):
    """Compute TP, FP, FN, TN, precision, recall, F1, IoU, accuracy from logits and targets"""
    preds = preds.argmax(dim=1)  # (N,H,W)
    valid = (targets != ignore_index)
    if not valid.any():
        return 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0

    p = preds[valid].cpu().numpy().ravel()
    g = targets[valid].cpu().numpy().ravel()

    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    tn = int(((p == 0) & (g == 0)).sum())

    eps = 1e-9
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    return tp, fp, fn, tn, prec, rec, f1, iou, acc

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None, amp=False, logger=None):
    model.train()
    total_loss = 0.0
    it = 0
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        it += 1
        pbar.set_postfix(train_loss=f"{total_loss/it:.4f}")

    avg = total_loss / max(1, it)
    if logger: logger.info(f"Train loss: {avg:.4f}")
    return avg

def validate(model, dataloader, criterion, device, logger=None, ignore_index=255):
    model.eval()
    total_loss = 0.0
    it = 0
    total_tp = total_fp = total_fn = total_tn = 0
    pbar = tqdm(dataloader, desc="Val", leave=False)
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
            it += 1
            pbar.set_postfix(val_loss=f"{total_loss/it:.4f}")

            # Compute metrics
            tp, fp, fn, tn, _, _, _, _, _ = compute_metrics(pred, y, ignore_index)
            total_tp += tp; total_fp += fp; total_fn += fn; total_tn += tn

    avg_loss = total_loss / max(1, it)

    eps = 1e-9
    prec = total_tp / (total_tp + total_fp + eps)
    rec = total_tp / (total_tp + total_fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    iou = total_tp / (total_tp + total_fp + total_fn + eps)
    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + eps)

    if logger:
        logger.info(f"Val loss: {avg_loss:.4f}")
        logger.info(f"Val metrics: F1={f1:.4f}, IoU={iou:.4f}, Recall={rec:.4f}, Precision={prec:.4f}")

    return avg_loss, prec, rec, f1, iou, acc

def save_checkpoint(state, path):
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scheduler is not None and 'scheduler_state' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state'])
    if scaler is not None and 'scaler_state' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state'])
    start_epoch = ckpt.get('epoch', 0) + 1
    best_val = ckpt.get('best_val_loss', float('inf'))
    return start_epoch, best_val

def main(config_path, args):
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Merge args -> cfg (allow CLI overrides)
    if args.batch_size: cfg['batch_size'] = args.batch_size
    if args.epochs: cfg['epochs'] = args.epochs
    if args.lr: cfg['lr'] = args.lr
    if args.index: cfg['train_index'] = args.index
    if args.val_index: cfg['val_index'] = args.val_index

    # output dir
    out_dir = args.output_dir or cfg.get('output_dir', 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    # Logging
    log_path = os.path.join(out_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = make_logger(log_path)
    logger.info(f"Using config: {config_path}")
    logger.debug(json.dumps(cfg, indent=2))

    # Save copy of config used
    with open(os.path.join(out_dir, "used_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    # device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    # seed
    seed = cfg.get('seed', 42)
    set_seed(seed)
    logger.info(f"Seed: {seed}")

    # read index parquet and determine base directory
    index_path = cfg['train_index']
    df = pd.read_parquet(index_path)
    
    # Calculate base directory from index path
    index_path_obj = Path(index_path)
    base_dir = index_path_obj.parent  # e.g., data/LEVIR_CD/chips_256
    logger.info(f"Dataset base directory: {base_dir}")
    
    if 'split' in df.columns:
        train_df = df[df.split == 'train'].reset_index(drop=True)
        val_df = df[df.split == 'val'].reset_index(drop=True)
    else:
        train_df = df
        val_df = df

    # Log dataset stats
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")

    # datasets with base_dir parameter
    train_ds = ChangeDataset(train_df, base_dir=base_dir)
    val_ds = ChangeDataset(val_df, base_dir=base_dir)

    train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg.get('num_workers', 4), pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg.get('num_workers', 4), pin_memory=True)
    
    def init_encoder_for_6ch(model):
        """
        Fix the first convolutional layer of smp.Unet when in_channels=6.
        If the model was initialized with encoder_weights='imagenet' and in_channels=6,
        smp duplicates the first 3 channels → we fix it by averaging them for T0 and T1.
        """
        if hasattr(model.encoder, 'conv1'):
            conv1 = model.encoder.conv1
            if conv1.in_channels == 6:
            # Get the current 6-channel weights (already duplicated by smp)
                orig_weights = conv1.weight.data  # shape: [64, 6, 7, 7]
            
            # Extract the original 3-channel weights (first half)
                orig_3ch = orig_weights[:, :3, :, :].clone()  # shape: [64, 3, 7, 7]
            
            # Set both halves to half the original weights
                orig_weights[:, :3, :, :] = orig_3ch / 2.0   # T0 half
                orig_weights[:, 3:, :, :] = orig_3ch / 2.0   # T1 half
            
                logger.info("Successfully re-initialized encoder.conv1 for 6-channel input.")
            else:
                logger.warning(f"Encoder conv1 has unexpected in_channels: {conv1.in_channels}")
        else:
            logger.warning("Could not find encoder.conv1 to re-initialize.")
    
    # model
    model = Unet(
    encoder_name=cfg.get('encoder', 'resnet34'),
    in_channels=cfg.get('in_channels', 3),
    classes=cfg.get('classes', 2),
    encoder_weights=cfg.get('encoder_weights', 'imagenet')
    ).to(device)

    # Call init function right after model creation
    init_encoder_for_6ch(model)

    # optimizer
    lr = float(cfg.get('lr', 1e-3))
    weight_decay = float(cfg.get('weight_decay', 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # scheduler
    scheduler = None
    sched_cfg = cfg.get('scheduler', {})
    if sched_cfg.get('name'):
        params = sched_cfg.get('params', {})
        if sched_cfg['name'] == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
        else:
            try:
                scheduler_class = getattr(torch.optim.lr_scheduler, sched_cfg['name'])
                scheduler = scheduler_class(optimizer, **params)
            except Exception as e:
                logger.warning(f"Could not create scheduler {sched_cfg['name']}: {e}")

    # criterion
    crit_name = cfg.get('criterion', 'CrossEntropyLoss')
    ignore_index = cfg.get('ignore_index', 255)

    if crit_name == 'focal_dice_hybrid':
        focal_alpha = cfg.get('focal_alpha', [1.0, 1.0])
        focal_gamma = cfg.get('focal_gamma', 2.0)
        ce_weight = cfg.get('ce_weight', 0.5)
        dice_weight = cfg.get('dice_weight', 0.5)

        def criterion(pred, target):
            focal_l = focal_loss(pred, target, alpha=focal_alpha[1], gamma=focal_gamma, ignore_index=ignore_index)
            dice_l = dice_loss_binary(pred, target, ignore_index=ignore_index)
            return ce_weight * focal_l + dice_weight * dice_l

    elif crit_name == 'focal_loss':
        focal_alpha = cfg.get('focal_alpha', [1.0, 1.0])
        focal_gamma = cfg.get('focal_gamma', 2.0)
        def criterion(pred, target):
            return focal_loss(pred, target, alpha=focal_alpha[1], gamma=focal_gamma, ignore_index=ignore_index)

    elif crit_name == 'dice_loss':
        def criterion(pred, target):
            return dice_loss_binary(pred, target, ignore_index=ignore_index)

    else:
        # fallback: standard CrossEntropy
        criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    # amp scaler
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == 'cuda') else None

    start_epoch = 1
    best_val_loss = float('inf')

    # resume
    if args.resume:
        ckpt_path = args.resume
        if os.path.exists(ckpt_path):
            logger.info(f"Resuming from checkpoint: {ckpt_path}")
            start_epoch, best_val_loss = load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, device)
            logger.info(f"Resumed at epoch {start_epoch}, best_val={best_val_loss:.4f}")
        else:
            logger.warning(f"Resume path not found: {ckpt_path}")

    # metrics CSV
    metrics_file = os.path.join(out_dir, "metrics.csv")
    if not os.path.exists(metrics_file):
        with open(metrics_file, "w", newline='') as mf:
            writer = csv.writer(mf)
            writer.writerow([
                "epoch", "train_loss", "val_loss",
                "lr", "tp", "fp", "fn", "tn",
                "precision", "recall", "f1", "iou", "acc"
            ])

    try:
        for epoch in range(start_epoch, cfg.get('epochs', 10) + 1):
            logger.info(f"Epoch {epoch}/{cfg.get('epochs')}")
            train_loss = train_one_epoch(model, train_dl, optimizer, criterion, device, scaler=scaler, amp=(args.amp and device.type == 'cuda'), logger=logger)
            val_loss, prec, rec, f1, iou, acc = validate(model, val_dl, criterion, device, logger=logger, ignore_index=ignore_index)

            # scheduler step
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # current lr
            current_lr = optimizer.param_groups[0]['lr']

            # write metrics to CSV
            with open(metrics_file, "a", newline='') as mf:
                writer = csv.writer(mf)
                writer.writerow([
                    epoch, train_loss, val_loss,
                    current_lr,
                    0, 0, 0, 0,  # tp, fp, fn, tn (we'll recompute in validate)
                    prec, rec, f1, iou, acc
                ])

            # save best
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                'best_val_loss': best_val_loss,
            }
            if scaler is not None:
                ckpt['scaler_state'] = scaler.state_dict()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(out_dir, "checkpoints", "best.pth")
                ckpt['best_val_loss'] = best_val_loss
                save_checkpoint(ckpt, best_path)
                logger.info(f"New best model (val={best_val_loss:.4f}) saved to {best_path}")

            # periodic save
            if epoch % cfg.get('save_interval', 5) == 0 or epoch == cfg.get('epochs'):
                epoch_path = os.path.join(out_dir, "checkpoints", f"epoch_{epoch:03d}.pth")
                ckpt['best_val_loss'] = best_val_loss
                save_checkpoint(ckpt, epoch_path)
                logger.info(f"Checkpoint saved: {epoch_path}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving last checkpoint...")
        save_checkpoint({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
            'best_val_loss': best_val_loss,
            'interrupted': True,
            'timestamp': datetime.now().isoformat()
        }, os.path.join(out_dir, "checkpoints", "interrupted.pth"))
    except Exception as e:
        logger.exception(f"Unhandled exception during training: {e}")
        # attempt to save checkpoint
        try:
            save_checkpoint({
                'epoch': epoch if 'epoch' in locals() else -1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                'best_val_loss': best_val_loss,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, os.path.join(out_dir, "checkpoints", "error_dump.pth"))
            logger.info("Saved error_dump checkpoint.")
        except Exception as e2:
            logger.exception(f"Failed to save error dump: {e2}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training entrypoint")
    parser.add_argument("--config", "-c", required=True, help="Path to yaml config")
    parser.add_argument("--resume", "-r", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", default=None, help="Directory to save outputs/checkpoints")
    parser.add_argument("--device", default=None, help="Device to use, e.g. cpu or cuda:0")
    parser.add_argument("--amp", action='store_true', help="Use mixed precision training (only with CUDA)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size in config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs in config")
    parser.add_argument("--lr", type=float, default=None, help="Override LR in config")
    parser.add_argument("--index", default=None, help="Override train_index path in config")
    parser.add_argument("--val-index", default=None, help="Override val_index path in config")
    parsed = parser.parse_args()

    main(parsed.config, parsed)
    
'''
   
  
  
  python -m src.train.train_new ^
  --config configs/svcd_train.yaml ^
  --index data\LEVIR_CD\chips_256\index.parquet
  
  python -m scripts.apply_threshold_postprocess_eval ^
  --config configs/svcd_train.yaml ^
  --index data\LEVIR_CD\chips_256\index.parquet ^
  --ckpt outputs\checkpoints\best.pth ^
  --threshold 0.3 ^
  --min_area 20 ^
  --device cpu
  '''