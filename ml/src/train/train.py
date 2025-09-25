"""
Train script for UrbanEyeML (SVCD). Usage (example):
    python -m src.train.train --config configs/svcd_train_improved.yaml --index data/chips_256/npy/index.parquet
    

"""

import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from datetime import datetime
from tqdm import tqdm
import os
from PIL import Image
from skimage.morphology import remove_small_objects
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

# optional libs (smp, albumentations). We'll guard imports.
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except Exception:
    SMP_AVAILABLE = False

try:
    import albumentations as A
    ALB_AVAILABLE = True
except Exception:
    ALB_AVAILABLE = False

# local imports - adapt if your repo layout differs
from src.dataset.change_dataset import ChangeDataset
# if you have a local UNet fallback
try:
    from src.models.unet import UNet as LocalUNet
    LOCAL_UNET_AVAILABLE = True
except Exception:
    LOCAL_UNET_AVAILABLE = False

# ---------------- helpers ----------------
def load_cfg(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_device(cfg):
    dev_cfg = cfg.get("model", {}).get("device", "auto")
    if isinstance(dev_cfg, str) and dev_cfg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(dev_cfg, str) and dev_cfg.startswith("cuda"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")

def as_float(x, default):
    try:
        return float(x)
    except Exception:
        return default

def as_int(x, default):
    try:
        return int(x)
    except Exception:
        return default

def dice_loss_binary_from_probs(probs_pos, target, ignore_index=255, eps=1e-6):
    """
    probs_pos: torch.Tensor (N, H, W) probabilities for positive class
    target: torch.Tensor (N, H, W) integer labels {0,1} or 255 for ignore
    """
    # mask out valid pixels
    valid = (target != ignore_index)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=probs_pos.device)
    probs = probs_pos[valid]
    tgt = target[valid].float()
    num = 2.0 * (probs * tgt).sum() + eps
    den = probs.sum() + tgt.sum() + eps
    return 1.0 - (num / den)

class FocalCrossEntropy(nn.Module):
    """Simple focal wrapper around torch.nn.functional.cross_entropy for multi-class CE"""
    def __init__(self, gamma=2.0, alpha=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # list/tuple or None -> will be converted to tensor
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        # logits: (N, C, H, W); target: (N, H, W)
        n, c, h, w = logits.shape
        logits_flat = logits.permute(0,2,3,1).reshape(-1, c)
        target_flat = target.view(-1)
        # mask out ignore
        valid_mask = (target_flat != self.ignore_index)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=False)
        logits_flat = logits_flat[valid_mask]
        target_flat = target_flat[valid_mask]
        probs = torch.softmax(logits_flat, dim=1)
        pt = probs[range(len(target_flat)), target_flat.long()]
        logpt = torch.log(pt + 1e-9)
        if self.alpha is not None:
            # alpha per class
            if not torch.is_tensor(self.alpha):
                alpha = torch.tensor(self.alpha, device=logits.device, dtype=torch.float32)
            else:
                alpha = self.alpha.to(logits.device)
            at = alpha[target_flat.long()]
            loss = - at * ((1 - pt) ** self.gamma) * logpt
        else:
            loss = - ((1 - pt) ** self.gamma) * logpt
        return loss.mean()

# ---------------- index building ----------------
def build_index_from_chips_folder(folder: Path):
    rows = []
    # prefer explicit A/B/label structure under folder/{train,val,test}/A etc.
    for split in ["train", "val", "test"]:
        a_dir = folder / split / "A"
        b_dir = folder / split / "B"
        l_dir = folder / split / "label"
        if a_dir.exists() and b_dir.exists() and l_dir.exists():
            for a in sorted(a_dir.glob("*")):
                name = a.name
                b = b_dir / name
                l = l_dir / name
                if b.exists() and l.exists():
                    rows.append({"t0_npy": str(a), "t1_npy": str(b), "mask_npy": str(l), "split": split})
    # fallback: search for *_A.*
    if not rows:
        for a in sorted(folder.rglob("*_A.*")):
            prefix = a.name.rsplit("_A", 1)[0]
            parent = a.parent
            b_candidate = parent / f"{prefix}_B{a.suffix}"
            out_candidate = parent / f"{prefix}_OUT{a.suffix}"
            if not b_candidate.exists():
                b_candidate = next(parent.glob(f"{prefix}_B*"), None)
            if not out_candidate.exists():
                out_candidate = next(parent.glob(f"{prefix}_OUT*"), None)
            if b_candidate and out_candidate:
                rows.append({"t0_npy": str(a), "t1_npy": str(b_candidate), "mask_npy": str(out_candidate), "split": "train"})
    df = pd.DataFrame(rows)
    return df

# ---------------- dataloaders ----------------
def prepare_dataloaders(df: pd.DataFrame, cfg, sampler=None):
    if "split" not in df.columns:
        df["split"] = "train"
    tr = df[df.split == "train"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)

    # augmentations
    train_aug = None
    if ALB_AVAILABLE:
        tile_size = cfg.get("preprocess", {}).get("tile_size", 256)
        train_aug = A.Compose([
            A.RandomCrop(tile_size, tile_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], additional_targets={'image_2': 'image'})

    train_ds = ChangeDataset(tr, aug=train_aug)
    val_ds = ChangeDataset(va, aug=None) if not va.empty else None
    test_ds = ChangeDataset(te, aug=None) if not te.empty else None

    bs = as_int(cfg["model"].get("batch_size", 8), 8)
    nw = as_int(cfg["training"].get("num_workers", 2) if cfg.get("training") else cfg["model"].get("num_workers", 2), 2)

    if sampler is not None:
        train_dl = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=nw, pin_memory=(torch.cuda.is_available()))
    else:
        train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=(torch.cuda.is_available()))

    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=(torch.cuda.is_available())) if val_ds else None
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=(torch.cuda.is_available())) if test_ds else None

    return train_dl, val_dl, test_dl, train_ds, val_ds, test_ds

# ---------------- training & validation ----------------
def train_one_epoch(model, dataloader, optimizer, device, criterion, num_classes, scaler=None, dice_weight=0.0):
    model.train()
    total_loss = 0.0
    total_samples = 0
    pbar = tqdm(dataloader, desc="Train")
    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).long()  # shape (N,H,W)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(xb)
                if isinstance(logits, dict):
                    logits = logits.get("out", list(logits.values())[0])
                base_loss = criterion(logits, yb)
                # if binary, add dice on positive class
                if num_classes == 2 and dice_weight > 0:
                    probs = torch.softmax(logits, dim=1)[:,1,:,:]  # (N,H,W)
                    dloss = dice_loss_binary_from_probs(probs, yb)
                    loss = (1.0 - dice_weight) * base_loss + dice_weight * dloss
                else:
                    loss = base_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            if isinstance(logits, dict):
                logits = logits.get("out", list(logits.values())[0])
            base_loss = criterion(logits, yb)
            if num_classes == 2 and dice_weight > 0:
                probs = torch.softmax(logits, dim=1)[:,1,:,:]
                dloss = dice_loss_binary_from_probs(probs, yb)
                loss = (1.0 - dice_weight) * base_loss + dice_weight * dloss
            else:
                loss = base_loss
            loss.backward()
            optimizer.step()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs
        pbar.set_postfix({"loss": f"{(total_loss/total_samples):.6f}"})
    return total_loss / total_samples if total_samples else 0.0

def validate_and_metrics(model, dataloader, device, threshold=0.5, min_area=0, ignore_index=255, save_sample_dir=None, max_save=4):
    """
    Evaluate model on dataloader, applying threshold and min_area (small-object removal).
    If save_sample_dir is provided (Path or str), saves up to max_save binarized masks and probability maps for debugging.
    Returns: (tp, fp, fn, tn), stats_dict
    """
    if dataloader is None:
        return (0, 0, 0, 0), {}
    model.eval()
    tp = fp = fn = tn = 0
    prob_stats = []
    saved = 0
    save_sample_dir = Path(save_sample_dir) if save_sample_dir is not None else None
    if save_sample_dir is not None:
        save_sample_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx_batch, (xb, yb) in enumerate(tqdm(dataloader, desc="Val")):
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            logits = model(xb)
            if isinstance(logits, dict):
                logits = logits.get("out", list(logits.values())[0])
            probs_batch = torch.softmax(logits, dim=1)[:,1,:,:].cpu().numpy()  # (N,H,W)
            gts_batch = yb.cpu().numpy()

            for j in range(probs_batch.shape[0]):
                p_map = probs_batch[j]
                gt = gts_batch[j]

                flat = p_map.ravel()
                prob_stats.append((float(flat.mean()), float(np.median(flat)), float(flat.max()), float((flat > threshold).mean())))

                # binarize
                bin_mask = (p_map >= threshold)

                # apply ignore mask before computing metrics
                valid = (gt != ignore_index)
                if valid.sum() == 0:
                    continue

                # apply min_area filtering (only on predicted positive regions)
                if min_area and min_area > 0:
                    # ensure boolean
                    bin_mask = remove_small_objects(bin_mask.astype(bool), min_size=min_area)
                    bin_mask = bin_mask.astype(np.uint8)

                # compute confusion on valid pixels only
                p = bin_mask[valid].ravel()
                g = gt[valid].ravel()

                tp += int(((p == 1) & (g == 1)).sum())
                tn += int(((p == 0) & (g == 0)).sum())
                fp += int(((p == 1) & (g == 0)).sum())
                fn += int(((p == 0) & (g == 1)).sum())

                # optionally save sample visualizations (prob map + bin mask)
                if save_sample_dir is not None and saved < max_save:
                    bname = f"val_batch{idx_batch}_sample{j}_thr{threshold:.2f}_minarea{min_area}.png"
                    # probability heatmap (0..255)
                    hm = img_as_ubyte(np.clip(p_map, 0, 1))
                    Image.fromarray(hm).save(save_sample_dir / ("prob_" + bname))
                    # classification mask (0/255)
                    Image.fromarray((bin_mask * 255).astype('uint8')).save(save_sample_dir / ("bin_" + bname))
                    saved += 1

    # Compute metrics (stable)
    eps = 1e-9
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    stats = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "prec": prec, "rec": rec, "f1": f1, "iou": iou, "acc": acc
    }

    # Summarize probability stats
    if prob_stats:
        arr = np.array(prob_stats)
        stats.update({
            "prob_mean": float(arr[:,0].mean()),
            "prob_median": float(arr[:,1].mean()),
            "prob_max_mean": float(arr[:,2].mean()),
            "prob_frac_gt_threshold_mean": float(arr[:,3].mean())
        })

    return (tp, fp, fn, tn), stats

def save_sample_prediction(model, row, device, out_path, threshold=0.5, min_area=0):
    """Save one sample prediction (single chip) to out_path (PNG). Also save binarized version with min_area applied."""
    model.eval()
    a = np.load(row["t0_npy"])
    b = np.load(row["t1_npy"])
    # ensure shape CHW (the dataset uses concatenated channels)
    x = np.concatenate([a, b], axis=0)  # CHW
    x = torch.from_numpy(x).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, dict):
            logits = logits.get("out", list(logits.values())[0])
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)     # 0/1
        prob_pos = probs[:,1,:,:].squeeze(0).cpu().numpy()

    # Save classification map (0/1 -> 0/255)
    Image.fromarray((pred * 255).astype('uint8')).save(out_path.parent / (out_path.stem + "_cls.png"))
    # Save probability heatmap (0..1 -> 0..255)
    hm = (np.clip(prob_pos, 0, 1) * 255).astype('uint8')
    Image.fromarray(hm).save(out_path.parent / (out_path.stem + "_prob.png"))

    # Save binarized mask at threshold + apply min_area
    bin_mask = (prob_pos >= threshold)
    if min_area and min_area > 0:
        bin_mask = remove_small_objects(bin_mask.astype(bool), min_size=min_area).astype(np.uint8)
    Image.fromarray((bin_mask * 255).astype('uint8')).save(out_path.parent / (out_path.stem + f"_bin_thr{threshold:.2f}_min{min_area}.png"))

# ---------------- Helper: Initialize smp.Unet's first conv layer for 6-channel input ----------------
def init_encoder_for_6ch(model, encoder_name="resnet34"):
    """
    Correctly initialize the first conv layer of an smp Unet for 6-channel input.
    smp automatically duplicates the first 3 channels to make 6.
    We fix this by averaging the duplicated halves to form two independent halves.
    """
    if hasattr(model.encoder, 'conv1'):
        conv1 = model.encoder.conv1
        if conv1.in_channels == 6:
            # Get the current 6-channel weight (already duplicated by smp)
            current_weight = conv1.weight.data  # shape: [64, 6, 7, 7]

            # Extract the first 3 channels (which are a copy of the original ImageNet weights)
            orig_3ch = current_weight[:, :3, :, :].clone()  # shape: [64, 3, 7, 7]

            # Now split the 6-channel weight into two halves and average them
            # This ensures both T0 and T1 get the same, halved weight → better gradient flow
            current_weight[:, :3, :, :] = orig_3ch / 2.0   # T0 half
            current_weight[:, 3:, :, :] = orig_3ch / 2.0   # T1 half

            print("✅ Successfully re-initialized encoder.conv1 for 6-channel input.")
        else:
            print("⚠️ Encoder conv1 has unexpected in_channels:", conv1.in_channels)
    else:
        print("⚠️ Could not find encoder.conv1 to re-initialize.")
        
        
# ---------------- main ----------------
def main(config_path, index_arg):
    project_root = Path(__file__).resolve().parents[2]

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (project_root / cfg_path).resolve()

    idx_path = Path(str(index_arg))
    if not idx_path.is_absolute():
        idx_path = (project_root / idx_path).resolve()

    cfg = load_cfg(cfg_path)
    device = resolve_device(cfg)
    print("Using device:", device)

    # read / build index
    if idx_path.exists() and idx_path.suffix in [".parquet", ".pq"]:
        print("Loading index parquet:", idx_path)
        df = pd.read_parquet(idx_path)
    elif idx_path.exists() and idx_path.is_dir():
        print("Building index from folder:", idx_path)
        df = build_index_from_chips_folder(idx_path)
    else:
        raise SystemExit(f"Index not found or unsupported: {idx_path}")

    if df.empty:
        raise SystemExit("Index dataframe is empty — nothing to train on.")

    # ensure split column present
    if "split" not in df.columns:
        df["split"] = "train"

    # create small val split in-memory if none present
    val_fraction = float(cfg["model"].get("val_fraction", 0.1))
    if df[df.split == "val"].empty and val_fraction > 0.0:
        train_idxs = df[df.split == "train"].index
        n_val = max(1, int(len(train_idxs) * val_fraction))
        val_idxs = df[df.split == "train"].sample(n_val, random_state=42).index
        df.loc[val_idxs, "split"] = "val"
        print(f"No val split found — allocated {n_val} samples from train -> val (in-memory)")

    # numeric config resolve
    lr = as_float(cfg["model"].get("lr", 3e-4), 3e-4)
    epochs = as_int(cfg["model"].get("epochs", 30), 30)
    batch_size = as_int(cfg["model"].get("batch_size", 8), 8)
    in_ch = as_int(cfg["model"].get("in_channels", 6), 6)
    base_ch = as_int(cfg["model"].get("base_channels", 32), 32)
    num_classes = as_int(cfg["model"].get("num_classes", 2), 2)
    oversample_positive = bool(cfg["model"].get("oversample_positive", False))
    oversample_factor = float(cfg["model"].get("oversample_factor", 1.0))
    use_focal = bool(cfg["model"].get("use_focal", False))
    focal_gamma = float(cfg["model"].get("focal_gamma", 2.0))
    focal_alpha = cfg["model"].get("focal_alpha", None)
    if focal_alpha is not None and isinstance(focal_alpha, (list,tuple)):
        focal_alpha = list(focal_alpha)

    dice_weight = float(cfg["model"].get("dice_weight", 0.0))
    ce_weight = float(cfg["model"].get("ce_weight", 1.0))

    print(f"Config resolved: lr={lr}, epochs={epochs}, batch_size={batch_size}, in_ch={in_ch}, num_classes={num_classes}")

    # compute pixel counts and stable class weights (binary only)
    IGNORE_INDEX = int(cfg["model"].get("ignore_index", 255))
    class_weights = None
    if num_classes == 2:
        cnt0 = 0
        cnt1 = 0
        for p in df['mask_npy']:
            arr = np.load(p)
            if arr.ndim == 3:
                arr = arr[...,0]
            cnt0 += int((arr == 0).sum())
            cnt1 += int(((arr == 1) & (arr != IGNORE_INDEX)).sum())
        total = cnt0 + cnt1 + 1e-9
        w0 = np.sqrt(total / (cnt0 + 1e-9))
        w1 = np.sqrt(total / (cnt1 + 1e-9))
        clip_max = float(cfg["model"].get("max_class_weight", 100.0))  # ← Increased to 100
        w0 = float(np.clip(w0, 1e-6, clip_max))
        w1 = float(np.clip(w1, 1e-6, clip_max))
        class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
        print("Pixel counts:", {0: cnt0, 1: cnt1})
        print("Using class weights (clipped):", class_weights.tolist())

    # prepare sampler to oversample positive chips if requested
    sampler = None
    if oversample_positive and num_classes == 2:
        print("Preparing WeightedRandomSampler to oversample positive chips...")
        train_df = df[df.split == "train"].reset_index(drop=True)
        row_weights = []
        up = float(oversample_factor)
        for p in train_df['mask_npy']:
            a = np.load(p)
            if a.ndim == 3: a = a[...,0]
            has_pos = bool(((a == 1) & (a != IGNORE_INDEX)).any())
            row_weights.append(up if has_pos else 1.0)
        sampler = WeightedRandomSampler(row_weights, num_samples=len(row_weights), replacement=True)

    # dataloaders
    train_dl, val_dl, test_dl, train_ds, val_ds, test_ds = prepare_dataloaders(df, cfg, sampler=sampler)

    # instantiate model (prefer smp.Unet)
    print("Instantiating model (smp.Unet if available, otherwise try local UNet)...")
    model = None
    if SMP_AVAILABLE:
        enc = cfg["model"].get("encoder", "resnet34")
        enc_w = cfg["model"].get("encoder_weights", "imagenet")
        try:
            model = smp.Unet(encoder_name=enc, encoder_weights=enc_w, in_channels=in_ch, classes=num_classes)
            print(f"Building smp.Unet encoder={enc} weights={enc_w} in_ch={in_ch} out_ch={num_classes}")
        except Exception as e:
            print("smp.Unet construction failed:", e)
            model = None
    if model is None and LOCAL_UNET_AVAILABLE:
        print("Falling back to local UNet.")
        model = LocalUNet(in_ch=in_ch, base=base_ch, out_ch=num_classes if hasattr(LocalUNet, '__init__') else 1)
    if model is None:
        raise SystemExit("No model available (install segmentation_models_pytorch or provide local UNet).")

    model = model.to(device)

    # ✅ CRITICAL FIX: Initialize first conv layer for 6-channel input
    init_encoder_for_6ch(model, cfg["model"].get("encoder", "resnet34"))

    # criterion
    if use_focal:
        criterion = FocalCrossEntropy(gamma=focal_gamma, alpha=focal_alpha, ignore_index=IGNORE_INDEX)
        print(f"Using FocalCrossEntropy gamma={focal_gamma} alpha={focal_alpha} ignore_index={IGNORE_INDEX}")
    else:
        if class_weights is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
            print("Using weighted CrossEntropyLoss with ignore_index=", IGNORE_INDEX)
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            print("Using CrossEntropyLoss with ignore_index=", IGNORE_INDEX)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg["model"].get("weight_decay", 0.0)))

    # scheduler (ReduceLROnPlateau)
    scheduler = None
    if cfg.get("scheduler", {}).get("type", None) == "reduce":
        patience = int(cfg["scheduler"].get("patience", 5))
        factor = float(cfg["scheduler"].get("factor", 0.5))
        min_lr = float(cfg["scheduler"].get("min_lr", 1e-6))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor, min_lr=min_lr)

    # amp
    scaler = None
    amp_allowed = cfg["model"].get("amp", False) and device.type == "cuda"
    if amp_allowed:
        scaler = torch.cuda.amp.GradScaler()
        print("Using AMP (cuda).")

    # outputs
    out_dir = Path(cfg.get("paths", {}).get("outputs_dir", "outputs"))
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = out_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(cfg.get("paths", {}).get("logs_dir", "outputs/logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = logs_dir / "metrics.csv"
    if not metrics_csv.exists():
        metrics_csv.write_text("epoch,train_loss,val_loss,tp,fp,fn,tn,prec,rec,f1,iou,acc,prob_mean,prob_median,prob_max_mean,prob_frac_gt_threshold_mean,timestamp\n")

    best_f1 = -1.0
    monitor_metric = cfg["model"].get("monitor_metric", "f1")

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} === {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        train_loss = train_one_epoch(model, train_dl, optimizer, device, criterion, num_classes, scaler=scaler, dice_weight=dice_weight)

        # Validation
        val_threshold = float(cfg["model"].get("default_threshold", 0.1))  # ← Lowered to 0.1
        val_min_area = int(cfg["model"].get("val_min_area", 0))
        tp = fp = fn = tn = 0
        stats = {}
        val_loss = None

        if val_dl is not None:
            (tp, fp, fn, tn), stats = validate_and_metrics(
                model, val_dl, device,
                threshold=val_threshold,
                min_area=val_min_area,
                ignore_index=IGNORE_INDEX
            )
            val_loss = stats.get("val_loss", None)  # Optional if computed inside validate_and_metrics

        # Print summary
        print(f"Epoch {epoch}: train_loss={train_loss:.6f} val_f1={stats.get('f1', 'N/A'):.4f} val_prec={stats.get('prec','N/A'):.4f} val_rec={stats.get('rec','N/A'):.4f}")

        # Save latest checkpoint
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, ckpt_dir / "latest.pth")

        # Save best by monitor metric (prefer F1)
        current_metric_val = stats.get(monitor_metric, None)
        if current_metric_val is not None:
            if current_metric_val > best_f1:
                best_f1 = current_metric_val
                torch.save({"epoch": epoch, "model_state": model.state_dict(), monitor_metric: current_metric_val}, ckpt_dir / "best.pth")
                print("Saved best checkpoint ->", ckpt_dir / "best.pth")
        else:
            # Fallback to val_loss if no metric
            if val_loss is not None and (not hasattr(best_f1, 'item') or val_loss < best_f1):
                best_f1 = val_loss
                torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss}, ckpt_dir / "best.pth")
                print("Saved best checkpoint ->", ckpt_dir / "best.pth")

        # Scheduler step
        if scheduler is not None:
            scheduler.step(val_loss if val_loss is not None else train_loss)

        # Save sample prediction
        sample_row = None
        if val_ds is not None and len(val_ds) > 0:
            val_rows = df[df.split == "val"].reset_index(drop=True)
            if not val_rows.empty:
                sample_row = val_rows.iloc[0]
        else:
            train_rows = df[df.split == "train"].reset_index(drop=True)
            if not train_rows.empty:
                sample_row = train_rows.iloc[0]
        if sample_row is not None:
            try:
                out_path = preds_dir / f"epoch_{epoch}.png"
                save_sample_prediction(model, sample_row, device, out_path)
            except Exception as e:
                print("Failed to save sample prediction:", e)

        # Log to CSV
        ts = datetime.now().isoformat()
        def g(k, d=0):
            return stats.get(k, d)
        csv_line = f"{epoch},{train_loss},{g('val_loss','')},{int(g('tp',0))},{int(g('fp',0))},{int(g('fn',0))},{int(g('tn',0))},{g('prec','')},{g('rec','')},{g('f1','')},{g('iou','')},{g('acc','')},{g('prob_mean','')},{g('prob_median','')},{g('prob_max_mean','')},{g('prob_frac_gt_threshold_mean','')},{ts}\n"
        with open(metrics_csv, "a") as fh:
            fh.write(csv_line)

    print("Training finished.")
    return

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--index", required=True, help="Path to parquet index or folder with chips (A/B/label or _A/_B/_OUT)")
    args = ap.parse_args()
    main(args.config, args.index)
    
    
'''
python -m src.train.train ^
  --config configs/svcd_train_improved.yaml ^
  --index data/chips_256/npy/index.parquet
'''