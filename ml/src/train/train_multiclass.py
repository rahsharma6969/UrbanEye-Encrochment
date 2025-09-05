import os, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score, jaccard_score
import torch.nn.functional as F

# ---- import your model ----
from src.models.change_unet_multi import ChangeUNetMulti

IGNORE_INDEX = 255
NUM_CLASSES = 4  # CHANGED: 4 classes instead of 5 (0=background, 1=building, 2=road, 3=change)

# ---------------- Dataset ----------------
class ChipsMC(Dataset):
    def __init__(self, parquet, labels_csv, band_order=(0,1,2,3), target_size=256):
        self.df = pd.read_parquet(parquet).reset_index(drop=True)
        lab = pd.read_csv(labels_csv)
        self.labmap = {r["chip_id"]: r["label_npy"] for _, r in lab.iterrows()}
        # keep only rows that have labels
        self.df = self.df[self.df["chip_id"].isin(self.labmap.keys())].reset_index(drop=True)
        self.band_order = tuple(band_order)
        self.target_size = int(target_size)
        
        print(f"Dataset initialized with {len(self.df)} chips")

    def __len__(self): 
        return len(self.df)

    def _ensure_CHW_and_select(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array, got {arr.shape}")
        # channels-first
        if arr.shape[0] in (3,4):
            chw = arr
        # channels-last
        elif arr.shape[-1] in (3,4):
            chw = np.moveaxis(arr, -1, 0)  # (C,H,W)
        else:
            raise ValueError(f"Cannot infer channel axis for shape {arr.shape}")
        # select desired bands explicitly along channel axis
        idx = np.array(self.band_order, dtype=int)
        return chw[idx, :, :].astype("float32")  # (4,H,W)

    def _norm(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0)
        mx = np.percentile(x, 99)
        if not np.isfinite(mx) or mx <= 0: 
            mx = 1.0
        return (x / mx).clip(0, 1)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        chip_id = r["chip_id"]

        t0 = self._ensure_CHW_and_select(np.load(r["t0_npy"]))
        t1 = self._ensure_CHW_and_select(np.load(r["t1_npy"]))
        y  = np.load(self.labmap[chip_id]).astype("uint8")  # (H,W) with {0,1,2,3,255}

        t0 = self._norm(t0)
        t1 = self._norm(t1)

        # Convert old 5-class labels to 4-class if needed
        y_converted = y.copy()
        if np.max(y_converted) > 4:  # If using old 5-class system
            y_converted[y == 4] = 3  # Map water_change (4) to change (3)
        
        # to torch for resizing
        x0 = torch.from_numpy(t0).unsqueeze(0)       # (1,4,H,W)
        x1 = torch.from_numpy(t1).unsqueeze(0)
        yy = torch.from_numpy(y_converted).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)

        # resize to common size
        ts = self.target_size
        x0 = torch.nn.functional.interpolate(x0, size=(ts, ts), mode="bilinear", align_corners=False)
        x1 = torch.nn.functional.interpolate(x1, size=(ts, ts), mode="bilinear", align_corners=False)
        yy = torch.nn.functional.interpolate(yy, size=(ts, ts), mode="nearest")

        x0 = x0.squeeze(0)                   # (4,ts,ts)
        x1 = x1.squeeze(0)                   # (4,ts,ts)
        y  = yy.long().squeeze(0).squeeze(0) # (ts,ts) long

        return x0, x1, y

def collate(batch):
    xs0, xs1, ys = zip(*batch)
    return torch.stack(xs0), torch.stack(xs1), torch.stack(ys)

# ------------- helpers -------------
def compute_class_weights(labels_csv: str) -> torch.Tensor:
    """Compute class weights for balanced training"""
    lab = pd.read_csv(labels_csv)
    counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 4 classes
    total_chips = len(lab)
    
    print("Computing class weights from labels...")
    for i, p in enumerate(lab["label_npy"]):
        if i % 100 == 0:
            print(f"  Processing {i+1}/{total_chips} labels...")
            
        y = np.load(p)
        
        # Convert old 5-class to 4-class if needed
        if np.max(y) > 4:
            y[y == 4] = 3  # Map water to change class
        
        # Count pixels for each class (excluding ignore)
        valid_mask = y != IGNORE_INDEX
        if np.sum(valid_mask) > 0:
            for c in counts.keys():
                counts[c] += int((y == c).sum())
    
    print(f"Class pixel counts: {counts}")
    
    # Calculate inverse frequency weights
    total_pixels = sum(counts.values())
    if total_pixels == 0:
        return torch.ones(NUM_CLASSES, dtype=torch.float32)
    
    weights = []
    for c in range(NUM_CLASSES):
        if counts[c] > 0:
            weight = total_pixels / (NUM_CLASSES * counts[c])
        else:
            weight = 1.0
        weights.append(weight)
    
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"Class weights: {weight_tensor.numpy()}")
    return weight_tensor

def calculate_metrics(model, dataloader, device, verbose=False):
    """Calculate validation metrics"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch_idx, (x0, x1, y) in enumerate(dataloader):
            x0, x1, y = x0.to(device), x1.to(device), y.to(device)
            
            # Forward pass
            logits = model(x0, x1)  # (B, 4, H, W)
            preds = torch.softmax(logits, dim=1).argmax(dim=1)  # (B, H, W)
            
            # Only evaluate valid pixels
            for b in range(y.shape[0]):
                valid_mask = y[b] != IGNORE_INDEX
                if torch.sum(valid_mask) > 0:
                    all_preds.extend(preds[b][valid_mask].cpu().numpy())
                    all_labels.extend(y[b][valid_mask].cpu().numpy())
    
    if len(all_preds) == 0:
        return 0.0, 0.0, 0.0  # acc, miou, f1
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    
    # IoU calculation
    try:
        iou_scores = jaccard_score(all_labels, all_preds, average=None, labels=[0,1,2,3], zero_division=0)
        miou = np.mean(iou_scores)
    except:
        miou = 0.0
    
    # F1 score
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    if verbose:
        print(f"  Validation - Accuracy: {acc:.3f}, mIoU: {miou:.3f}, F1: {f1:.3f}")
        
        # Class distribution in validation
        class_names = ["background", "building", "road", "change"]
        print("  Class distribution in validation:")
        for i, name in enumerate(class_names):
            count = np.sum(all_labels == i)
            pct = 100 * count / len(all_labels) if len(all_labels) > 0 else 0
            print(f"    {name}: {count:,} ({pct:.1f}%)")
    
    return acc, miou, f1

# ------------- training -------------
def train(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels_csv = a.labels_csv or os.path.join(a.labels_dir, "labels_index.csv")

    # Check if files exist
    if not os.path.exists(a.parquet):
        raise FileNotFoundError(f"Parquet file not found: {a.parquet}")
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    # dataset
    ds = ChipsMC(a.parquet, labels_csv, band_order=(0,1,2,3), target_size=a.target_size)
    total_chips = len(pd.read_parquet(a.parquet))
    total_labels = len(pd.read_csv(labels_csv))
    
    print(f"Using device: {device}")
    print(f"Total chips in parquet: {total_chips}")
    print(f"Total chips with labels: {total_labels}")
    print(f"Dataset will use {len(ds)} samples")

    if len(ds) == 0:
        raise ValueError("No matching chips found between parquet and labels!")

    # split 80/20 (at least 1 val)
    n = len(ds)
    if n < 2:
        raise ValueError(f"Not enough labeled samples to train (need >=2, got {n}).")
    
    n_val = max(int(0.2 * n), 1)
    n_train = n - n_val
    
    # Create indices for train/val split
    indices = list(range(n))
    np.random.seed(42)  # For reproducible splits
    np.random.shuffle(indices)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_subset = torch.utils.data.Subset(ds, train_indices)
    val_subset = torch.utils.data.Subset(ds, val_indices)

    print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")

    tl = DataLoader(train_subset, batch_size=a.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    vl = DataLoader(val_subset, batch_size=a.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    # Model and training setup
    model = ChangeUNetMulti(in_ch=8, n_classes=NUM_CLASSES).to(device)
    
    # Compute class weights for balanced training
    class_weights = compute_class_weights(labels_csv).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    
    # Use AdamW with weight decay and learning rate scheduling
    optim = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='max', factor=0.5, patience=5
    )

    def run_epoch(dl, train_mode=True):
        model.train(train_mode)
        total_loss, total_pix = 0.0, 0
        correct_pix = 0
        
        with torch.set_grad_enabled(train_mode):
            for batch_idx, (x0, x1, y) in enumerate(dl):
                x0, x1, y = x0.to(device), x1.to(device), y.to(device)
                
                # Forward pass
                logits = model(x0, x1)  # (B, 4, H, W)
                
                # Ensure output size matches target
                if logits.shape[-2:] != y.shape[-2:]:
                    logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)
                
                loss = criterion(logits, y)
                
                if train_mode:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                
                # Calculate accuracy for valid pixels
                valid_mask = y != IGNORE_INDEX
                if torch.sum(valid_mask) > 0:
                    preds = torch.softmax(logits, dim=1).argmax(dim=1)
                    correct_pix += torch.sum((preds == y) & valid_mask).item()
                    total_pix += torch.sum(valid_mask).item()
                else:
                    total_pix += y.numel()
                
                total_loss += loss.item() * y.numel()
                
                # Print progress for training
                if train_mode and batch_idx % 10 == 0:
                    current_acc = correct_pix / max(total_pix, 1)
                    print(f"    Batch {batch_idx}/{len(dl)}, Loss: {loss.item():.4f}, Acc: {current_acc:.3f}")
        
        avg_loss = total_loss / max(len(dl.dataset) * (a.target_size ** 2), 1)
        avg_acc = correct_pix / max(total_pix, 1)
        
        return avg_loss, avg_acc

    # Training loop with better monitoring
    best_miou = 0.0
    best_acc = 0.0
    os.makedirs(a.out_dir, exist_ok=True)
    
    print(f"\nStarting training for {a.epochs} epochs...")
    print("="*60)
    
    for ep in range(1, a.epochs + 1):
        print(f"\nEpoch {ep}/{a.epochs}")
        print("-" * 40)
        
        # Training
        tr_loss, tr_acc = run_epoch(tl, True)
        
        # Validation
        val_loss, val_acc = run_epoch(vl, False)
        
        # Detailed validation metrics every 5 epochs
        if ep % 5 == 0 or ep == a.epochs:
            val_acc_detailed, val_miou, val_f1 = calculate_metrics(model, vl, device, verbose=True)
            
            # Update learning rate based on mIoU
            scheduler.step(val_miou)
            
            # Save best model based on mIoU
            if val_miou > best_miou:
                best_miou = val_miou
                best_acc = val_acc_detailed
                
                # Save model with detailed info
                torch.save({
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'train_loss': tr_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc_detailed,
                    'val_miou': val_miou,
                    'val_f1': val_f1,
                    'class_weights': class_weights.cpu(),
                    'num_classes': NUM_CLASSES
                }, os.path.join(a.out_dir, "change_unet_multi_best.pth"))
                
                print(f"🎯 NEW BEST MODEL! mIoU: {val_miou:.3f}, Acc: {val_acc_detailed:.3f}")
        
        else:
            print(f"Epoch {ep:02d}/{a.epochs}  train_loss={tr_loss:.6f}  train_acc={tr_acc:.3f}  val_loss={val_loss:.6f}  val_acc={val_acc:.3f}")
    
    print(f"\n✅ Training completed!")
    print(f"📈 Best validation mIoU: {best_miou:.3f}")
    print(f"📈 Best validation accuracy: {best_acc:.3f}")
    print(f"💾 Model saved: {os.path.join(a.out_dir, 'change_unet_multi_best.pth')}")
    
    # Save training summary
    summary = {
        "final_epoch": a.epochs,
        "best_miou": best_miou,
        "best_accuracy": best_acc,
        "num_classes": NUM_CLASSES,
        "train_samples": len(train_subset),
        "val_samples": len(val_subset),
        "target_size": a.target_size
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(a.out_dir, "training_summary.csv"), index=False)

def calculate_metrics(model, dataloader, device, verbose=False):
    """Calculate detailed validation metrics"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x0, x1, y in dataloader:
            x0, x1, y = x0.to(device), x1.to(device), y.to(device)
            
            # Forward pass
            logits = model(x0, x1)  # (B, 4, H, W)
            
            # Ensure output size matches target
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)
            
            preds = torch.softmax(logits, dim=1).argmax(dim=1)  # (B, H, W)
            
            # Only evaluate valid pixels
            for b in range(y.shape[0]):
                valid_mask = y[b] != IGNORE_INDEX
                if torch.sum(valid_mask) > 0:
                    all_preds.extend(preds[b][valid_mask].cpu().numpy())
                    all_labels.extend(y[b][valid_mask].cpu().numpy())
    
    if len(all_preds) == 0:
        if verbose:
            print("  WARNING: No valid pixels found in validation set!")
        return 0.0, 0.0, 0.0
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    
    # IoU calculation
    try:
        iou_scores = jaccard_score(all_labels, all_preds, average=None, labels=[0,1,2,3], zero_division=0)
        miou = np.mean(iou_scores)
        
        if verbose:
            class_names = ["background", "building", "road", "change"]
            print("  Per-class IoU:")
            for i, (name, iou) in enumerate(zip(class_names, iou_scores)):
                print(f"    {name}: {iou:.3f}")
    except Exception as e:
        if verbose:
            print(f"  Warning: IoU calculation failed: {e}")
        miou = 0.0
    
    # F1 score
    from sklearn.metrics import f1_score
    try:
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    except:
        f1 = 0.0
    
    return acc, miou, f1

# ------------- CLI -------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train 4-class change detection model")
    ap.add_argument("--parquet", default="outputs/chips_index_s2.parquet",
                    help="Path to chips index parquet file")
    ap.add_argument("--labels_dir", default="data/labels/multiclass_4class",
                    help="Directory containing label files")
    ap.add_argument("--labels_csv", default="",
                    help="Path to labels index CSV (auto-detected if empty)")
    ap.add_argument("--target_size", type=int, default=256,
                    help="Target size for resizing chips")
    ap.add_argument("--epochs", type=int, default=50,
                    help="Number of training epochs")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="Batch size for training")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate")
    ap.add_argument("--out_dir", default="outputs/models_multi",
                    help="Output directory for saved models")
    
    args = ap.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.parquet):
        raise FileNotFoundError(f"Parquet file not found: {args.parquet}")
    
    labels_csv = args.labels_csv or os.path.join(args.labels_dir, "labels_index.csv")
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    
    train(args)
    
    
    
# python scripts/train_multiclass.py ^
#   --parquet outputs/chips_index_s2.parquet ^
#   --labels_dir data/labels/multiclass_4class ^
#   --epochs 50 ^
#   --batch_size 4 ^
#   --lr 1e-3 ^
#   --target_size 256 ^
#   --out_dir outputs/models_multi


# python -m src.train.train_multiclass ^
#   --parquet outputs/chips_index_s2.parquet ^
#   --labels_dir data/labels/multiclass_4class ^
#   --epochs 50 ^
#   --batch_size 4 ^
#   --lr 1e-3 ^
#   --out_dir outputs/models_multi




