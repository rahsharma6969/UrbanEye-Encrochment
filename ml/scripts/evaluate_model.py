\
# ml/scripts/evaluate_test.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.dataset.change_dataset import ChangeDataset
from torch.utils.data import DataLoader
from src.models.unet import UNet
import yaml

CONF = Path("configs/svcd_train.yaml")
CKPT = Path("outputs/checkpoints/latest.pth")  # adjust if different

def load_cfg(p):
    import yaml
    return yaml.safe_load(open(p,'r'))

cfg = load_cfg(CONF)
device = torch.device("cuda" if torch.cuda.is_available() and cfg['model'].get('device','auto')!='cpu' else "cpu")
print("Eval device:", device)

# read index and prepare test dataloader
idx = pd.read_parquet("data/chips_256/npy/index.parquet")
test_df = idx[idx.split=='test'].reset_index(drop=True)
if test_df.empty:
    raise SystemExit("No test data in index.parquet (split='test')")

ds = ChangeDataset(test_df)
dl = DataLoader(ds, batch_size=cfg['model'].get('batch_size',4), shuffle=False, num_workers=2)

# build model
in_ch = int(cfg['model'].get('in_channels',6))
base_ch = int(cfg['model'].get('base_channels',32))
num_classes = int(cfg['model'].get('num_classes',255))

# try to build with same logic as training (constructor may differ)
try:
    model = UNet(in_ch=in_ch, base=base_ch, out_ch=num_classes)
except TypeError:
    model = UNet(in_ch=in_ch, base=base_ch)
    # wrap (simple 1x1) if needed
    import torch.nn as nn
    with torch.no_grad():
        dummy = torch.zeros(1, in_ch, 256, 256)
        out = model(dummy)
        if isinstance(out, dict):
            out = list(out.values())[0]
        base_out_ch = out.shape[1]
    class Wrap(nn.Module):
        def __init__(self, base, base_out_ch, out_ch):
            super().__init__()
            self.base = base
            self.head = nn.Conv2d(base_out_ch, out_ch, 1)
        def forward(self,x):
            y = self.base(x)
            if isinstance(y, dict):
                y = y.get('out', list(y.values())[0])
            return self.head(y)
    model = Wrap(model, base_out_ch, num_classes)

model = model.to(device)
ckpt = torch.load(CKPT, map_location=device)
if 'model_state' in ckpt:
    model.load_state_dict(ckpt['model_state'])
else:
    model.load_state_dict(ckpt)

model.eval()

# metrics: per-class intersection & union
intersection = np.zeros(num_classes, dtype=np.float64)
union = np.zeros(num_classes, dtype=np.float64)
present = np.zeros(num_classes, dtype=np.int64)  # counts where class present (optional)

with torch.no_grad():
    for xb, yb in tqdm(dl, desc="Eval"):
        xb = xb.to(device).float()
        yb = yb.to(device).long()  # shape (N,H,W)
        logits = model(xb)
        if isinstance(logits, dict):
            logits = logits.get('out', list(logits.values())[0])
        preds = logits.argmax(1)  # (N,H,W)
        preds_np = preds.cpu().numpy().astype(np.int32)
        y_np = yb.cpu().numpy().astype(np.int32)
        # compute per-class intersection/union, ignore label 255
        for c in range(num_classes):
            if c == 255:  # skip if 255 used as ignore; num_classes should exclude 255, but keep guard
                continue
            pred_c = (preds_np == c)
            gt_c = (y_np == c)
            inter = np.logical_and(pred_c, gt_c).sum()
            uni = np.logical_or(pred_c, gt_c).sum()
            intersection[c] += inter
            union[c] += uni
            present[c] += gt_c.sum()

# compute IoU
eps = 1e-6
ious = []
for c in range(num_classes):
    if union[c] > 0:
        iou = intersection[c] / (union[c] + eps)
        ious.append(iou)
    else:
        ious.append(np.nan)

# report
mean_iou = np.nanmean(ious)
print(f"\nMean IoU over {num_classes} classes (ignoring empty classes): {mean_iou:.4f}")
# print top/k classes by IoU
for c in range(num_classes):
    if not np.isnan(ious[c]):
        print(f"Class {c:3d}: IoU={ious[c]:.4f}  present_pixels={present[c]}")
