# ml/scripts/eval_binary.py
import torch, pandas as pd, numpy as np
from torch.utils.data import DataLoader
from src.dataset.change_dataset import ChangeDataset
from src.models.unet import UNet
import yaml
from pathlib import Path

cfg = yaml.safe_load(open("configs/svcd_train.yaml"))
device = torch.device("cuda" if torch.cuda.is_available() and cfg['model'].get('device','auto')!='cpu' else "cpu")
print("Device:", device)

df = pd.read_parquet('data/chips_256/npy/index.parquet')
test_df = df[df.split=='test'].reset_index(drop=True)
ds = ChangeDataset(test_df)
dl = DataLoader(ds, batch_size=cfg['model'].get('batch_size',4), num_workers=2)

in_ch = int(cfg['model'].get('in_channels',6))
base_ch = int(cfg['model'].get('base_channels',32))
num_classes = int(cfg['model'].get('num_classes',2))
try:
    model = UNet(in_ch=in_ch, base=base_ch, out_ch=num_classes)
except TypeError:
    model = UNet(in_ch=in_ch, base=base_ch)
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
ckpt = Path('outputs/checkpoints/best.pth')
if not ckpt.exists():
    ckpt = Path('outputs/checkpoints/latest.pth')
if not ckpt.exists():
    raise SystemExit("No checkpoint found in outputs/checkpoints")

sd = torch.load(ckpt, map_location=device)
model.load_state_dict(sd.get('model_state', sd))
model.eval()

tp = fp = fn = tn = 0
with torch.no_grad():
    for xb, yb in dl:
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        logits = model(xb)
        if isinstance(logits, dict):
            logits = logits.get('out', list(logits.values())[0])
        preds = logits.argmax(1).cpu().numpy()
        gts = yb.cpu().numpy()
        # ignore 255
        mask_valid = (gts != 255)
        preds = preds[mask_valid]
        gts = gts[mask_valid]
        tp += ((preds==1) & (gts==1)).sum()
        tn += ((preds==0) & (gts==0)).sum()
        fp += ((preds==1) & (gts==0)).sum()
        fn += ((preds==0) & (gts==1)).sum()

# compute metrics
eps = 1e-6
precision = tp / (tp + fp + eps)
recall = tp / (tp + fn + eps)
f1 = 2*precision*recall / (precision + recall + eps)
iou = tp / (tp + fp + fn + eps)
acc = (tp + tn) / (tp + tn + fp + fn + eps)
print(f"TP {tp}, FP {fp}, FN {fn}, TN {tn}")
print(f"Precision {precision:.4f}, Recall {recall:.4f}, F1 {f1:.4f}, IoU {iou:.4f}, Acc {acc:.4f}")
