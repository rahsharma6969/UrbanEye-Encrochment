import numpy as np
import torch
from pathlib import Path
import pandas as pd
from src.dataset.change_dataset import ChangeDataset
from torch.utils.data import DataLoader
import yaml
import segmentation_models_pytorch as smp

cfg = yaml.safe_load(open("configs/svcd_train_improved.yaml", "r"))
device = torch.device("cpu")

# Load model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=cfg["model"]["in_channels"],
    classes=cfg["model"]["num_classes"],
).to(device)

ckpt = torch.load("outputs/checkpoints/best.pth", map_location=device)
state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
new_state = {}
for k, v in state.items():
    nk = k[len("module."):] if k.startswith("module.") else k
    new_state[nk] = v
model.load_state_dict(new_state)
model.eval()

# Load data
df = pd.read_parquet("data/chips_256/npy/index.parquet")
val_df = df[df.split == 'val']
ds = ChangeDataset(val_df)
dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

all_probs = []

with torch.no_grad():
    for xb, yb in dl:
        xb = xb.to(device).float()
        logits = model(xb)
        if isinstance(logits, dict):
            logits = logits.get("out", list(logits.values())[0])
        probs = torch.softmax(logits, dim=1)[:,1,:,:].cpu().numpy()
        all_probs.append(probs)

all_probs = np.concatenate(all_probs, axis=0)
print(f"Global max probability: {all_probs.max():.6f}")
print(f"Global min probability: {all_probs.min():.6f}")
print(f"Global mean probability: {all_probs.mean():.6f}")
print(f"Percent of pixels > 0.5: {(all_probs > 0.5).sum() / all_probs.size * 100:.4f}%")