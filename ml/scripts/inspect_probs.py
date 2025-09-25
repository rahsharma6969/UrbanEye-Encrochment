# ml/scripts/inspect_probs.py
import torch, numpy as np, pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from src.dataset.change_dataset import ChangeDataset
from src.models.unet import UNet
import yaml
from tqdm import tqdm

cfg = yaml.safe_load(open("configs/svcd_train.yaml"))
idx = pd.read_parquet("data/chips_256/npy/index.parquet")
val = idx[idx.split=="val"].reset_index(drop=True)
ds = ChangeDataset(val)
dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

# load model
import torch
from pathlib import Path
ckpt = Path("outputs/checkpoints/best.pth")
m = torch.load(ckpt, map_location="cpu")
# adapt to your model loader; using same build as train.py would be better
from segmentation_models_pytorch import Unet
model = Unet(encoder_name="resnet34", encoder_weights=None, in_channels=6, classes=2)
model.load_state_dict({k.replace("module.",""):v for k,v in m['model_state'].items()})
model.eval()

all_stats = []
with torch.no_grad():
    for xb, yb in tqdm(dl):
        logits = model(xb.float())
        probs = torch.softmax(logits, dim=1)[:,1,:,:].numpy()
        flat = probs.ravel()
        all_stats.append((float(flat.mean()), float(np.median(flat)), float(flat.max()), float((flat>0.5).mean())))
df = pd.DataFrame(all_stats, columns=["mean","median","max","frac_gt_0.5"])
print(df.describe())
