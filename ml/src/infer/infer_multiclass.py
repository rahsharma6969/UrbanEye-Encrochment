import os, argparse, numpy as np, torch, rasterio
from torch.utils.data import Dataset, DataLoader
from src.model.change_unet_multi import ChangeUNetMulti
import pandas as pd

class ChipsPredict(Dataset):
    def __init__(self, parquet, band_order=(0,1,2,3)):
        self.df = pd.read_parquet(parquet).reset_index(drop=True)
        self.band_order = band_order
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        t0 = np.load(r["t0_npy"]).astype("float32")[self.band_order]
        t1 = np.load(r["t1_npy"]).astype("float32")[self.band_order]
        def norm(x):
            x = np.nan_to_num(x, nan=0.0)
            mx = np.percentile(x, 99); return (x/max(mx,1e-6)).clip(0,1)
        return (r["chip_id"], r["xmin"], r["ymin"], r["xmax"], r["ymax"], r["crs"],
                norm(t0), norm(t1))

def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ChipsPredict(a.parquet)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    model = ChangeUNetMulti(in_ch=8, n_classes=5).to(device)
    model.load_state_dict(torch.load(a.weights, map_location=device))
    model.eval()

    os.makedirs(a.out_dir, exist_ok=True)
    for (chip_id,xmin,ymin,xmax,ymax,crs,t0,t1) in dl:
        chip_id = chip_id[0]
        xmin,ymin,xmax,ymax = [float(v) for v in (xmin,ymin,xmax,ymax)]
        t0 = torch.from_numpy(t0.numpy()[0]).unsqueeze(0).to(device) # (1,4,H,W)
        t1 = torch.from_numpy(t1.numpy()[0]).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(t0,t1)          # (1,C,H,W)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (C,H,W)
            pred  = np.argmax(probs, axis=0).astype("uint8")       # (H,W)
        # save typed raster (uint8 class)
        height, width = pred.shape
        from affine import Affine
        resx = (xmax - xmin) / width
        resy = (ymin - ymax) / height
        transform = Affine.translation(xmin, ymax) * Affine.scale(resx, resy)
        prof = {"driver":"GTiff","height":height,"width":width,"count":1,"dtype":"uint8","crs":crs,"transform":transform}
        out_tif = os.path.join(a.out_dir, f"{chip_id}_typed.tif")
        with rasterio.open(out_tif, "w", **prof) as dst:
            dst.write(pred, 1)
        print("wrote", out_tif)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="outputs/chips_index_s2.parquet")
    ap.add_argument("--weights", default="outputs/models_multi/change_unet_multi_best.pth")
    ap.add_argument("--out_dir", default="outputs/preds_typed")
    args = ap.parse_args()
    main(args)
