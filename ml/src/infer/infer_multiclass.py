import os, argparse, numpy as np, torch, rasterio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from affine import Affine
from src.models.change_unet_multi import ChangeUNetMulti

class ChipsPredict(Dataset):
    def __init__(self, parquet, band_order=(0,1,2,3)):
        self.df = pd.read_parquet(parquet).reset_index(drop=True)
        self.band_order = band_order

    def __len__(self): 
        return len(self.df)

    def _ensure_CHW_and_select(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            if arr.shape[0] in (3,4):  # already CHW
                chw = arr
            elif arr.shape[-1] in (3,4):  # HWC
                chw = np.moveaxis(arr, -1, 0)
            else:
                raise ValueError(f"Cannot interpret chip shape {arr.shape}")
        else:
            raise ValueError(f"Expected 3D array, got {arr.shape}")
        return chw[np.array(self.band_order), :, :].astype("float32")

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        chip_id = r["chip_id"]

        t0 = self._ensure_CHW_and_select(np.load(r["t0_npy"]))
        t1 = self._ensure_CHW_and_select(np.load(r["t1_npy"]))

        def norm(x):
            x = np.nan_to_num(x, nan=0.0)
            mx = np.percentile(x, 99)
            return (x / max(mx, 1e-6)).clip(0,1)

        return chip_id, float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"]), str(r["crs"]), norm(t0), norm(t1)

def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ChipsPredict(a.parquet)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model = ChangeUNetMulti(in_ch=8, n_classes=3).to(device)  # ✅ 3 classes
    model.load_state_dict(torch.load(a.weights, map_location=device))
    model.eval()

    os.makedirs(a.out_dir, exist_ok=True)

    with torch.no_grad():
        for batch in dl:
            chip_id, xmin, ymin, xmax, ymax, crs, t0, t1 = batch
            chip_id = chip_id[0]
            xmin, ymin, xmax, ymax = map(float, [xmin[0], ymin[0], xmax[0], ymax[0]])
            crs = str(crs[0])

            t0 = t0.to(device)
            t1 = t1.to(device)

            logits = model(t0, t1)  # (1, 3, H, W)
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze().cpu().numpy().astype("uint8")

            # Save prediction as GeoTIFF
            height, width = pred.shape
            resx = (xmax - xmin) / width
            resy = (ymin - ymax) / height
            transform = Affine.translation(xmin, ymax) * Affine.scale(resx, resy)

            prof = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": "uint8",
                "crs": crs,
                "transform": transform
            }
            out_tif = os.path.join(a.out_dir, f"{chip_id}_pred.tif")
            with rasterio.open(out_tif, "w", **prof) as dst:
                dst.write(pred, 1)
            print("✅ wrote", out_tif)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="outputs/chips_index_s2.parquet")
    ap.add_argument("--weights", default="outputs/models_multi/change_unet_multi_best.pth")
    ap.add_argument("--out_dir", default="outputs/preds")
    args = ap.parse_args()
    main(args)
