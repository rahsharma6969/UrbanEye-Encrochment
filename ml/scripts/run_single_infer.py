# scripts/run_single_infer.py
import os, sys, argparse
import numpy as np
import torch
import rasterio
from affine import Affine
from src.models.change_unet_multi import ChangeUNetMulti

def save_tif(path, arr, transform, crs, dtype="float32"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prof = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform
    }
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", required=True, help="path to t0 .npy")
    ap.add_argument("--t1", required=True, help="path to t1 .npy")
    ap.add_argument("--chip_id", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out_dir", default="outputs/preds")
    ap.add_argument("--out_typed", default="outputs/preds_typed")
    ap.add_argument("--crs", default="EPSG:32642")
    ap.add_argument("--xmin", type=float, required=True)
    ap.add_argument("--ymin", type=float, required=True)
    ap.add_argument("--xmax", type=float, required=True)
    ap.add_argument("--ymax", type=float, required=True)
    args = ap.parse_args()

    x0 = np.load(args.t0).astype("float32")  # expected (4,H,W)
    x1 = np.load(args.t1).astype("float32")
    # normalization used by your dataset (percentile 99)
    def norm(x):
        x = np.nan_to_num(x, nan=0.0)
        mx = np.percentile(x, 99)
        if mx <= 0 or not np.isfinite(mx): mx = 1.0
        return (x / mx).clip(0,1)
    x0 = norm(x0)
    x1 = norm(x1)

    # convert to torch with batch dim
    t0 = torch.from_numpy(x0).unsqueeze(0)
    t1 = torch.from_numpy(x1).unsqueeze(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChangeUNetMulti(in_ch=8, n_classes=5).to(device)
    sd = torch.load(args.weights, map_location=device)
    # support both raw state_dict and checkpoint dict
    if isinstance(sd, dict) and "model_state_dict" in sd:
        model.load_state_dict(sd["model_state_dict"])
    else:
        model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        logits = model(t0.to(device), t1.to(device))  # (1,C,H,W)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (C,H,W)
        change_prob = probs[1:].sum(axis=0)  # sum probs for non-background classes (or use specific index)
        pred = probs.argmax(axis=0).astype("uint8")

    # build transform
    width = change_prob.shape[1]
    height = change_prob.shape[0]
    resx = (args.xmax - args.xmin) / width
    resy = (args.ymin - args.ymax) / height
    transform = Affine.translation(args.xmin, args.ymax) * Affine.scale(resx, resy)

    out_prob = os.path.join(args.out_dir, f"{args.chip_id}_change.tif")
    out_typed = os.path.join(args.out_typed, f"{args.chip_id}_typed.tif")
    save_tif(out_prob, change_prob, transform, args.crs, dtype="float32")
    save_tif(out_typed, pred, transform, args.crs, dtype="uint8")
    print("Wrote:", out_prob, out_typed)

if __name__ == "__main__":
    main()


'''
python -m scripts.run_single_infer ^
  --t0 "data\chips_256\s2_t0_0_50.npy" ^
  --t1 "data\chips_256\s2_t1_0_50.npy" ^
  --chip_id s2_0_50 ^
  --weights "outputs\models_multi\change_unet_multi_best.pth" ^
  --xmin 802540 --ymin 2175700 --xmax 805100 --ymax 2178260
'''