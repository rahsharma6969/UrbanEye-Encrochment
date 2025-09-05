# ml/scripts/visualize_predictions.py
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

# === Config ===
CHIP_ID = "s2_0_0"  # ← change to your chip
T0_NPY = f"outputs/chips/{CHIP_ID}_t0.npy"
T1_NPY = f"outputs/chips/{CHIP_ID}_t1.npy"
PRED_TIF = f"outputs/preds_typed/{CHIP_ID}_typed.tif"

# Classes
CLASSES = ["Background", "Building", "Road", "Change"]
COLORS = ["white", "red", "blue", "yellow"]
cmap = ListedColormap(COLORS)

def load_chip(path):
    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        return np.moveaxis(arr, 0, -1)  # CHW → HWC
    return arr

def show():
    if not os.path.exists(PRED_TIF):
        print(f"❌ File not found: {PRED_TIF}")
        return

    # Load data
    t0 = load_chip(T0_NPY)
    t1 = load_chip(T1_NPY)
    with rasterio.open(PRED_TIF) as src:
        pred = src.read(1)

    # Clip to 3-band RGB
    t0_rgb = t0[..., :3] if t0.shape[-1] >= 3 else t0
    t1_rgb = t1[..., :3] if t1.shape[-1] >= 3 else t1

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].imshow(t0_rgb)
    axes[0,0].set_title("Time 0")
    axes[0,1].imshow(t1_rgb)
    axes[0,1].set_title("Time 1")
    im = axes[1,0].imshow(pred, cmap=cmap, vmin=0, vmax=3)
    axes[1,0].set_title("Prediction")
    axes[1,1].imshow(t1_rgb)
    axes[1,1].imshow(pred, cmap=cmap, vmin=0, vmax=3, alpha=0.5)
    axes[1,1].set_title("Overlay (T1 + Pred)")

    # Colorbar
    cbar = plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95, ticks=range(4))
    cbar.ax.set_yticklabels(CLASSES)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show()