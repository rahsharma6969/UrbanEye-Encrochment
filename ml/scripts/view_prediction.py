import matplotlib.pyplot as plt
import numpy as to
from PIL import Image
import pandas as pd
from pathlib import Path

df = pd.read_parquet(r"data\LEVIR_CD\chips_256\index.parquet")
val_df = df[df.split == 'val']

for _, row in val_df.head(5).iterrows():
    base_name = Path(row['mask_npy']).stem
    pred_path = Path(f"outputs/preds/thresholded/{base_name}.npy_th0.30_ma20.png")

    if not pred_path.exists():
        continue

    pred = Image.open(pred_path)
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.imshow(pred, cmap='gray')
    ax.set_title(f"Change Prediction: {base_name}")
    ax.axis('off')

plt.show()