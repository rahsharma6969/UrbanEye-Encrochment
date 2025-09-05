# ml/scripts/evaluate.py

import argparse, os, numpy as np, torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, jaccard_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
from skimage.transform import resize

IGNORE_INDEX = 255
CLASSES = ["road", "building", "other"]  # our 3 classes
LABELS = [0, 1, 2]

# === Dataset Loader ===
class LabeledChips(Dataset):
    def __init__(self, parquet, labels_dir, band_order=(0,1,2,3)):
        self.df = pd.read_parquet(parquet).reset_index(drop=True)
        self.labels_dir = labels_dir
        self.band_order = band_order

        # only keep chips with labels
        self.df = self.df[self.df["chip_id"].apply(
            lambda cid: os.path.exists(os.path.join(labels_dir, f"{cid}_label.npy"))
        )]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        chip_id = r["chip_id"]

        # Load chips
        t0 = np.load(r["t0_npy"]).astype("float32")
        t1 = np.load(r["t1_npy"]).astype("float32")

        if t0.ndim == 3: t0 = np.transpose(t0, (2, 0, 1))  # HWC -> CHW
        if t1.ndim == 3: t1 = np.transpose(t1, (2, 0, 1))

        # Band selection
        t0 = t0[list(self.band_order)]
        t1 = t1[list(self.band_order)]

        def norm(x):
            x = np.nan_to_num(x, nan=0.0)
            mx = np.percentile(x, 99)
            return (x / max(mx, 1e-6)).clip(0, 1)

        y = np.load(os.path.join(self.labels_dir, f"{chip_id}_label.npy")).astype("int64")

        return torch.tensor(norm(t0)), torch.tensor(norm(t1)), torch.tensor(y), chip_id

# === Evaluation Function ===
def evaluate(args):
    from src.models.change_unet_multi import ChangeUNetMulti

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ds = LabeledChips(args.parquet, args.labels_dir)
    if len(ds) == 0:
        raise ValueError(f"No labeled chips found in {args.labels_dir}")

    dl = DataLoader(ds, batch_size=1, shuffle=False)

    # load model (3 classes now)
    model = ChangeUNetMulti(in_ch=8, n_classes=3).to(device)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for t0, t1, y, chip_id in dl:
            t0, t1, y = t0.to(device), t1.to(device), y.to(device)

            logits = model(t0, t1)
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze().cpu().numpy()
            label = y.squeeze().cpu().numpy()

            if pred.shape != label.shape:
                label = resize(label, pred.shape, order=0,
                               preserve_range=True, anti_aliasing=False).astype(np.int64)

            # mask ignore pixels
            mask = (label != IGNORE_INDEX)
            all_preds.extend(pred[mask].flatten())
            all_labels.extend(label[mask].flatten())

    # === Metrics ===
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        labels=LABELS,
        target_names=CLASSES,
        digits=3,
        zero_division=0
    ))

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Accuracy: {acc:.3f}")

    cm = confusion_matrix(all_labels, all_preds, labels=LABELS)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    os.makedirs(args.out_dir, exist_ok=True)
    plt.savefig(os.path.join(args.out_dir, "confusion_matrix.png"))
    plt.close()

    iou = jaccard_score(all_labels, all_preds, average=None, labels=LABELS)
    for i, cls in enumerate(CLASSES):
        print(f"IoU for {cls}: {iou[i]:.3f}")
    print(f"Mean IoU: {np.nanmean(iou):.3f}")

    pd.DataFrame({"class": CLASSES, "IoU": iou}).to_csv(
        os.path.join(args.out_dir, "metrics.csv"), index=False
    )
    print(f"\n✅ Saved metrics → {args.out_dir}/metrics.csv")

# === CLI ===
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="outputs/chips_index_s2.parquet")
    ap.add_argument("--labels_dir", default="data/labels/multiclass")
    ap.add_argument("--weights", default="outputs/models_multi/change_unet_multi_best.pth")
    ap.add_argument("--out_dir", default="outputs/eval")
    args = ap.parse_args()
    evaluate(args)



    
    
    
    





# python -m scripts.evaluate_model ^
#   --parquet outputs/chips_index_s2.parquet ^
#   --labels_dir data/labels/multiclass_4class ^
#   --weights outputs/models_multi/change_unet_multi_best.pth ^
#   --out_dir outputs/eval

# python -m scripts.evaluate_model ^
#   --parquet outputs/chips_index_s2.parquet ^
#   --labels_dir data/labels/multiclass_4class ^
#   --weights outputs/models_multi/change_unet_multi_best.pth ^
#   --out_dir outputs/eval
