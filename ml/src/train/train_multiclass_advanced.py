import os, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ---- import your model ----
from src.models.change_unet_multi import ChangeUNetMulti

IGNORE_INDEX = 255
NUM_CLASSES = 3  # 0=road, 1=building, 2=other

# ---------------- Label Remap ----------------
def remap_labels(y: np.ndarray) -> np.ndarray:
    """
    Map raw labels {1,2,3,4,255} → {0,1,2,IGNORE}
      1 → road (0)
      2 → building (1)
      3,4 → other (2)
      255 → IGNORE_INDEX
    """
    out = np.full_like(y, IGNORE_INDEX)
    out[y == 1] = 0  # road
    out[y == 2] = 1  # building
    out[(y == 3) | (y == 4)] = 2  # others
    return out

# ---------------- Dataset ----------------
class ChipsMC(Dataset):
    def __init__(self, parquet, labels_csv, band_order=(0,1,2,3), target_size=256):
        self.df = pd.read_parquet(parquet).reset_index(drop=True)
        lab = pd.read_csv(labels_csv)
        self.labmap = {r["chip_id"]: r["label_npy"] for _, r in lab.iterrows()}
        # keep only rows that have labels
        self.df = self.df[self.df["chip_id"].isin(self.labmap.keys())].reset_index(drop=True)
        self.band_order = tuple(band_order)
        self.target_size = int(target_size)

    def __len__(self): 
        return len(self.df)

    def _ensure_CHW_and_select(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array, got {arr.shape}")
        # channels-first
        if arr.shape[0] in (3,4):
            chw = arr
        elif arr.shape[-1] in (3,4):  # channels-last
            chw = np.moveaxis(arr, -1, 0)
        else:
            raise ValueError(f"Cannot infer channel axis for shape {arr.shape}")
        idx = np.array(self.band_order, dtype=int)
        return chw[idx, :, :].astype("float32")

    def _norm(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0)
        mx = np.percentile(x, 99)
        if not np.isfinite(mx) or mx <= 0:
            mx = 1.0
        return (x / mx).clip(0, 1)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        chip_id = r["chip_id"]

        t0 = self._ensure_CHW_and_select(np.load(r["t0_npy"]))
        t1 = self._ensure_CHW_and_select(np.load(r["t1_npy"]))
        y  = np.load(self.labmap[chip_id]).astype("uint8")  # (H,W) {1,2,3,4,255}

        # remap to {0,1,2,IGNORE}
        y = remap_labels(y)

        t0 = self._norm(t0)
        t1 = self._norm(t1)

        # to torch for resizing
        x0 = torch.from_numpy(t0).unsqueeze(0)       # (1,4,H,W)
        x1 = torch.from_numpy(t1).unsqueeze(0)
        yy = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)

        ts = self.target_size
        x0 = torch.nn.functional.interpolate(x0, size=(ts, ts), mode="bilinear", align_corners=False)
        x1 = torch.nn.functional.interpolate(x1, size=(ts, ts), mode="bilinear", align_corners=False)
        yy = torch.nn.functional.interpolate(yy, size=(ts, ts), mode="nearest")

        x0 = x0.squeeze(0)                   # (4,ts,ts)
        x1 = x1.squeeze(0)                   # (4,ts,ts)
        y  = yy.long().squeeze(0).squeeze(0) # (ts,ts)

        return x0, x1, y

def collate(batch):
    xs0, xs1, ys = zip(*batch)
    return torch.stack(xs0), torch.stack(xs1), torch.stack(ys)

# ------------- helpers -------------
def compute_class_weights(labels_csv: str) -> torch.Tensor:
    lab = pd.read_csv(labels_csv)
    counts = {0:0, 1:0, 2:0}
    for p in lab["label_npy"]:
        y = np.load(p)
        y = remap_labels(y)
        for c in counts.keys():
            counts[c] += int((y == c).sum())
    # log balancing
    wvec = []
    for c in range(NUM_CLASSES):
        freq = counts[c]
        w = 1.0 / np.log(1.02 + freq)
        wvec.append(w)
    wvec = torch.tensor(wvec, dtype=torch.float32)
    print("Class counts:", counts)
    print("Class weights:", wvec)
    return wvec

# ------------- training -------------
def train(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels_csv = a.labels_csv or os.path.join(a.labels_dir, "labels_index.csv")

    # dataset
    ds = ChipsMC(a.parquet, labels_csv, band_order=(0,1,2,3), target_size=a.target_size)
    total_chips = len(pd.read_parquet(a.parquet))
    total_labels = len(pd.read_csv(labels_csv))
    print(f"Using device: {device}")
    print(f"Total chips in parquet: {total_chips}")
    print(f"Total chips with labels: {total_labels}")
    print(f"Common chips (will be used): {len(ds)}")

    # split 80/20
    n = len(ds)
    if n < 2:
        raise ValueError("Not enough labeled samples to train (need >=2).")
    n_val = max(int(0.2 * n), 1)
    n_train = n - n_val
    train_subset = torch.utils.data.Subset(ds, list(range(0, n_train)))
    val_subset   = torch.utils.data.Subset(ds, list(range(n_train, n)))

    tl = DataLoader(train_subset, batch_size=a.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    vl = DataLoader(val_subset,   batch_size=a.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    model = ChangeUNetMulti(in_ch=8, n_classes=NUM_CLASSES).to(device)
    class_weights = compute_class_weights(labels_csv).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    optim = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    def run_epoch(dl, train_mode=True):
        model.train(train_mode)
        total_loss, total_pix = 0.0, 0
        with torch.set_grad_enabled(train_mode):
            for x0, x1, y in dl:
                x0, x1, y = x0.to(device), x1.to(device), y.to(device)
                logits = model(x0, x1)            # (B,C,H,W)
                loss = criterion(logits, y)       # CE with ignore_index
                if train_mode:
                    optim.zero_grad(); loss.backward(); optim.step()
                total_loss += loss.item() * y.numel()
                total_pix  += y.numel()
        return total_loss / max(total_pix, 1)

    best = float("inf")
    os.makedirs(a.out_dir, exist_ok=True)
    for ep in range(1, a.epochs + 1):
        tr = run_epoch(tl, True)
        val = run_epoch(vl, False)
        print(f"Epoch {ep:02d}/{a.epochs}  train_ce={tr:.6f}  val_ce={val:.6f}")
        if val < best:
            best = val
            torch.save(model.state_dict(), os.path.join(a.out_dir, "change_unet_multi_best.pth"))
    print("Saved:", os.path.join(a.out_dir, "change_unet_multi_best.pth"))

# ------------- CLI -------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="outputs/chips_index_s2.parquet")
    ap.add_argument("--labels_dir", default="data/labels/multiclass")
    ap.add_argument("--labels_csv", default="")
    ap.add_argument("--target_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", default="outputs/models_multi")
    args = ap.parse_args()
    train(args)
