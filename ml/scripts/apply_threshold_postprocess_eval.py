import argparse
from pathlib import Path
import yaml, torch, numpy as np, pandas as pd
from tqdm import tqdm
from PIL import Image
from src.dataset.change_dataset import ChangeDataset
from torch.utils.data import DataLoader
import os

# Try optional imports
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(ckpt, in_ch, base_ch, num_classes, device):
    """
    Load model using smp.Unet (must match training architecture).
    Assumes model was trained with: smp.Unet(encoder='resnet34', in_channels=6, classes=2)
    """
    if not SMP_AVAILABLE:
        raise RuntimeError(
            "segmentation_models_pytorch (smp) is required for inference. "
            "Install via: pip install segmentation-models-pytorch"
        )

    print("Loading model as smp.Unet (encoder=resnet34)...")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=in_ch,
        classes=num_classes,
    ).to(device)

    ck = torch.load(ckpt, map_location=device)

    # Handle different checkpoint formats
    if isinstance(ck, dict):
        if 'model_state' in ck:
            st = ck['model_state']
        elif 'state_dict' in ck:
            st = ck['state_dict']
        elif 'model_state_dict' in ck:
            st = ck['model_state_dict']
        else:
            st = ck
    else:
        st = ck

    # Remove 'module.' prefix if present (DataParallel)
    new_st = {}
    for k, v in st.items():
        nk = k[len("module."):] if isinstance(k, str) and k.startswith("module.") else k
        new_st[nk] = v

    model.load_state_dict(new_st)
    model.eval()
    print(f"✅ Successfully loaded checkpoint from {ckpt}")
    return model


def remove_small_objects(bin_mask, min_size):
    """
    bin_mask: numpy uint8 or bool (0/1) shape (H,W)
    returns: numpy uint8 (0/1)
    """
    bin_mask = (bin_mask > 0).astype(np.uint8)
    if min_size <= 0:
        return bin_mask

    # Prefer OpenCV if available
    if _HAS_CV2:
        img255 = (bin_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(img255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(bin_mask, dtype=np.uint8)
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_size:
                cv2.drawContours(out, [c], -1, 1, thickness=-1)
        return out

    # fallback: scipy.ndimage if available
    try:
        from scipy import ndimage
        labeled, n = ndimage.label(bin_mask)
        out = np.zeros_like(bin_mask, dtype=np.uint8)
        for lab in range(1, n + 1):
            area = int((labeled == lab).sum())
            if area >= min_size:
                out[labeled == lab] = 1
        return out
    except Exception:
        # final slow fallback: naive connected components using BFS (safe)
        H, W = bin_mask.shape
        out = np.zeros_like(bin_mask, dtype=np.uint8)
        visited = np.zeros_like(bin_mask, dtype=bool)
        neigh = [(1,0),(-1,0),(0,1),(0,-1)]
        for y in range(H):
            for x in range(W):
                if bin_mask[y,x] and not visited[y,x]:
                    stack = [(y,x)]
                    comp = []
                    visited[y,x] = True
                    while stack:
                        yy, xx = stack.pop()
                        comp.append((yy, xx))
                        for dy, dx in neigh:
                            ny, nx = yy+dy, xx+dx
                            if 0 <= ny < H and 0 <= nx < W and not visited[ny,nx] and bin_mask[ny,nx]:
                                visited[ny,nx] = True
                                stack.append((ny,nx))
                    if len(comp) >= min_size:
                        for (yy, xx) in comp:
                            out[yy, xx] = 1
        return out


def main(args):
    cfg = load_cfg(args.config)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    idx = Path(args.index)
    assert idx.exists(), "index parquet not found"
    df = pd.read_parquet(idx)
    val_df = df[df.split == 'val'].reset_index(drop=True)
    if val_df.empty:
        raise SystemExit("No val rows in index.")

    index_dir = idx.parent  # e.g., data\LEVIR_CD\chips_256
    ds = ChangeDataset(val_df, base_dir=index_dir)
    dl = DataLoader(ds, batch_size=args.batch_size or 4, shuffle=False, num_workers=0)
    
    # Updated to handle both nested and flat config structures
    if "model" in cfg:
        # Nested structure
        in_ch = cfg["model"].get("in_channels", 6)
        base_ch = cfg["model"].get("base_channels", 32)  # unused but kept for compatibility
        num_classes = cfg["model"].get("classes", cfg["model"].get("num_classes", 2))
    else:
        # Flat structure (your current config)
        in_ch = cfg.get("in_channels", 6)
        base_ch = 32  # default value
        num_classes = cfg.get("classes", 2)

    model = load_model(args.ckpt, in_ch, base_ch, num_classes, device)

    out_dir = Path("outputs/preds/thresholded")
    out_dir.mkdir(parents=True, exist_ok=True)

    IGNORE = 255
    th = float(args.threshold)
    min_area = int(args.min_area)

    # Morphological kernels (only if cv2 available)
    ker_open = ker_close = None
    if _HAS_CV2:
        ker_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ker_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    tp = fp = fn = tn = 0
    global_idx = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(dl, desc="Val inference")):
            x = x.to(device).float()
            logits = model(x)
            if isinstance(logits, dict):
                logits = logits.get("out", list(logits.values())[0])
            probs = torch.softmax(logits, dim=1)[:, 1, :, :].cpu().numpy()  # (N,H,W)
            gts = y.numpy()
            if gts.ndim == 4:
                gts = gts[:, 0, :, :]

            batch_size = probs.shape[0]
            for j in range(batch_size):
                p_map = probs[j]
                gt = gts[j].astype(np.int32)

                # Step 1: Threshold
                bin_mask = (p_map >= th).astype(np.uint8)

                # Step 2: Morphological opening/closing (denoise)
                if _HAS_CV2:
                    # Convert to 0-255 for morphology
                    bin_255 = (bin_mask * 255).astype(np.uint8)
                    bin_255 = cv2.morphologyEx(bin_255, cv2.MORPH_OPEN, ker_open)
                    bin_255 = cv2.morphologyEx(bin_255, cv2.MORPH_CLOSE, ker_close)
                    bin_mask = (bin_255 > 127).astype(np.uint8)

                # Step 3: Remove small objects
                if min_area > 0:
                    try:
                        bin_mask = remove_small_objects(bin_mask, min_area)
                    except Exception as e:
                        print("Postprocessing error:", e)

                # Step 4: Evaluate against ground truth (ignore 255)
                valid = (gt != IGNORE)
                if valid.sum() == 0:
                    global_idx += 1
                    continue

                p = bin_mask[valid].ravel()
                g = gt[valid].ravel()

                tp += int(((p == 1) & (g == 1)).sum())
                tn += int(((p == 0) & (g == 0)).sum())
                fp += int(((p == 1) & (g == 0)).sum())
                fn += int(((p == 0) & (g == 1)).sum())

                # Step 5: Save mask
                row_path = Path(val_df.iloc[global_idx]['mask_npy'])
                fname = row_path.name
                out_name = f"{fname}_th{th:.2f}_ma{min_area}.png"
                Image.fromarray((bin_mask * 255).astype('uint8')).save(out_dir / out_name)

                global_idx += 1

    # Compute metrics
    eps = 1e-9
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    print("Results (threshold, min_area) =", th, min_area)
    print("TP", tp, "FP", fp, "FN", fn, "TN", tn)
    print(f"prec {prec:.4f} rec {rec:.4f} f1 {f1:.4f} iou {iou:.4f} acc {acc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--threshold", type=float, default=0.4)   # ← Updated default to 0.4
    ap.add_argument("--min_area", type=int, default=20, help="remove components smaller than this (px)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    main(args)
    
    '''
    python -m scripts.apply_threshold_postprocess_eval ^
  --config configs/svcd_train_improved.yaml ^
  --index data/chips_256/npy/index.parquet ^
  --ckpt outputs/checkpoints/best.pth ^
  --threshold 0.35 ^
  --min_area 100 ^
  --device cpu
  '''