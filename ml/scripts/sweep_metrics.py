"""
sweep_metrics.py

Usage:
    python sweep_metrics.py --pred_dir ./preds --gt_dir ./gts --out results.csv
    python sweep_metrics.py --pred_dir ./preds --gt_dir ./gts --out outputs/th_minarea_sweep.csv \
    --thresholds 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --min_areas 0 20 50 100 200
    
    
    python scripts/sweep_metrics.py ^
  --pred_dir ./preds ^
  --gt_dir ./gts ^
  --out outputs/th_minarea_sweep.csv ^
  --thresholds 0.3 0.4 0.5 0.6 0.7 0.8 0.9 ^
  --min_areas 0 20 50 100 200 ^
  --save_samples

"""

import os
import argparse
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import io
from skimage.morphology import remove_small_objects
from skimage.color import rgb2gray

# ---------- utils ----------
def load_image(path):
    """Load image and return float array in [0,1]. Handles uint8 and other integer types.
       If RGB, convert to grayscale."""
    arr = io.imread(path)
    if np.issubdtype(arr.dtype, np.integer):
        # scale integer types to [0,1]
        arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
    else:
        arr = arr.astype(np.float32)
    if arr.ndim == 3:
        # convert RGB to grayscale (preserves relative probability if RGB used for probs)
        arr = rgb2gray(arr)
    return np.asarray(arr, dtype=np.float32)

def binarize_and_filter(prob_map, thr, min_area):
    bin_mask = prob_map >= thr
    if min_area and min_area > 0:
        bin_mask = remove_small_objects(bin_mask.astype(bool), min_size=min_area)
    return bin_mask.astype(bool)

def confusion_from_masks(pred_mask, gt_mask):
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
    fn = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
    tn = np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask)).sum()
    return int(tp), int(fp), int(fn), int(tn)

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def compute_metrics(tp, fp, fn, tn):
    prec = safe_div(tp, tp + fp)
    rec  = safe_div(tp, tp + fn)
    f1   = safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
    iou  = safe_div(tp, tp + fp + fn)
    return prec, rec, f1, iou

def normalize_basename(fname):
    """Normalize filename by removing common suffixes like _prob, _cls, _pred, _gt, _mask."""
    s = fname
    for suffix in ("_prob", "_cls", "_pred", "_mask", "_probmap", "_probs"):
        if s.endswith(suffix):
            return s[: -len(suffix)]
    # also strip trailing "_gt" (useful if GTs are named with _gt and preds are not)
    if s.endswith("_gt"):
        return s[: -3]
    return s

def save_sample_masks(out_dir, basename, pred_mask, gt_mask, thr, min_area, idx):
    """Save side-by-side sample visualization (pred, gt) as png."""
    os.makedirs(out_dir, exist_ok=True)
    # convert boolean masks to uint8 images
    pm = (pred_mask.astype(np.uint8) * 255)
    gm = (gt_mask.astype(np.uint8) * 255)
    # create 2-row image: pred on top, gt below
    h, w = pm.shape
    vis = np.zeros((h * 2, w), dtype=np.uint8)
    vis[0:h, :] = pm
    vis[h:, :] = gm
    fname = f"{basename}_thr{thr:.2f}_ma{min_area}_sample{idx}.png"
    io.imsave(os.path.join(out_dir, fname), vis)

# ---------- main ----------
def main(pred_dir, gt_dir, out_csv,
         thresholds = None, min_areas = None,
         save_plots = True, save_samples=False, sample_visualize_n = 4):

    if thresholds is None:
        thresholds = np.linspace(0.1, 0.95, 18)
    else:
        thresholds = np.array(thresholds, dtype=float)

    if min_areas is None:
        min_areas = [0]
    else:
        min_areas = [int(x) for x in min_areas]

    # collect GT lookup
    gt_lookup = {}
    for p in glob(os.path.join(gt_dir, "*")):
        bn = os.path.splitext(os.path.basename(p))[0]
        norm = normalize_basename(bn)
        gt_lookup[norm] = p
        # also store original basename mapping (in case exact matches are used)
        gt_lookup[bn] = p

    # collect prediction files and normalized basenames
    raw_pred_paths = sorted(glob(os.path.join(pred_dir, "*")))
    if len(raw_pred_paths) == 0:
        raise SystemExit(f"No prediction files found in {pred_dir}")

    pred_files = []
    for p in raw_pred_paths:
        bn = os.path.splitext(os.path.basename(p))[0]
        norm = normalize_basename(bn)
        pred_files.append((p, bn, norm))  # store original bn and normalized

    rows = []
    samples_outdir = os.path.join(os.path.dirname(out_csv) or ".", "samples")
    # iterate thresholds and min_areas
    for thr in thresholds:
        for min_area in min_areas:
            TP = FP = FN = TN = 0
            saved_sample_count = 0
            # iterate predictions
            for ppath, orig_bn, norm_bn in tqdm(pred_files, desc=f"thr={thr:.2f}, min_area={min_area}", leave=False):
                # find matching GT: prefer normalized basename then original
                gpath = None
                if norm_bn in gt_lookup:
                    gpath = gt_lookup[norm_bn]
                elif orig_bn in gt_lookup:
                    gpath = gt_lookup[orig_bn]
                else:
                    # no GT for this pred; skip
                    continue

                pred = load_image(ppath)
                gt   = load_image(gpath)
                # ensure same shape -- if not, try to resize? for now skip mismatched shapes
                if pred.shape != gt.shape:
                    # try simple shape check (allow singleton channel)
                    if pred.ndim == 3 and pred.shape[2] == 1:
                        pred = pred[..., 0]
                    if gt.ndim == 3 and gt.shape[2] == 1:
                        gt = gt[..., 0]
                    if pred.shape != gt.shape:
                        # skip and warn
                        print(f"Skipping {ppath} because shape {pred.shape} != GT {gpath} shape {gt.shape}")
                        continue

                gt_mask = gt >= 0.5
                pred_mask = binarize_and_filter(pred, thr, min_area)
                tp, fp, fn, tn = confusion_from_masks(pred_mask, gt_mask)
                TP += tp; FP += fp; FN += fn; TN += tn

                # optionally save a few sample visualizations (pred above, gt below)
                if save_samples and saved_sample_count < sample_visualize_n:
                    try:
                        save_sample_masks(samples_outdir, normalize_basename(orig_bn), pred_mask, gt_mask, thr, min_area, saved_sample_count)
                        saved_sample_count += 1
                    except Exception as e:
                        print("Failed saving sample visualization:", e)

            prec, rec, f1, iou = compute_metrics(TP, FP, FN, TN)
            rows.append({
                "threshold": float(thr),
                "min_area": int(min_area),
                "tp": TP, "fp": FP, "fn": FN, "tn": TN,
                "precision": prec, "recall": rec, "f1": f1, "iou": iou
            })
            print(f"[thr={thr:.2f} ma={min_area}] TP={TP} FP={FP} FN={FN} TN={TN} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} iou={iou:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

    # ---------- plotting ----------
    if save_plots:
        plt.rcParams.update({'figure.max_open_warning': 0})
        if len(min_areas) == 1:
            ma = min_areas[0]
            d = df[df["min_area"] == ma].sort_values("threshold")
            plt.figure(figsize=(8,5))
            plt.plot(d["threshold"], d["precision"], marker='o', label="Precision")
            plt.plot(d["threshold"], d["recall"], marker='o', label="Recall")
            plt.plot(d["threshold"], d["f1"], marker='o', label="F1")
            plt.plot(d["threshold"], d["iou"], marker='o', label="IoU")
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.title(f"Metrics vs Threshold (min_area={ma})")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            ppath = os.path.splitext(out_csv)[0] + f"_metrics_minarea_{ma}.png"
            plt.savefig(ppath, dpi=150)
            print("Saved plot:", ppath)
            plt.close()
        else:
            pivot = df.pivot(index="min_area", columns="threshold", values="f1")
            plt.figure(figsize=(10,6))
            plt.imshow(pivot.values, aspect='auto', origin='lower')
            plt.colorbar(label='F1')
            plt.yticks(range(len(pivot.index)), pivot.index)
            plt.xticks(range(len(pivot.columns)), [f"{v:.2f}" for v in pivot.columns], rotation=45)
            plt.xlabel("Threshold")
            plt.ylabel("min_area")
            plt.title("F1 heatmap (min_area vs threshold)")
            plt.tight_layout()
            hpath = os.path.splitext(out_csv)[0] + f"_f1_heatmap.png"
            plt.savefig(hpath, dpi=150)
            print("Saved heatmap:", hpath)
            plt.close()

    if save_samples:
        print(f"Saved sample visualizations to {samples_outdir}")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, help="Directory with predicted probability maps (npy/png/tif).")
    parser.add_argument("--gt_dir", required=True, help="Directory with ground-truth binary masks.")
    parser.add_argument("--out", dest="out_csv", default="sweep_results.csv", help="Output CSV path.")
    parser.add_argument("--thresholds", nargs="+", type=float, help="List of thresholds to try, e.g. 0.1 0.2 0.3")
    parser.add_argument("--min_areas", nargs="+", type=int, help="List of min_area values to try, e.g. 0 20 50 100")
    parser.add_argument("--save_samples", action="store_true", help="Save a few sample binarized masks for visual debugging.")
    parser.add_argument("--sample_n", type=int, default=4, help="Number of sample visualizations to save per thr/min_area (default 4).")
    args = parser.parse_args()

    thresholds = np.array(args.thresholds) if args.thresholds else None
    min_areas = args.min_areas if args.min_areas else None

    main(pred_dir=args.pred_dir, gt_dir=args.gt_dir, out_csv=args.out_csv,
         thresholds=thresholds, min_areas=min_areas,
         save_plots=True, save_samples=args.save_samples, sample_visualize_n=args.sample_n)