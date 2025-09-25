from sklearn.metrics import f1_score
import numpy as np

def f1_score_threshold(pred_probs, target, threshold=0.5, ignore_index=255):
    """
    Compute F1 score between binarized prediction and target mask.
    Params:
        pred_probs: np.array, predicted probabilities (H, W)
        target: np.array, ground truth mask (H, W)
        threshold: float for binarization
        ignore_index: int for pixels to ignore
    Returns:
        f1: float F1 score ignoring ignore_index pixels
    """
    pred_bin = (pred_probs >= threshold).astype(np.uint8)
    mask = (target != ignore_index)
    return f1_score(target[mask].flatten(), pred_bin[mask].flatten())
