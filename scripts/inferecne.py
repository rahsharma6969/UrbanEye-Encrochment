# ml/scripts/inference.py
"""
Run inference using your trained smp.Unet model
"""

import torch
import segmentation_models_pytorch as smp
import numpy as  np


def run_inference(t0, t1):
    """
    Run trained model on two normalized images.
    Input: t0, t1 — (H,W,3), float32 [0,1]
    Output: binary mask — (H,W), uint8 {0,1}
    """
    # Load model
    model = smp.Unet(
        encoder_name="resnet34",
        in_channels=6,
        classes=2,
        encoder_weights=None
    ).to("cpu")

    ckpt = torch.load("outputs/checkpoints/best.pth", map_location="cpu")
    state = ckpt.get('model_state', ckpt)

    new_state = {}
    for k, v in state.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        new_state[nk] = v

    model.load_state_dict(new_state)
    model.eval()

    # Stack into 6-channel input
    x = np.concatenate([t0, t1], axis=-1)  # (H,W,6)
    x = np.transpose(x, (2,0,1))           # (6,H,W)
    x = torch.from_numpy(x).unsqueeze(0).float().to("cpu")

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0,1].cpu().numpy()  # (H,W)

    # Apply threshold and post-processing
    th = 0.3
    min_area = 20
    bin_mask = (probs >= th).astype(np.uint8)

    # Morphology to remove noise
    try:
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        bin_mask = cv2.morphologyEx(bin_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
    except ImportError:
        pass

    # Remove small components
    try:
        from scipy import ndimage
        labels, _ = ndimage.label(bin_mask)
        out = np.zeros_like(bin_mask)
        for i in range(1, _+1):
            area = (labels == i).sum()
            if area >= min_area:
                out[labels == i] = 1
        bin_mask = out
    except ImportError:
        pass

    return bin_mask