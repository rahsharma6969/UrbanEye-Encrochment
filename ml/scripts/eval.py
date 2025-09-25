import torch
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet
from src.dataset.change_dataset import ChangeDataset
import pandas as pd
import yaml
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_classes = pred.argmax(dim=1).cpu().numpy().flatten()
            labels = y.cpu().numpy().flatten()
            
            # Filter out ignore_index (e.g., 255)
            valid_idx = labels != 255
            all_preds.append(pred_classes[valid_idx])
            all_labels.append(labels[valid_idx])
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return preds, labels

def main(config_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_parquet(cfg['val_index'])
    val_ds = ChangeDataset(df[df.split=='val'].reset_index(drop=True))
    val_dl = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)

    model = Unet(
        encoder_name=cfg['encoder'],
        in_channels=cfg['in_channels'],
        classes=cfg['classes'],
        encoder_weights=None
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    preds, labels = evaluate(model, val_dl, device)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python eval.py <config.yaml> <checkpoint.pth>")
        exit(1)
    main(sys.argv[1], sys.argv[2])
