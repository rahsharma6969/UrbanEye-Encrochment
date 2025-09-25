# scripts/sweep_threshold_minarea.py
import argparse, numpy as np, pandas as pd, torch
from pathlib import Path
from src.dataset.change_dataset import ChangeDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from src.models.unet import UNet

def load_model_ckpt(ckpt, in_ch, base_ch, num_classes, device):
    import torch
    try:
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=in_ch, classes=num_classes)
    except Exception:
        model = UNet(in_ch=in_ch, base=base_ch, out_ch=num_classes)
    ck = torch.load(ckpt, map_location=device)
    if isinstance(ck, dict) and 'model_state' in ck:
        st = ck['model_state']
    elif isinstance(ck, dict) and 'state_dict' in ck:
        st = ck['state_dict']
    else:
        st = ck
    # strip module.
    new = {k[len("module."):]:v for k,v in st.items()} if any(k.startswith("module.") for k in st) else st
    model.load_state_dict(new)
    model.to(device).eval()
    return model

def eval_with_params(model, dl, device, threshold, min_area, ignore_index=255):
    tp=fp=fn=tn=0
    with torch.no_grad():
        for xb,yb in tqdm(dl, desc="Eval"):
            xb = xb.to(device).float()
            logits = model(xb)
            if isinstance(logits, dict):
                logits = logits.get("out", list(logits.values())[0])
            probs = torch.softmax(logits, dim=1)[:,1,:,:].cpu().numpy()
            gts = yb.numpy()
            if gts.ndim==4: gts = gts[:,0,:,:]
            for j in range(probs.shape[0]):
                p = (probs[j] >= threshold).astype(np.uint8)
                gt = gts[j]
                valid = gt != ignore_index
                if valid.sum()==0: continue
                pv = p[valid].ravel()
                gv = gt[valid].ravel()
                tp += int(((pv==1)&(gv==1)).sum())
                fp += int(((pv==1)&(gv==0)).sum())
                fn += int(((pv==0)&(gv==1)).sum())
                tn += int(((pv==0)&(gv==0)).sum())
    eps=1e-9
    prec = tp/(tp+fp+eps); rec = tp/(tp+fn+eps); f1 = 2*prec*rec/(prec+rec+eps); iou = tp/(tp+fp+fn+eps)
    return {"threshold":threshold,"min_area":min_area,"tp":tp,"fp":fp,"fn":fn,"tn":tn,"prec":prec,"rec":rec,"f1":f1,"iou":iou}

def main(cfg, index, ckpt, device, batch_size):
    import yaml
    cfgd = yaml.safe_load(open(cfg))
    df = pd.read_parquet(index)
    val_df = df[df.split=='val'].reset_index(drop=True)
    ds = ChangeDataset(val_df)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    num_classes = cfgd["model"].get("num_classes", 2)
    in_ch = cfgd["model"].get("in_channels", 6)
    base_ch = cfgd["model"].get("base_channels", 32)
    device = torch.device(device)
    model = load_model_ckpt(ckpt, in_ch, base_ch, num_classes, device)
    out = []
    for th in np.linspace(0.1,0.9,9):
        for ma in [0,20,50,100,200]:
            stats = eval_with_params(model, dl, device, th, ma)
            out.append(stats)
    df_out = pd.DataFrame(out).sort_values("f1", ascending=False)
    out_path = Path("outputs/logs/th_minarea_sweep.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print("Wrote:", out_path)
    print(df_out.head(5))

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()
    main(args.config, args.index, args.ckpt, args.device, args.batch_size)
