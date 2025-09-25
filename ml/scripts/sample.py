# import pandas as pd, numpy as np, torch
# from src.dataset.change_dataset import ChangeDataset
# from torch.utils.data import DataLoader
# df = pd.read_parquet('data/chips_256/npy/index.parquet')
# df = df[df.split=='train'].reset_index(drop=True)
# ds = ChangeDataset(df)
# dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
# cnt_pos_batches = 0
# for i,(x,y) in enumerate(dl):
#     # y shape maybe (N,H,W) or (N,1,H,W)
#     arr = y.numpy()
#     if arr.ndim==4: arr = arr[:,0]
#     if (arr==1).any():
#         cnt_pos_batches += 1
#     if i>=49: break
# print("Positive-containing batches in first 50 batches:", cnt_pos_batches)

# import pandas as pd, numpy as np
# from src.dataset.change_dataset import ChangeDataset
# from torch.utils.data import DataLoader
# from pathlib import Path
# from src.train.train import BalancedBatchSampler, build_index_from_chips_folder
# df = pd.read_parquet('C:/Users/Rahul/CollegeProject/UrbanEyeML/ml/data/chips_256/npy/index.parquet')
# tr = df[df.split=='train'].reset_index(drop=True)
# bs = 4
# bbs = BalancedBatchSampler(tr, positives_per_batch=1, batch_size=bs)
# ds = ChangeDataset(tr)
# dl = DataLoader(ds, batch_sampler=bbs, num_workers=0)
# pos_batches=0
# total=0
# for batch in dl:
#     x,y = batch
#     y_np = y.numpy()
#     if y_np.ndim==4: y_np=y_np[:,0]
#     if (y_np==1).any(): pos_batches+=1
#     total+=1
#     if total>=50: break
# print("Positive-containing batches in first 50 batches:", pos_batches)



# import torch, numpy as np, pandas as pd
# from src.dataset.change_dataset import ChangeDataset
# from torch.utils.data import DataLoader
# from src.models.unet import UNet
# cfg = __import__('yaml').safe_load(open('configs/svcd_train.yaml'))
# df = pd.read_parquet('data/chips_256/npy/index.parquet')
# val = df[df.split=='val'].reset_index(drop=True)
# from pathlib import Path
# row = val.iloc[0]
# print("sample:", row.mask_npy)
# ds = ChangeDataset(val.iloc[[0]])
# dl = DataLoader(ds, batch_size=1)
# device='cpu'
# model = UNet(in_ch=cfg['model']['in_channels'], base=cfg['model'].get('base_channels',32), out_ch=cfg['model']['num_classes'])
# ck = torch.load('outputs/checkpoints/best.pth', map_location='cpu')
# state = ck.get('model_state', ck)
# new_state={}
# for k,v in state.items():
#     nk=k[len('module.'):] if k.startswith('module.') else k
#     new_state[nk]=v
# model.load_state_dict(new_state)
# model.eval()
# with torch.no_grad():
#     for x,y in dl:
#         logits = model(x.float())
#         if isinstance(logits, dict):
#             logits = logits.get('out', list(logits.values())[0])
#         probs = torch.softmax(logits, dim=1).numpy()
#         print("probs min,max per class:", probs.min(), probs.max())
#         print("argmax unique:", np.unique(probs.argmax(1)))
#         print("gt unique:", np.unique(y.numpy()))
#         break

# import torch, yaml, numpy as np, pandas as pd
# from src.dataset.change_dataset import ChangeDataset
# from torch.utils.data import DataLoader
# from src.models.unet import UNet
# cfg = yaml.safe_load(open("configs/svcd_train.yaml"))
# df = pd.read_parquet("data/chips_256/npy/index.parquet")
# val = df[df.split=="val"].reset_index(drop=True)
# from pathlib import Path
# ds = ChangeDataset(val.iloc[[0]])
# dl = DataLoader(ds, batch_size=1)
# device='cpu'
# model = UNet(in_ch=cfg['model']['in_channels'], base=cfg['model'].get('base_channels',32), out_ch=cfg['model'].get('num_classes',2))
# ck = torch.load("outputs/checkpoints/best.pth", map_location='cpu')
# st = ck.get('model_state', ck)
# ns = {k[len('module.'):]:v for k,v in st.items()} if list(st.keys())[0].startswith('module.') else st
# model.load_state_dict(ns)
# model.eval()
# with torch.no_grad():
#     for x,y in dl:
#         out = model(x.float())
#         if isinstance(out, dict): out = out.get('out', list(out.values())[0])
#         probs = torch.softmax(out, dim=1).cpu().numpy()[0]  # (C,H,W)
#         p0 = probs[0].ravel(); p1 = probs[1].ravel()
#         print("class0: mean,median,max,min", p0.mean(), np.median(p0), p0.max(), p0.min())
#         print("class1: mean,median,max,min", p1.mean(), np.median(p1), p1.max(), p1.min())
#         print("percentage pixels with p1>0.5:", (p1>0.5).mean())
#         break



import yaml, torch
from src.models.unet import UNet
cfg = yaml.safe_load(open("configs/svcd_train.yaml"))
in_ch = cfg["model"]["in_channels"]
out_ch = cfg["model"]["num_classes"]
m = UNet(in_ch=in_ch, base=cfg["model"].get("base_channels",32), out_ch=out_ch)
x = torch.zeros(1, in_ch, 256, 256)
y = m(x)
if isinstance(y, dict):
    y = list(y.values())[0]
print("model out shape:", y.shape)  # expect (1, out_ch, 256,256)

'''
python -m scripts.apply_threshold_postprocess_eval ^
  --config configs/svcd_train.yaml ^
  --index data/chips_256/npy/index.parquet ^
  --ckpt outputs/checkpoints/best.pth ^
  --threshold 0.30 --min_area 20 --device cpu
'''