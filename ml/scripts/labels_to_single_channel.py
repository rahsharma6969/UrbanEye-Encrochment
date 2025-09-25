
# ml/scripts/labels_to_single_channel.py
from pathlib import Path
from PIL import Image
import numpy as np

root = Path('C:/Users/Rahul/CollegeProject/UrbanEyeML/ml/data/SVCD/train/label')   # adapt for val/test later
out_dir = root  # overwrite in-place
for p in sorted(root.glob('*')):
    img = np.array(Image.open(p))
    if img.ndim == 3:
        # assume R==G==B or palette -> take first channel
        single = img[...,0].astype('uint8')
    else:
        single = img.astype('uint8')
    # Save as single-channel PNG (values are class ids 0..N)
    Image.fromarray(single).save(out_dir / p.name)
    print("Converted:", p.name)
print("Done.")
