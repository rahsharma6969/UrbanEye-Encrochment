# ml/scripts/svcd_train_val_test_split.py
from pathlib import Path
import random, shutil

SRC = Path('ml/data/SVCD/train')   # current folder with A/B/label
DST = Path('ml/data/SVCD')
DST.mkdir(parents=True, exist_ok=True)

prefixes = [p.stem for p in (SRC/'A').glob('*')]
prefixes = sorted(prefixes)
random.seed(42)
random.shuffle(prefixes)

n = len(prefixes)
n_train = int(0.8*n)
n_val = int(0.1*n)
train = prefixes[:n_train]
val = prefixes[n_train:n_train+n_val]
test = prefixes[n_train+n_val:]

def copy_list(lst, split):
    for pref in lst:
        for sub, suf in [('A','A'), ('B','B'), ('label','label')]:
            src_folder = SRC / sub
            dst_folder = DST / split / sub
            dst_folder.mkdir(parents=True, exist_ok=True)
            # find matching file (starts with prefix)
            found = list(src_folder.glob(f"{pref}*"))
            if not found:
                print("Missing for", pref, sub)
            for f in found:
                shutil.copy2(f, dst_folder / f.name)

copy_list(train, 'train')
copy_list(val, 'val')
copy_list(test, 'test')

print("Counts -> train,val,test:", len(train), len(val), len(test))
