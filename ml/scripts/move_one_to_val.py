# ml/scripts/move_one_to_val.py
from pathlib import Path
import shutil

src = Path('data/SVCD/train')
dst = Path('data/SVCD/val')
dst.mkdir(parents=True, exist_ok=True)
for sub in ['A','B','label']:
    (dst/sub).mkdir(parents=True, exist_ok=True)

# pick one prefix from A
a_files = sorted((src/'A').glob('*'))
if not a_files:
    print("No files in train/A")
else:
    pick = a_files[0].name  # moves first file; safe to change index
    prefix = Path(pick).stem
    print("Moving prefix:", prefix)
    for sub in ['A','B','label']:
        src_file = src/sub/ pick
        if src_file.exists():
            shutil.move(str(src_file), str(dst/sub/ src_file.name))
    print("Moved one sample to val")
