# ml/scripts/make_val_split.py
import pandas as pd
from pathlib import Path

index_path = Path("data/chips_256/npy/index.parquet")
df = pd.read_parquet(index_path)

# currently you have 'train' and 'test'
print(df['split'].value_counts())

# take 10% of train rows and mark as val
train_df = df[df.split == 'train']
val_count = max(1, len(train_df) // 10)

val_rows = train_df.sample(val_count, random_state=42).index
df.loc[val_rows, 'split'] = 'val'

# save new index
df.to_parquet(index_path, index=False)
print("New split counts:")
print(df['split'].value_counts())
