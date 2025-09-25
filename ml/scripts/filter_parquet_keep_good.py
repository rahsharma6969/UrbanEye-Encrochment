# scripts/filter_parquet_keep_good.py
import pandas as pd
from pathlib import Path

SUMMARY = Path("outputs/chips_inspect_summary.csv")
PARQUET_IN = Path("outputs/mumbai_index.parquet")
PARQUET_OUT = Path("outputs/mumbai_index_clean.parquet")
threshold = 0.5

df_sum = pd.read_csv(SUMMARY)
# map file name -> valid_ratio
vr = df_sum.set_index('name')['valid_ratio'].to_dict()

# load original index
idx = pd.read_parquet(PARQUET_IN)
keep_rows = []
for _, row in idx.iterrows():
    t0_name = Path(row['t0_npy']).name
    t1_name = Path(row['t1_npy']).name
    v0 = float(vr.get(t0_name, 0.0))
    v1 = float(vr.get(t1_name, 0.0))
    if v0 >= threshold and v1 >= threshold:
        keep_rows.append(row)
print(f"Kept {len(keep_rows)} / {len(idx)} rows (threshold {threshold})")
if keep_rows:
    pd.DataFrame(keep_rows).to_parquet(PARQUET_OUT)
    print("Wrote:", PARQUET_OUT)
else:
    print("No rows kept; consider lowering threshold or regenerating chips.")
