# scripts/check_label_stats.py
import os
import numpy as np
import pandas as pd
import argparse

def main(a):
    df = pd.read_csv(os.path.join(a.labels_dir, "labels_index.csv"))
    unique_values = set()

    print(f"Checking {len(df)} label files in {a.labels_dir}...\n")

    for i, row in df.iterrows():
        label_path = row["label_npy"]
        if not os.path.exists(label_path):
            print(f"⚠️ Missing: {label_path}")
            continue
        
        y = np.load(label_path)
        vals, counts = np.unique(y, return_counts=True)
        unique_values.update(vals)

        # Show first few label distributions
        if i < 5:
            print(f"{row['chip_id']} → {dict(zip(vals.tolist(), counts.tolist()))}")

    print("\n✅ Overall unique class values found:", unique_values)
    if unique_values == {0}:
        print("⚠️ WARNING: Only background found! Labels are not useful.")
    elif unique_values <= {0, 3}:
        print("⚠️ WARNING: Only background+change found. No buildings/roads.")
    else:
        print("🎯 Good: Multiple classes detected.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", default="data/labels/multiclass_4class")
    args = ap.parse_args()
    main(args)
