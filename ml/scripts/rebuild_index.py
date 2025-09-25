import pandas as pd
from pathlib import Path
import re

# Path to your chip directory
CHIPS_DIR = Path(r"C:\Users\Rahul\CollegeProject\UrbanEyeML\ml\data\chips_256\npy")

# List all .npy files
npy_files = list(CHIPS_DIR.glob("*.npy"))

# Dictionary to store rows
rows = []

# Regex to extract base name (e.g., "2_0_0" from "2_0_0_A.npy")
pattern = r"^(.+?)_[AB]OUT?$"

for f in npy_files:
    name = f.name
    # Skip if not matching pattern
    match = re.match(pattern, name)
    if not match:
        continue

    base_name = match.group(1)  # e.g., "2_0_0"
    
    # Build full paths
    t0_path = CHIPS_DIR / f"{base_name}_A.npy"
    t1_path = CHIPS_DIR / f"{base_name}_B.npy"
    mask_path = CHIPS_DIR / f"{base_name}_OUT.npy"

    # Only include if ALL three files exist
    if t0_path.exists() and t1_path.exists() and mask_path.exists():
        # Assign split: first 10% = val, next 10% = test, rest = train
        # Use hash of filename for reproducible split
        hash_val = hash(base_name) % 100
        if hash_val < 10:
            split = "val"
        elif hash_val < 20:
            split = "test"
        else:
            split = "train"

        rows.append({
            "t0_npy": str(t0_path),
            "t1_npy": str(t1_path),
            "mask_npy": str(mask_path),
            "split": split
        })

# Create DataFrame
df = pd.DataFrame(rows)

# Save as parquet
output_path = CHIPS_DIR / "index.parquet"
df.to_parquet(output_path, index=False)

print(f"✅ Successfully regenerated index.parquet")
print(f"Total chips: {len(df)}")
print(f"Train: {len(df[df.split == 'train'])}")
print(f"Val: {len(df[df.split == 'val'])}")
print(f"Test: {len(df[df.split == 'test'])}")
print(f"Saved to: {output_path}")