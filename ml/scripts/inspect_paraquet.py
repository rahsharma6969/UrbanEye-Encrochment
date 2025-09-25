# import pandas as pd
# import numpy as np
# from pathlib import Path
# import os

# # --- CONFIGURE THIS PATH TO MATCH YOUR INDEX FILE ---
# INDEX_PATH = r"C:\Users\Rahul\CollegeProject\UrbanEyeML\ml\data\LEVIR_CD\chips_256\index.parquet"

# print("🔍 INSPECTING INDEX.PARQUET")
# print("="*60)

# # Load index
# df = pd.read_parquet(INDEX_PATH)
# print(f"📁 File: {INDEX_PATH}")
# print(f"📊 Total rows (chips): {len(df):,}")

# # Check required columns
# required_cols = ['t0_npy', 't1_npy', 'mask_npy', 'split']
# missing = [col for col in required_cols if col not in df.columns]
# if missing:
#     print(f"\n❌ CRITICAL ERROR: Missing columns: {missing}")
# else:
#     print(f"✅ Required columns present: {list(df.columns)}")

# # Check for duplicates
# duplicates = df.duplicated(subset=['t0_npy', 't1_npy', 'mask_npy']).sum()
# if duplicates > 0:
#     print(f"\n⚠️ WARNING: {duplicates} duplicate chip entries found!")
# else:
#     print(f"✅ No duplicate chip entries")

# # Split distribution
# print("\n--- SPLIT DISTRIBUTION ---")
# split_counts = df['split'].value_counts()
# for split, count in split_counts.items():
#     pct = count / len(df) * 100
#     print(f"  {split:>5}: {count:>6} chips ({pct:>5.1f}%)")

# # Validate file paths exist
# print("\n--- FILE PATH VALIDATION ---")
# missing_files = []
# for idx, row in df.iterrows():
#     for col in ['t0_npy', 't1_npy', 'mask_npy']:
#         path = Path(row[col])
#         if not path.exists():
#             missing_files.append((col, row[col]))

# if missing_files:
#     print(f"\n❌ {len(missing_files)} files are missing:")
#     for col, path in missing_files[:5]:  # Show first 5
#         print(f"   - {col}: {path}")
#     if len(missing_files) > 5:
#         print(f"   ... and {len(missing_files)-5} more")
# else:
#     print(f"✅ All .npy files exist on disk")

# # Sample rows
# print("\n--- SAMPLE ROWS ---")
# sample = df.head(3).copy()
# for _, row in sample.iterrows():
#     print(f"\n📄 {Path(row['mask_npy']).name}")
#     print(f"   T0: {Path(row['t0_npy']).name}")
#     print(f"   T1: {Path(row['t1_npy']).name}")
#     print(f"   Split: {row['split']}")

# # Analyze mask content (first 10 chips)
# print("\n--- MASK CONTENT ANALYSIS (first 10 chips) ---")
# mask_stats = []
# for idx, row in df.head(10).iterrows():
#     mask = np.load(row['mask_npy'])
#     if mask.ndim == 3:
#         mask = mask[0]  # handle CHW
#     unique_vals = np.unique(mask)
#     n_change = (mask == 1).sum()
#     n_bg = (mask == 0).sum()
#     n_ignore = (mask == 255).sum()

#     mask_stats.append({
#         'file': Path(row['mask_npy']).name,
#         'shape': mask.shape,
#         'unique': sorted(unique_vals),
#         'change_pixels': n_change,
#         'bg_pixels': n_bg,
#         'ignore_pixels': n_ignore,
#         'total': mask.size
#     })

# for stat in mask_stats:
#     print(f"  {stat['file']:<20} | change={stat['change_pixels']:>4} | bg={stat['bg_pixels']:>6} | ignore={stat['ignore_pixels']:>4} | unique={stat['unique']}")

# # Overall stats across all masks
# print("\n--- OVERALL DATASET STATISTICS ---")
# all_masks = []
# for _, row in df.iterrows():
#     mask = np.load(row['mask_npy'])
#     if mask.ndim == 3:
#         mask = mask[0]
#     all_masks.append(mask)

# all_masks = np.concatenate([m.ravel() for m in all_masks])
# unique_vals = np.unique(all_masks)
# total_pixels = len(all_masks)
# change_pixels = (all_masks == 1).sum()
# bg_pixels = (all_masks == 0).sum()
# ignore_pixels = (all_masks == 255).sum()

# print(f"Total pixels analyzed: {total_pixels:,}")
# print(f"Change pixels (1):     {change_pixels:,} ({change_pixels/total_pixels*100:.4f}%)")
# print(f"Background (0):        {bg_pixels:,} ({bg_pixels/total_pixels*100:.4f}%)")
# print(f"Ignore (255):          {ignore_pixels:,} ({ignore_pixels/total_pixels*100:.4f}%)")

# if change_pixels == 0:
#     print("\n🚨 CRITICAL: NO CHANGE PIXELS FOUND IN ENTIRE DATASET!")
# elif change_pixels < total_pixels * 0.001:
#     print("\n⚠️ Warning: Very few change pixels — consider augmenting or checking labels.")
# else:
#     print("\n✅ Healthy change pixel ratio.")

# print("\n" + "="*60)
# print("✅ INSPECTION COMPLETE")

import pandas as pd
import numpy as np
from pathlib import Path
import os

# --- CONFIGURE THIS PATH TO MATCH YOUR INDEX FILE ---
INDEX_PATH = Path(r"C:\Users\Rahul\CollegeProject\UrbanEyeML\ml\data\LEVIR_CD\chips_256\index.parquet")

print("🔍 INSPECTING INDEX.PARQUET")
print("="*60)

# Load index
df = pd.read_parquet(INDEX_PATH)
print(f"📁 File: {INDEX_PATH}")
print(f"📊 Total rows (chips): {len(df):,}")

# Check required columns
required_cols = ['t0_npy', 't1_npy', 'mask_npy', 'split']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    print(f"\n❌ CRITICAL ERROR: Missing columns: {missing}")
else:
    print(f"✅ Required columns present: {list(df.columns)}")

# Check for duplicates
duplicates = df.duplicated(subset=['t0_npy', 't1_npy', 'mask_npy']).sum()
if duplicates > 0:
    print(f"\n⚠️ WARNING: {duplicates} duplicate chip entries found!")
else:
    print(f"✅ No duplicate chip entries")

# Split distribution
print("\n--- SPLIT DISTRIBUTION ---")
split_counts = df['split'].value_counts()
for split, count in split_counts.items():
    pct = count / len(df) * 100
    print(f"  {split:>5}: {count:>6} chips ({pct:>5.1f}%)")

# Validate file paths exist — resolve relative to index.parquet's parent
base_dir = INDEX_PATH.parent  # e.g., data\LEVIR_CD\chips_256
print(f"\n--- FILE PATH VALIDATION ---")
print(f"Base directory: {base_dir}")

missing_files = []
for idx, row in df.iterrows():
    for col in ['t0_npy', 't1_npy', 'mask_npy']:
        # Resolve path relative to base_dir
        path = (base_dir / row[col]).resolve()
        if not path.exists():
            missing_files.append((col, row[col]))

if missing_files:
    print(f"❌ {len(missing_files)} files are missing:")
    for col, path in missing_files[:5]:  # Show first 5
        print(f"   - {col}: {path}")
    if len(missing_files) > 5:
        print(f"   ... and {len(missing_files)-5} more")
else:
    print(f"✅ All .npy files exist on disk")

# Sample rows
print("\n--- SAMPLE ROWS ---")
sample = df.head(3).copy()
for _, row in sample.iterrows():
    print(f"\n📄 {Path(row['mask_npy']).name}")
    print(f"   T0: {Path(row['t0_npy']).name}")
    print(f"   T1: {Path(row['t1_npy']).name}")
    print(f"   Split: {row['split']}")

# 🎯 CRITICAL: DETAILED CLASS/PIXEL VALUE ANALYSIS
print("\n" + "="*60)
print("🎯 CRITICAL: PIXEL CLASS VALUE INSPECTION")
print("="*60)

# First, check a few masks in detail
print("\n--- DETAILED MASK ANALYSIS (first 10 chips) ---")
mask_stats = []
has_change_pixels = False

for idx, row in df.head(10).iterrows():
    try:
        mask_path = (base_dir / row['mask_npy']).resolve()
        if not mask_path.exists():
            print(f"⚠️ Skipping missing mask: {mask_path}")
            continue
            
        mask = np.load(mask_path)
        original_shape = mask.shape
        
        # Handle different mask formats
        if mask.ndim == 3:
            print(f"📐 3D mask detected: {mask.shape} -> squeezing to 2D")
            mask = mask.squeeze()
        
        unique_vals = np.unique(mask)
        value_counts = {val: (mask == val).sum() for val in unique_vals}
        
        # Count specific classes
        n_change = (mask == 1).sum()
        n_bg = (mask == 0).sum()
        n_ignore = (mask == 255).sum()
        
        if n_change > 0:
            has_change_pixels = True
        
        print(f"\n📄 {Path(row['mask_npy']).name}")
        print(f"   📐 Shape: {original_shape} -> {mask.shape}")
        print(f"   🎨 Unique values: {unique_vals}")
        print(f"   📊 Value counts: {value_counts}")
        print(f"   🔴 Change pixels (1): {n_change:,} ({n_change/mask.size*100:.2f}%)")
        print(f"   ⚫ Background (0): {n_bg:,} ({n_bg/mask.size*100:.2f}%)")
        print(f"   ⚪ Ignore (255): {n_ignore:,} ({n_ignore/mask.size*100:.2f}%)")
        print(f"   ✅ Has change pixels: {'YES' if n_change > 0 else 'NO'}")

        mask_stats.append({
            'file': Path(row['mask_npy']).name,
            'shape': mask.shape,
            'unique': sorted(unique_vals),
            'change_pixels': n_change,
            'bg_pixels': n_bg,
            'ignore_pixels': n_ignore,
            'total': mask.size,
            'change_percent': n_change/mask.size*100
        })
    except Exception as e:
        print(f"❌ Error loading {row['mask_npy']}: {e}")
        continue

print(f"\n--- SUMMARY OF FIRST 10 CHIPS ---")
print(f"{'File':<25} | {'Change':<6} | {'BG':<8} | {'Ignore':<6} | {'Change%':<8} | {'Classes'}")
print("-" * 80)
for stat in mask_stats:
    change_pct = f"{stat['change_percent']:.2f}%"
    unique_str = str(stat['unique'])
    print(f"{stat['file']:<25} | {stat['change_pixels']:<6} | {stat['bg_pixels']:<8} | {stat['ignore_pixels']:<6} | {change_pct:<8} | {unique_str}")

if not has_change_pixels:
    print("\n🚨 CRITICAL ISSUE: NO CHANGE PIXELS (class=1) FOUND IN FIRST 10 CHIPS!")
    print("   This means your preprocessing is not converting masks correctly.")
else:
    print(f"\n✅ SUCCESS: Found change pixels in the analyzed chips!")

# Check class distribution across more samples
print("\n--- EXTENDED CLASS DISTRIBUTION ANALYSIS ---")
sample_size = min(50, len(df))
print(f"Analyzing {sample_size} random chips for class distribution...")

# Sample random chips for broader analysis
random_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

total_change = 0
total_bg = 0
total_ignore = 0
total_pixels = 0
chips_with_changes = 0

for idx, row in random_sample.iterrows():
    try:
        mask_path = (base_dir / row['mask_npy']).resolve()
        if not mask_path.exists():
            continue
            
        mask = np.load(mask_path)
        if mask.ndim == 3:
            mask = mask.squeeze()
        
        n_change = (mask == 1).sum()
        n_bg = (mask == 0).sum()
        n_ignore = (mask == 255).sum()
        
        total_change += n_change
        total_bg += n_bg
        total_ignore += n_ignore
        total_pixels += mask.size
        
        if n_change > 0:
            chips_with_changes += 1
            
    except Exception as e:
        continue

print(f"\n📊 EXTENDED ANALYSIS RESULTS ({sample_size} chips):")
print(f"   🔴 Total change pixels (1): {total_change:,} ({total_change/total_pixels*100:.4f}%)")
print(f"   ⚫ Total background (0): {total_bg:,} ({total_bg/total_pixels*100:.4f}%)")
print(f"   ⚪ Total ignore (255): {total_ignore:,} ({total_ignore/total_pixels*100:.4f}%)")
print(f"   📈 Chips with changes: {chips_with_changes}/{sample_size} ({chips_with_changes/sample_size*100:.1f}%)")

if total_change == 0:
    print(f"\n🚨🚨 CRITICAL DATASET ISSUE 🚨🚨")
    print(f"   NO CHANGE PIXELS FOUND ACROSS {sample_size} CHIPS!")
    print(f"   Your mask preprocessing is NOT working correctly.")
    print(f"   Expected: masks should have pixels with value 1 for changes")
    print(f"   Found: only values 0 and/or 255")
elif total_change < total_pixels * 0.001:
    print(f"\n⚠️ WARNING: Very few change pixels ({total_change/total_pixels*100:.4f}%)")
    print(f"   This might indicate issues with data preprocessing or labeling.")
else:
    print(f"\n✅ HEALTHY DATASET: Found {total_change/total_pixels*100:.4f}% change pixels")

# Overall stats across all masks (sample to avoid memory issues)
print("\n" + "="*60)
print("📈 FINAL DATASET ASSESSMENT")
print("="*60)

sample_size = min(100, len(df))
print(f"Analyzing first {sample_size} masks for final assessment...")

all_masks = []
dataset_has_changes = False

for idx, row in df.head(sample_size).iterrows():
    try:
        mask_path = (base_dir / row['mask_npy']).resolve()
        if not mask_path.exists():
            continue
            
        mask = np.load(mask_path)
        if mask.ndim == 3:
            mask = mask.squeeze()
        
        # Check if this mask has any change pixels
        if (mask == 1).sum() > 0:
            dataset_has_changes = True
            
        all_masks.append(mask)
    except Exception as e:
        print(f"⚠️ Error loading mask {idx}: {e}")
        continue

if all_masks:
    all_masks = np.concatenate([m.ravel() for m in all_masks])
    unique_vals = np.unique(all_masks)
    total_pixels = len(all_masks)
    change_pixels = (all_masks == 1).sum()
    bg_pixels = (all_masks == 0).sum()
    ignore_pixels = (all_masks == 255).sum()

    print(f"\n📊 FINAL STATISTICS:")
    print(f"   Total pixels analyzed: {total_pixels:,}")
    print(f"   🎨 Unique pixel values found: {unique_vals}")
    print(f"   🔴 Change pixels (1):     {change_pixels:,} ({change_pixels/total_pixels*100:.4f}%)")
    print(f"   ⚫ Background (0):        {bg_pixels:,} ({bg_pixels/total_pixels*100:.4f}%)")
    print(f"   ⚪ Ignore (255):          {ignore_pixels:,} ({ignore_pixels/total_pixels*100:.4f}%)")

    print(f"\n🎯 DATASET STATUS:")
    if change_pixels == 0:
        print("   🚨🚨 CRITICAL FAILURE 🚨🚨")
        print("   ❌ NO CHANGE PIXELS (class=1) FOUND IN ENTIRE DATASET!")
        print("   ❌ Your model will NOT learn change detection!")
        print("   ❌ Fix your mask preprocessing immediately!")
        print(f"   💡 Expected: Masks should have value 1 for change pixels")
        print(f"   💡 Found: Only values {unique_vals}")
    elif change_pixels < total_pixels * 0.001:
        print("   ⚠️ WARNING: Very few change pixels detected")
        print(f"   ⚠️ Only {change_pixels/total_pixels*100:.4f}% change pixels")
        print("   ⚠️ This might cause training issues")
    else:
        print("   ✅ SUCCESS: Dataset has healthy change pixel ratio")
        print(f"   ✅ {change_pixels/total_pixels*100:.4f}% change pixels detected")
        print("   ✅ Model should be able to learn change detection")

    # Provide specific guidance
    print(f"\n💡 NEXT STEPS:")
    if change_pixels == 0:
        print("   1. 🔧 Fix your load_mask() function in preprocessing")
        print("   2. 🔧 Ensure masks convert 255→1 (not 255→0)")  
        print("   3. 🔧 Re-run chip generation with fixed preprocessing")
        print("   4. 🔧 Re-inspect this dataset")
    elif dataset_has_changes:
        print("   1. ✅ Dataset looks good for training!")
        print("   2. ✅ Start/resume your change detection training")
        print("   3. ✅ Monitor training metrics for improvement")
    
else:
    print("❌ No valid mask files found to analyze!")
    print("💡 Check if your chip generation completed successfully")

print("\n" + "="*60)
print("✅ INSPECTION COMPLETE")