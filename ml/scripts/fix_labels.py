import os

def fix_label_filenames(labels_dir="data/labels/multiclass"):
    renamed = []
    skipped = []
    for f in os.listdir(labels_dir):
        if f.endswith("_labels.npy"):  # wrong naming
            old_path = os.path.join(labels_dir, f)
            new_name = f.replace("_labels.npy", "_label.npy")
            new_path = os.path.join(labels_dir, new_name)

            os.rename(old_path, new_path)
            renamed.append((f, new_name))
        else:
            skipped.append(f)

    print(f"✅ Renamed {len(renamed)} files")
    for old, new in renamed:
        print(f"  {old} → {new}")
    if skipped:
        print(f"ℹ️ Skipped {len(skipped)} files (already correct): {skipped[:5]}...")

if __name__ == "__main__":
    fix_label_filenames()
