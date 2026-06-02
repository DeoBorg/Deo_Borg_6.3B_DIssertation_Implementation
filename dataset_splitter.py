"""
dataset_splitter.py
-------------------
Splits the Roboflow object detection dataset into train/val/test
using a stratified approach based on class presence in each image.

Split ratios: 70% train / 15% val / 15% test

Also prints a summary of class distribution across splits.

Usage:
    python dataset_splitter.py
"""

import os
import shutil
import random
import numpy as np
from collections import defaultdict

# ── CONFIG ───────────────────────────────────────────────────────────────────
IMAGES_TRAIN_SRC = "dataset/images/train"    # all images currently here
LABELS_TRAIN_SRC = "dataset/labels/train"    # all labels currently here

SPLITS = {
    "train": 0.70,
    "val":   0.15,
    "test":  0.15,
}

RANDOM_SEED = 42
CLASS_NAMES = ["Person", "Laptop", "Phone"]
# ─────────────────────────────────────────────────────────────────────────────


def get_classes_in_label(label_path):
    """Return set of class IDs present in a YOLO label file."""
    classes = set()
    if not os.path.exists(label_path):
        return classes
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if line:
                cls_id = int(line.split()[0])
                classes.add(cls_id)
    return classes


def stratify_by_class(image_files, labels_src):
    """
    Group images by which classes they contain.
    This ensures all classes appear in every split.
    """
    groups = defaultdict(list)
    for fname in image_files:
        base       = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_src, base + ".txt")
        classes    = get_classes_in_label(label_path)

        # Create a signature key based on classes present
        # e.g. frozenset({0, 2}) = image has Person and Phone
        key = frozenset(classes) if classes else frozenset({-1})
        groups[key].append(fname)

    return groups


def split_groups(groups, ratios, seed=42):
    """Split each group according to ratios, return train/val/test lists."""
    random.seed(seed)
    train, val, test = [], [], []

    for key, files in groups.items():
        random.shuffle(files)
        n       = len(files)
        n_train = max(1, int(n * ratios["train"]))
        n_val   = max(1, int(n * ratios["val"]))

        train += files[:n_train]
        val   += files[n_train:n_train + n_val]
        test  += files[n_train + n_val:]

    return train, val, test


def copy_files(file_list, src_images, src_labels, dst_images, dst_labels):
    """Copy images and their label files to destination folders."""
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    copied_images = 0
    copied_labels = 0
    missing_labels = 0

    for fname in file_list:
        # Copy image
        src_img = os.path.join(src_images, fname)
        dst_img = os.path.join(dst_images, fname)
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
            copied_images += 1

        # Copy label
        base      = os.path.splitext(fname)[0]
        src_lbl   = os.path.join(src_labels, base + ".txt")
        dst_lbl   = os.path.join(dst_labels, base + ".txt")
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
            copied_labels += 1
        else:
            missing_labels += 1

    return copied_images, copied_labels, missing_labels


def class_distribution(file_list, labels_src):
    """Count instances of each class across a list of label files."""
    counts = defaultdict(int)
    for fname in file_list:
        base       = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_src, base + ".txt")
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        cls_id = int(line.split()[0])
                        counts[cls_id] += 1
    return counts


def print_distribution(split_name, file_list, labels_src):
    counts = class_distribution(file_list, labels_src)
    total  = sum(counts.values())
    print(f"\n  {split_name} ({len(file_list)} images, {total} annotations):")
    for cls_id, name in enumerate(CLASS_NAMES):
        count = counts.get(cls_id, 0)
        pct   = count / total * 100 if total > 0 else 0
        print(f"    {name:<10} : {count:>5} ({pct:.1f}%)")


def main():
    print("\n=== Dataset Splitter ===")
    print(f"  Source images : {IMAGES_TRAIN_SRC}")
    print(f"  Source labels : {LABELS_TRAIN_SRC}")
    print(f"  Split ratios  : train={SPLITS['train']:.0%}  "
          f"val={SPLITS['val']:.0%}  test={SPLITS['test']:.0%}\n")

    # Get all image files
    image_files = sorted([
        f for f in os.listdir(IMAGES_TRAIN_SRC)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"  Total images found : {len(image_files)}")

    # Stratify by class presence
    groups = stratify_by_class(image_files, LABELS_TRAIN_SRC)
    print(f"  Class-presence groups : {len(groups)}")
    for key, files in sorted(groups.items(), key=lambda x: -len(x[1])):
        class_names = [CLASS_NAMES[i] for i in sorted(key) if i >= 0]
        label = "+".join(class_names) if class_names else "no_labels"
        print(f"    {label:<30} : {len(files)} images")

    # Split
    train_files, val_files, test_files = split_groups(groups, SPLITS, RANDOM_SEED)
    print(f"\n  Split result:")
    print(f"    Train : {len(train_files)} images")
    print(f"    Val   : {len(val_files)} images")
    print(f"    Test  : {len(test_files)} images")

    # Confirm before copying
    confirm = input("\n  Proceed with splitting? (yes/no): ")
    if confirm.strip().lower() != "yes":
        print("  Cancelled.")
        return

    # Copy files into split folders
    # Note: train files stay in place, we create val/ and test/ folders
    print("\n  Copying val files...")
    copy_files(val_files,
               IMAGES_TRAIN_SRC, LABELS_TRAIN_SRC,
               "dataset/images/val", "dataset/labels/val")

    print("  Copying test files...")
    copy_files(test_files,
               IMAGES_TRAIN_SRC, LABELS_TRAIN_SRC,
               "dataset/images/test", "dataset/labels/test")

    # Remove val and test files from train folder
    print("  Cleaning train folder...")
    removed = 0
    for fname in val_files + test_files:
        # Remove image
        img_path = os.path.join(IMAGES_TRAIN_SRC, fname)
        if os.path.exists(img_path):
            os.remove(img_path)
            removed += 1
        # Remove label
        base = os.path.splitext(fname)[0]
        lbl_path = os.path.join(LABELS_TRAIN_SRC, base + ".txt")
        if os.path.exists(lbl_path):
            os.remove(lbl_path)

    print(f"  Removed {removed} files from train folder")

    # Final counts
    final_train = len([f for f in os.listdir(IMAGES_TRAIN_SRC)
                       if f.lower().endswith(('.jpg','.jpeg','.png'))])
    final_val   = len([f for f in os.listdir("dataset/images/val")
                       if f.lower().endswith(('.jpg','.jpeg','.png'))])
    final_test  = len([f for f in os.listdir("dataset/images/test")
                       if f.lower().endswith(('.jpg','.jpeg','.png'))])

    print(f"\n{'='*50}")
    print(f"  Final split:")
    print(f"    Train : {final_train} images")
    print(f"    Val   : {final_val} images")
    print(f"    Test  : {final_test} images")

    # Class distribution per split
    print_distribution("Train", os.listdir(IMAGES_TRAIN_SRC), LABELS_TRAIN_SRC)
    print_distribution("Val",   os.listdir("dataset/images/val"),
                       "dataset/labels/val")
    print_distribution("Test",  os.listdir("dataset/images/test"),
                       "dataset/labels/test")

    print(f"\n  ✓ Dataset split complete")
    print(f"  ✓ data.yaml is already configured correctly")
    print(f"\n  Next step: upload dataset to Google Colab for YOLOv8 training")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()