"""
split_dataset.py
================
Splits features/<class>/rgb/ filenames into train/val/test index files.
NO file copying — just writes index.txt per split per class.

Run after precompute_maps.py:
    python split_dataset.py
"""

import os
import random
from config2 import Config

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".bmp")


def split_class(class_name: str):
    rgb_dir = os.path.join(Config.FEATURES_ROOT, class_name, "rgb")
    if not os.path.isdir(rgb_dir):
        print(f"  [SKIP] {rgb_dir} not found — run precompute_maps.py first")
        return {}

    files = sorted([f for f in os.listdir(rgb_dir)
                    if f.lower().endswith(IMG_EXTS)])
    if not files:
        print(f"  [SKIP] No images in {rgb_dir}")
        return {}

    random.shuffle(files)
    n     = len(files)
    n_tr  = int(n * Config.TRAIN_RATIO)
    n_val = int(n * Config.VAL_RATIO)

    splits = {
        "train": files[:n_tr],
        "val":   files[n_tr: n_tr + n_val],
        "test":  files[n_tr + n_val:],
    }

    for split_name, flist in splits.items():
        out_dir = os.path.join(Config.DATASET_ROOT, split_name, class_name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "index.txt"), "w") as f:
            f.write("\n".join(flist))

    return {k: len(v) for k, v in splits.items()}


def main():
    random.seed(Config.SEED)
    print(f"Features : {os.path.abspath(Config.FEATURES_ROOT)}")
    print(f"Output   : {os.path.abspath(Config.DATASET_ROOT)}")
    print(f"Split    : train={Config.TRAIN_RATIO}  "
          f"val={Config.VAL_RATIO}  test={Config.TEST_RATIO}  "
          f"seed={Config.SEED}\n")

    print(f"  {'Class':15s}  {'train':>7}  {'val':>7}  {'test':>7}  {'total':>7}")
    print("  " + "-" * 50)
    grand = {"train": 0, "val": 0, "test": 0}

    for cls in Config.CLASS_NAMES:
        counts = split_class(cls)
        if counts:
            total = sum(counts.values())
            print(f"  {cls:15s}  {counts['train']:7d}  "
                  f"{counts['val']:7d}  {counts['test']:7d}  {total:7d}")
            for k in grand:
                grand[k] += counts.get(k, 0)

    print("  " + "-" * 50)
    g_total = sum(grand.values())
    print(f"  {'TOTAL':15s}  {grand['train']:7d}  "
          f"{grand['val']:7d}  {grand['test']:7d}  {g_total:7d}")
    print("\nDone. Next → python train_classifier.py  OR  python train_localizer.py")


if __name__ == "__main__":
    main()
