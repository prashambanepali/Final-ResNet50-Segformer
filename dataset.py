"""
dataset.py
==========
Shared dataset loader for both:
  - Classifier  (ResNet50)  → returns (9ch_tensor, label)
  - Localizer   (SegFormer) → returns (9ch_tensor, mask, label)

Channel layout (9 total):
  0-2  RGB
  3-5  ELA  (R=q70 | G=q80 | B=q90)
  6-8  Noise (R=Gaussian | G=SRM-linear | B=SRM-edge)

Mask naming:  <image_base>_GT.png
"""

import os
import random
import io
import math
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np

from config2 import Config

# ── Normalisation ─────────────────────────────────────────────────────────
# ImageNet stats replicated across all 3 modalities
_M = [0.485, 0.456, 0.406] * 3
_S = [0.229, 0.224, 0.225] * 3
MEAN_T = torch.tensor(_M).view(9, 1, 1)
STD_T  = torch.tensor(_S).view(9, 1, 1)

def normalize_9ch(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN_T) / STD_T

def denormalize_9ch(x: torch.Tensor) -> torch.Tensor:
    return x * STD_T + MEAN_T


# ── Helpers ───────────────────────────────────────────────────────────────
def _jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def _add_noise(t: torch.Tensor, std=0.02) -> torch.Tensor:
    return torch.clamp(t + torch.randn_like(t) * std, 0.0, 1.0)


# ── Paired augmentation ───────────────────────────────────────────────────
class PairedAugment:
    """
    Applies identical spatial transforms to rgb, ela, noise
    and (optionally) a segmentation mask.
    """
    def __init__(self, train: bool = True):
        self.train = train

    def __call__(
        self,
        rgb:   Image.Image,
        ela:   Image.Image,
        noise: Image.Image,
        mask:  Optional[Image.Image] = None,
        out_size: tuple = (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
    ):
        def _spatial(imgs, interp):
            return [TF.resize(i, out_size, interp) for i in imgs]

        spatial  = [rgb, ela, noise] + ([mask] if mask else [])
        n_img    = 3   # rgb, ela, noise always present
        mask_idx = 3   # mask is at index 3 if present

        if not self.train:
            out = [TF.resize(x, out_size,
                   InterpolationMode.NEAREST if (i == mask_idx and mask)
                   else InterpolationMode.BILINEAR)
                   for i, x in enumerate(spatial)]
            return tuple(out) if mask else tuple(out)

        # ── Resize ────────────────────────────────────────────────────
        rgb   = TF.resize(rgb,   out_size, InterpolationMode.BILINEAR)
        ela   = TF.resize(ela,   out_size, InterpolationMode.BILINEAR)
        noise = TF.resize(noise, out_size, InterpolationMode.BILINEAR)
        if mask:
            mask = TF.resize(mask, out_size, InterpolationMode.NEAREST)

        # ── H-flip ────────────────────────────────────────────────────
        if random.random() < 0.5:
            rgb   = TF.hflip(rgb)
            ela   = TF.hflip(ela)
            noise = TF.hflip(noise)
            if mask: mask = TF.hflip(mask)

        # ── V-flip ────────────────────────────────────────────────────
        if random.random() < 0.2:
            rgb   = TF.vflip(rgb)
            ela   = TF.vflip(ela)
            noise = TF.vflip(noise)
            if mask: mask = TF.vflip(mask)

        # ── 90° rotation ──────────────────────────────────────────────
        if random.random() < 0.4:
            k = random.choice([1, 2, 3])
            rgb   = rgb.rotate(k * 90)
            ela   = ela.rotate(k * 90)
            noise = noise.rotate(k * 90)
            if mask: mask = mask.rotate(k * 90)

        # ── Small rotation ────────────────────────────────────────────
        angle = random.uniform(-15, 15)
        if abs(angle) > 0.5:
            rgb   = TF.rotate(rgb,   angle, InterpolationMode.BILINEAR, fill=0)
            ela   = TF.rotate(ela,   angle, InterpolationMode.BILINEAR, fill=0)
            noise = TF.rotate(noise, angle, InterpolationMode.BILINEAR, fill=0)
            if mask:
                mask = TF.rotate(mask, angle, InterpolationMode.NEAREST, fill=0)

        # ── RGB-only: JPEG ────────────────────────────────────────────
        if random.random() < 0.6:
            rgb = _jpeg(rgb, random.randint(55, 98))

        # ── RGB-only: color jitter ────────────────────────────────────
        if random.random() < 0.7:
            rgb = TF.adjust_brightness(rgb, random.uniform(0.8, 1.2))
            rgb = TF.adjust_contrast(rgb,   random.uniform(0.8, 1.2))
            rgb = TF.adjust_saturation(rgb, random.uniform(0.85, 1.15))
            rgb = TF.adjust_hue(rgb,        random.uniform(-0.04, 0.04))

        # ── RGB-only: blur ────────────────────────────────────────────
        if random.random() < 0.25:
            rgb = rgb.filter(ImageFilter.GaussianBlur(
                radius=random.uniform(0.3, 1.5)))

        # ── RGB-only: sharpen ─────────────────────────────────────────
        if random.random() < 0.2:
            rgb = ImageEnhance.Sharpness(rgb).enhance(random.uniform(1.2, 2.5))

        if mask:
            return rgb, ela, noise, mask
        return rgb, ela, noise


# ─────────────────────────────────────────────────────────────────────────
# Base 9-channel loader (shared between classifier and localizer)
# ─────────────────────────────────────────────────────────────────────────
def _load_9ch(rgb_path, ela_path, noise_path, augmentor, mask_img=None):
    """Load 3 modalities, augment, concatenate → (9, H, W) tensor."""
    rgb   = Image.open(rgb_path).convert("RGB")
    ela   = (Image.open(ela_path).convert("RGB")
             if os.path.exists(ela_path)
             else Image.new("RGB", rgb.size, 0))
    noise = (Image.open(noise_path).convert("RGB")
             if os.path.exists(noise_path)
             else Image.new("RGB", rgb.size, 0))

    if mask_img is not None:
        rgb, ela, noise, mask_img = augmentor(rgb, ela, noise, mask_img)
    else:
        rgb, ela, noise = augmentor(rgb, ela, noise)

    rgb_t   = TF.to_tensor(rgb)
    ela_t   = TF.to_tensor(ela)
    noise_t = TF.to_tensor(noise)

    # Gaussian noise on RGB only during training
    if augmentor.train and random.random() < 0.4:
        rgb_t = _add_noise(rgb_t)

    x = normalize_9ch(torch.cat([rgb_t, ela_t, noise_t], dim=0))  # (9,H,W)
    return x, mask_img


# ─────────────────────────────────────────────────────────────────────────
# 1.  Classifier Dataset  → (9ch, label)
# ─────────────────────────────────────────────────────────────────────────
class ClassifierDataset(Dataset):
    """Returns (x: 9ch tensor, label: int) for ResNet50 training."""

    def __init__(self, split: str, augment: bool = True):
        self.augmentor = PairedAugment(train=augment)
        # items: (rgb_path, ela_path, noise_path, label)
        self.items: List[Tuple] = []

        for cls in Config.CLASS_NAMES:
            label      = Config.CLASS_NAMES.index(cls)
            index_path = os.path.join(Config.DATASET_ROOT, split, cls, "index.txt")
            if not os.path.exists(index_path):
                print(f"  [WARNING] {index_path} not found — skipping {cls}")
                continue

            with open(index_path) as f:
                fnames = [l.strip() for l in f if l.strip()]

            rgb_dir   = os.path.join(Config.FEATURES_ROOT, cls, "rgb")
            ela_dir   = os.path.join(Config.FEATURES_ROOT, cls, "ela")
            noise_dir = os.path.join(Config.FEATURES_ROOT, cls, "noise")

            for fname in fnames:
                base = os.path.splitext(fname)[0]
                self.items.append((
                    os.path.join(rgb_dir,   fname),
                    os.path.join(ela_dir,   base + ".png"),
                    os.path.join(noise_dir, base + ".png"),
                    label,
                ))

        from collections import Counter
        counts = Counter(item[3] for item in self.items)
        print(f"[ClassifierDataset:{split}]  total={len(self.items)}")
        for i, name in enumerate(Config.CLASS_NAMES):
            print(f"  {name:15s}: {counts.get(i, 0)}")

    def __len__(self):  return len(self.items)

    def __getitem__(self, idx):
        rgb_p, ela_p, noise_p, label = self.items[idx]
        x, _ = _load_9ch(rgb_p, ela_p, noise_p, self.augmentor)
        return x, label


# ─────────────────────────────────────────────────────────────────────────
# 2.  Localizer Dataset  → (9ch, mask, label)
# ─────────────────────────────────────────────────────────────────────────
class LocalizerDataset(Dataset):
    """
    Returns (x: 9ch tensor, mask: (1,H,W) binary tensor, label: int).
    Authentic images get an all-zero mask automatically.
    Mask filename = <image_base>_GT.png  inside raw/<class>/masks/
    """

    def __init__(self, split: str, augment: bool = True,
                 include_authentic: bool = True):
        self.augmentor = PairedAugment(train=augment)
        self.items: List[Tuple] = []   # (rgb_p, ela_p, noise_p, mask_p_or_None, label)

        for cls in Config.CLASS_NAMES:
            label      = Config.CLASS_NAMES.index(cls)
            index_path = os.path.join(Config.DATASET_ROOT, split, cls, "index.txt")
            if not os.path.exists(index_path):
                print(f"  [WARNING] {index_path} not found — skipping {cls}")
                continue

            with open(index_path) as f:
                fnames = [l.strip() for l in f if l.strip()]

            rgb_dir   = os.path.join(Config.FEATURES_ROOT, cls, "rgb")
            ela_dir   = os.path.join(Config.FEATURES_ROOT, cls, "ela")
            noise_dir = os.path.join(Config.FEATURES_ROOT, cls, "noise")
            mask_dir  = os.path.join(Config.RAW_ROOT, cls, "masks")

            is_authentic = (cls == "authentic")

            for fname in fnames:
                base = os.path.splitext(fname)[0]

                if is_authentic:
                    if not include_authentic:
                        continue
                    mask_path = None   # zero mask generated in __getitem__
                else:
                    # mask: <base>_GT.png
                    mask_path = os.path.join(
                        mask_dir, base + Config.MASK_SUFFIX + Config.MASK_EXT
                    )
                    if not os.path.exists(mask_path):
                        continue   # skip if mask missing

                self.items.append((
                    os.path.join(rgb_dir,   fname),
                    os.path.join(ela_dir,   base + ".png"),
                    os.path.join(noise_dir, base + ".png"),
                    mask_path,
                    label,
                ))

        from collections import Counter
        counts = Counter(item[4] for item in self.items)
        print(f"[LocalizerDataset:{split}]  total={len(self.items)}")
        for i, name in enumerate(Config.CLASS_NAMES):
            print(f"  {name:15s}: {counts.get(i, 0)}")

    def __len__(self):  return len(self.items)

    def __getitem__(self, idx):
        rgb_p, ela_p, noise_p, mask_path, label = self.items[idx]

        # Load mask or create zero mask
        if mask_path is not None:
            mask_img = Image.open(mask_path).convert("L")
        else:
            ref = Image.open(rgb_p)
            mask_img = Image.new("L", ref.size, 0)

        x, mask_img = _load_9ch(rgb_p, ela_p, noise_p, self.augmentor, mask_img)

        mask_t = TF.to_tensor(mask_img)          # (1, H, W) in [0,1]
        mask_t = (mask_t > 0.5).float()          # binarize

        return x, mask_t, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────
# DataLoader factories
# ─────────────────────────────────────────────────────────────────────────
def get_classifier_loaders():
    train_ds = ClassifierDataset("train", augment=True)
    val_ds   = ClassifierDataset("val",   augment=False)
    test_ds  = ClassifierDataset("test",  augment=False)

    kw = dict(num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    return (
        DataLoader(train_ds, Config.BATCH_SIZE, shuffle=True,  drop_last=True, **kw),
        DataLoader(val_ds,   Config.BATCH_SIZE, shuffle=False, **kw),
        DataLoader(test_ds,  Config.BATCH_SIZE, shuffle=False, **kw),
    )


def get_localizer_loaders(include_authentic: bool = True):
    train_ds = LocalizerDataset("train", augment=True,  include_authentic=include_authentic)
    val_ds   = LocalizerDataset("val",   augment=False, include_authentic=include_authentic)
    test_ds  = LocalizerDataset("test",  augment=False, include_authentic=include_authentic)

    kw = dict(num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    return (
        DataLoader(train_ds, Config.BATCH_SIZE, shuffle=True,  drop_last=True, **kw),
        DataLoader(val_ds,   Config.BATCH_SIZE, shuffle=False, **kw),
        DataLoader(test_ds,  Config.BATCH_SIZE, shuffle=False, **kw),
    )
