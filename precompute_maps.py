"""
precompute_maps.py
==================
Reads raw images from raw/<class>/images/
Writes RGB copy, ELA (3-channel), Noise (3-channel) to features/<class>/

ELA channels  : quality 70, 80, 90  → 3 grayscale maps stacked as RGB PNG
Noise channels: Gaussian residual | SRM linear residual | SRM edge residual

Run once before training:
    python precompute_maps.py
"""

import os
import io
import cv2
import shutil
import numpy as np
from PIL import Image, ImageChops
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from config2 import Config

# ── SRM kernels ───────────────────────────────────────────────────────────
SRM_LINEAR = np.array([
    [ 0,  0, -1,  0,  0],
    [ 0, -1,  2, -1,  0],
    [-1,  2,  0,  2, -1],
    [ 0, -1,  2, -1,  0],
    [ 0,  0, -1,  0,  0],
], dtype=np.float32) / 8.0

SRM_EDGE = np.array([
    [-1, -1, -1, -1, -1],
    [-1,  2,  2,  2, -1],
    [-1,  2,  8,  2, -1],
    [-1,  2,  2,  2, -1],
    [-1, -1, -1, -1, -1],
], dtype=np.float32)
_s = SRM_EDGE.sum() - SRM_EDGE[2, 2]
SRM_EDGE[2, 2] = -_s
SRM_EDGE /= 8.0


def _pnorm(arr: np.ndarray) -> np.ndarray:
    """Percentile-normalize to uint8."""
    p2, p98 = np.percentile(arr, [2, 98])
    d = max(float(p98 - p2), 1e-6)
    return np.clip((arr - p2) / d * 255.0, 0, 255).astype(np.uint8)


# ── ELA ───────────────────────────────────────────────────────────────────
def _ela_single_gray(img_rgb: Image.Image, quality: int) -> np.ndarray:
    """Single-quality ELA → grayscale uint8 array."""
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recomp = Image.open(buf).convert("RGB")
    diff   = np.array(ImageChops.difference(img_rgb, recomp), dtype=np.float32)
    gray   = diff.mean(axis=2)          # collapse RGB → single channel
    return _pnorm(gray)


def compute_ela_3ch(img: Image.Image) -> Image.Image:
    """
    3-channel ELA: one grayscale ELA per quality level (70, 80, 90).
    Stored as RGB PNG:  R=q70  G=q80  B=q90
    """
    img_rgb = img.convert("RGB")
    chs = [_ela_single_gray(img_rgb, q) for q in Config.ELA_QUALITIES]
    return Image.fromarray(np.stack(chs, axis=2), mode="RGB")


# ── Noise ─────────────────────────────────────────────────────────────────
def compute_noise_3ch(img: Image.Image) -> Image.Image:
    """
    3-channel noise map:
      R = Gaussian residual   (sigma=1.5 high-pass)
      G = SRM linear residual
      B = SRM edge residual
    Stored as RGB PNG.
    """
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0

    # R: Gaussian residual
    blurred  = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    ch_r     = _pnorm(gray - blurred)

    # G: SRM linear
    ch_g = _pnorm(cv2.filter2D(gray, -1, SRM_LINEAR,
                               borderType=cv2.BORDER_REFLECT))

    # B: SRM edge
    ch_b = _pnorm(cv2.filter2D(gray, -1, SRM_EDGE,
                               borderType=cv2.BORDER_REFLECT))

    return Image.fromarray(np.stack([ch_r, ch_g, ch_b], axis=2), mode="RGB")


# ── Worker ────────────────────────────────────────────────────────────────
def worker(job: tuple) -> int:
    src_path, rgb_out, ela_out, noise_out = job
    try:
        img = Image.open(src_path).convert("RGB")

        if not os.path.exists(rgb_out):
            shutil.copy2(src_path, rgb_out)

        if not os.path.exists(ela_out):
            compute_ela_3ch(img).save(ela_out)

        if not os.path.exists(noise_out):
            compute_noise_3ch(img).save(noise_out)
    except Exception as e:
        print(f"[ERROR] {src_path}: {e}")
    return 1


# ── Job collection ────────────────────────────────────────────────────────
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".bmp")

def collect_jobs(class_name: str) -> list:
    src_dir   = os.path.join(Config.RAW_ROOT,      class_name, "images")
    rgb_dir   = os.path.join(Config.FEATURES_ROOT, class_name, "rgb")
    ela_dir   = os.path.join(Config.FEATURES_ROOT, class_name, "ela")
    noise_dir = os.path.join(Config.FEATURES_ROOT, class_name, "noise")

    if not os.path.isdir(src_dir):
        print(f"  [SKIP] {src_dir} not found")
        return []

    for d in [rgb_dir, ela_dir, noise_dir]:
        os.makedirs(d, exist_ok=True)

    jobs = []
    for fname in sorted(os.listdir(src_dir)):
        if not fname.lower().endswith(IMG_EXTS):
            continue
        base      = os.path.splitext(fname)[0]
        src_path  = os.path.join(src_dir,   fname)
        rgb_out   = os.path.join(rgb_dir,   fname)
        ela_out   = os.path.join(ela_dir,   base + ".png")
        noise_out = os.path.join(noise_dir, base + ".png")

        if not Config.OVERWRITE and all(
            os.path.exists(p) for p in [rgb_out, ela_out, noise_out]
        ):
            continue
        jobs.append((src_path, rgb_out, ela_out, noise_out))
    return jobs


def main():
    all_classes = Config.CLASS_NAMES
    print(f"RAW     : {os.path.abspath(Config.RAW_ROOT)}")
    print(f"OUTPUT  : {os.path.abspath(Config.FEATURES_ROOT)}")
    print(f"ELA     : 3-ch grayscale (q={Config.ELA_QUALITIES})")
    print(f"Noise   : 3-ch (Gaussian | SRM-linear | SRM-edge)")
    print(f"Workers : {Config.NUM_PROC_WORKERS}\n")

    for cls in all_classes:
        jobs = collect_jobs(cls)
        if not jobs:
            print(f"[{cls}] — already done or folder missing, skipping.")
            continue
        print(f"[{cls}]  {len(jobs)} images to process ...")
        with ProcessPoolExecutor(max_workers=Config.NUM_PROC_WORKERS) as ex:
            futs = [ex.submit(worker, j) for j in jobs]
            for _ in tqdm(as_completed(futs), total=len(futs),
                          unit="img", dynamic_ncols=True):
                pass

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────")
    for cls in all_classes:
        rgb_d = os.path.join(Config.FEATURES_ROOT, cls, "rgb")
        ela_d = os.path.join(Config.FEATURES_ROOT, cls, "ela")
        noi_d = os.path.join(Config.FEATURES_ROOT, cls, "noise")
        if os.path.isdir(rgb_d):
            nr = len(os.listdir(rgb_d))
            ne = len(os.listdir(ela_d))
            nn = len(os.listdir(noi_d))
            ok = "✓" if nr == ne == nn else "⚠ MISMATCH"
            print(f"  {cls:15s}  rgb={nr}  ela={ne}  noise={nn}  {ok}")

    print("\nDone. Next → python split_dataset.py")


if __name__ == "__main__":
    main()
