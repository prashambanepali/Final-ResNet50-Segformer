import os

class Config:
    # ── Raw dataset root ───────────────────────────────────────────────────
    # Expected structure:
    #   raw/
    #     copymove/
    #       images/   COCO_DF_*.jpg
    #       masks/    COCO_DF_*_GT.png   (not needed for authentic)
    #     splicing/   images/ masks/
    #     removal/    images/ masks/
    #     enhancement/images/ masks/
    #     authentic/  images/            (no masks folder)
    RAW_ROOT      = "raw"

    # ── Precomputed features (written by precompute_maps.py) ───────────────
    # features/
    #   copymove/  rgb/  ela/  noise/
    #   ...
    FEATURES_ROOT = "features"

    # ── Split index files (written by split_dataset.py) ────────────────────
    # dataset/
    #   train/copymove/index.txt  ...
    DATASET_ROOT  = "dataset"

    # ── Checkpoints & logs ─────────────────────────────────────────────────
    SAVE_DIR = "checkpoints"
    LOG_DIR  = "logs"

    # ── Class names (must stay in this order — index = label) ─────────────
    CLASS_NAMES = [
        "authentic",    # 0
        "copymove",     # 1
        "enhancement",  # 2
        "removal",      # 3
        "splicing",     # 4
    ]
    TAMPERED_CLASSES = ["copymove", "enhancement", "removal", "splicing"]
    NUM_CLASSES      = 5

    # ── Mask naming ────────────────────────────────────────────────────────
    # image  : COCO_DF_C100B00000_00400002.jpg
    # mask   : COCO_DF_C100B00000_00400002_GT.png
    MASK_SUFFIX = "_GT"        # appended to image base name
    MASK_EXT    = ".png"

    # ── ELA settings ──────────────────────────────────────────────────────
    ELA_QUALITIES = (70, 80, 90)   # 3 quality levels → 3 ELA channels

    # ── Image ─────────────────────────────────────────────────────────────
    IMAGE_SIZE  = 224
    IN_CHANNELS = 9    # RGB(3) + ELA(3) + Noise(3)

    # ── Training ──────────────────────────────────────────────────────────
    BATCH_SIZE    = 16
    EPOCHS        = 60
    WARMUP_EPOCHS = 5
    LR            = 1e-4
    WEIGHT_DECAY  = 1e-4
    GRAD_CLIP     = 1.0
    AMP           = True

    # ── Data split ────────────────────────────────────────────────────────
    TRAIN_RATIO = 0.80
    VAL_RATIO   = 0.10
    TEST_RATIO  = 0.10
    SEED        = 42

    # ── Loader ────────────────────────────────────────────────────────────
    NUM_WORKERS = 4
    PIN_MEMORY  = True

    # ── Precompute ────────────────────────────────────────────────────────
    OVERWRITE   = False
    NUM_PROC_WORKERS = 8

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)
