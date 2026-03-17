import os

class Config:
    # ── Paths ─────────────────────────────────────────────────────────────
    RAW_ROOT      = "raw"
    FEATURES_ROOT = "features"
    DATASET_ROOT  = "dataset"
    SAVE_DIR      = "checkpoints"
    LOG_DIR       = "logs"

    # ── Classes ───────────────────────────────────────────────────────────
    CLASS_NAMES = [
        "authentic",    # 0
        "copymove",     # 1
        "enhancement",  # 2
        "removal",      # 3
        "splicing",     # 4
    ]
    TAMPERED_CLASSES = ["copymove", "enhancement", "removal", "splicing"]
    NUM_CLASSES      = 5

    # ── Mask naming ───────────────────────────────────────────────────────
    # image : COCO_DF_C100B00000_00400002.jpg
    # mask  : COCO_DF_C100B00000_00400002_GT.png
    MASK_SUFFIX = "_GT"
    MASK_EXT    = ".png"

    # ── ELA ───────────────────────────────────────────────────────────────
    ELA_QUALITIES = (70, 80, 90)

    # ── Image ─────────────────────────────────────────────────────────────
    IMAGE_SIZE  = 224
    IN_CHANNELS = 9        # RGB(3) + ELA(3) + Noise(3)

    # ─────────────────────────────────────────────────────────────────────
    # CLASSIFIER settings  (used by train_classifier.py)
    # ─────────────────────────────────────────────────────────────────────
    BATCH_SIZE    = 16
    EPOCHS        = 60
    WARMUP_EPOCHS = 5
    LR            = 1e-4
    WEIGHT_DECAY  = 1e-4
    GRAD_CLIP     = 1.0
    AMP           = True

    # Adaptive LR — ReduceLROnPlateau
    LR_FACTOR   = 0.5      # multiply LR by this when plateau
    LR_PATIENCE = 5        # epochs to wait before reducing LR
    LR_MIN      = 1e-6     # minimum LR floor

    # Early stopping — classifier
    EARLY_STOP_PATIENCE   = 10    # stop after N epochs no improvement
    EARLY_STOP_MIN_DELTA  = 1e-4  # minimum improvement to reset counter

    # ─────────────────────────────────────────────────────────────────────
    # LOCALIZER settings  (used by train_localizer.py)
    # ─────────────────────────────────────────────────────────────────────
    LOC_BATCH_SIZE    = 8          # smaller batch for SegFormer (VRAM)
    LOC_EPOCHS        = 15
    LOC_WARMUP_EPOCHS = 5
    LOC_WEIGHT_DECAY  = 1e-4
    LOC_GRAD_CLIP     = 1.0
    LOC_ACCUM_STEPS   = 4          # effective batch = 8×4 = 32

    # Separate LR for encoder vs decoder
    ENCODER_LR = 6e-5              # lower — pretrained SegFormer encoder
    DECODER_LR = 6e-4              # higher — fresh decoder + projection

    # Adaptive LR — localizer
    LOC_LR_FACTOR   = 0.5
    LOC_LR_PATIENCE = 5
    LOC_LR_MIN      = 1e-7

    # Early stopping — localizer (monitors val F1)
    LOC_EARLY_STOP_PATIENCE  = 8
    LOC_EARLY_STOP_MIN_DELTA = 1e-4

    # Loss weights
    BCE_WEIGHT  = 0.4
    DICE_WEIGHT = 0.4
    EDGE_WEIGHT = 0.2

    # ─────────────────────────────────────────────────────────────────────
    # Shared
    # ─────────────────────────────────────────────────────────────────────
    TRAIN_RATIO = 0.80
    VAL_RATIO   = 0.10
    TEST_RATIO  = 0.10
    SEED        = 42
    NUM_WORKERS = 4
    PIN_MEMORY  = True

    # Precompute
    OVERWRITE        = False
    NUM_PROC_WORKERS = 8

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)