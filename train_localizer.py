"""
train_localizer.py
==================
Trains SegFormer-B2 localizer using frozen ResNet50 classifier for conditioning.
Supports RESUME from checkpoint — just run again after stopping.

    python train_localizer.py
"""

import os
import math
import json
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from config import Config
from dataset import get_localizer_loaders
from models import LocalizerPipeline
from losses import CombinedSegLoss
from evaluate import (evaluate_localizer, plot_roc_curve,
                      plot_training_curves, compute_seg_metrics)

CLASSIFIER_WEIGHTS = os.path.join(Config.SAVE_DIR, "best_classifier.pth")
RESUME_CHECKPOINT  = os.path.join(Config.SAVE_DIR, "best_localizer.pth")


def cosine_schedule(optimizer, warmup, total):
    def fn(ep):
        if ep < warmup:
            return ep / max(1, warmup)
        p = (ep - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    return LambdaLR(optimizer, fn)


def build_optimizer(model):
    enc_params = list(model.localizer.encoder.parameters())
    dec_params = (
        list(model.localizer.input_proj.parameters())  +
        list(model.localizer.proj.parameters())        +
        list(model.localizer.class_embed.parameters()) +
        list(model.localizer.dec4.parameters())        +
        list(model.localizer.dec3.parameters())        +
        list(model.localizer.dec2.parameters())        +
        list(model.localizer.dec1.parameters())        +
        list(model.localizer.head.parameters())
    )
    return optim.AdamW([
        {"params": enc_params, "lr": Config.ENCODER_LR},
        {"params": dec_params, "lr": Config.DECODER_LR},
    ], weight_decay=Config.LOC_WEIGHT_DECAY)


def train_one_epoch(model, loader, optimizer, criterion,
                    scaler, device, epoch, accum_steps=4):
    model.train()
    total_loss = 0.0
    all_preds, all_masks = [], []
    optimizer.zero_grad()

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch:03d} [Train]",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )

    for step, (images, masks, class_idx) in enumerate(pbar):
        images    = images.to(device,    non_blocking=True)
        masks     = masks.to(device,     non_blocking=True)
        class_idx = class_idx.to(device, non_blocking=True)

        with autocast("cuda", enabled=Config.AMP):
            preds       = model(images, class_idx)
            loss, _     = criterion(preds, masks.float())
            loss_scaled = loss / accum_steps

        scaler.scale(loss_scaled).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), Config.LOC_GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()

        # ── Collect preds and masks in float32 for metrics ────────────
        all_preds.append(torch.sigmoid(preds.float()).detach().cpu())
        all_masks.append(masks.float().detach().cpu())  # ← fixed

        pbar.set_postfix({
            "loss": f"{total_loss/(step+1):.4f}",
        }, refresh=True)

    pbar.close()

    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)
    m = compute_seg_metrics(all_preds, all_masks)
    m["loss"] = round(total_loss / len(loader), 5)
    return m, all_preds, all_masks


def main():
    torch.manual_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}", flush=True)

    if not os.path.exists(CLASSIFIER_WEIGHTS):
        raise FileNotFoundError(
            f"Classifier weights not found: {CLASSIFIER_WEIGHTS}\n"
            f"Run train_classifier.py first."
        )

    train_loader, val_loader, test_loader = get_localizer_loaders(
        include_authentic=True
    )

    model     = LocalizerPipeline(CLASSIFIER_WEIGHTS).to(device)
    criterion = CombinedSegLoss(0.4, 0.4, 0.2).to(device)
    optimizer = build_optimizer(model)
    scheduler = cosine_schedule(optimizer, Config.LOC_WARMUP_EPOCHS, Config.LOC_EPOCHS)
    scaler    = GradScaler("cuda", enabled=Config.AMP)

    print(f"Localizer params : {sum(p.numel() for p in model.localizer.parameters()):,}")
    print(f"Classifier frozen: {sum(p.numel() for p in model.classifier.parameters()):,}",
          flush=True)

    # ── Resume from checkpoint if exists ─────────────────────────────────
    start_epoch = 1
    best_f1     = 0.0
    history     = []

    if os.path.exists(RESUME_CHECKPOINT):
        print(f"\nResuming from {RESUME_CHECKPOINT} ...", flush=True)
        ckpt        = torch.load(RESUME_CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_f1     = ckpt["val_metrics"]["f1"]

        # Advance scheduler to correct epoch
        for _ in range(ckpt["epoch"]):
            scheduler.step()

        # Load history if exists
        hist_path = os.path.join(Config.LOG_DIR, "localizer_history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)

        print(f"Resumed  : epoch {ckpt['epoch']} → continuing from epoch {start_epoch}")
        print(f"Best F1  : {best_f1:.4f}\n", flush=True)
    else:
        print("\nNo checkpoint found — starting fresh.\n", flush=True)

    print(f"{'Epoch':>6} | {'TrLoss':>7} | {'TrAcc':>6} | {'TrF1':>6} | "
          f"{'VaLoss':>7} | {'VaAcc':>6} | {'VaF1':>6} | "
          f"{'VaIoU':>6} | {'VaAUC':>6}")
    print("─" * 82, flush=True)

    for epoch in range(start_epoch, Config.LOC_EPOCHS + 1):

        # ── Train ─────────────────────────────────────────────────────
        tr, _, _ = train_one_epoch(model, train_loader, optimizer,
                                   criterion, scaler, device, epoch)

        # ── Validate ──────────────────────────────────────────────────
        va, vp, vm = evaluate_localizer(model, val_loader, device, criterion,
                                        show_progress=True, epoch=epoch)
        scheduler.step()

        # ── Print summary ─────────────────────────────────────────────
        print(
            f"{epoch:>6} | {tr['loss']:>7.4f} | {tr['accuracy']:>6.4f} | "
            f"{tr['f1']:>6.4f} | {va['loss']:>7.4f} | {va['accuracy']:>6.4f} | "
            f"{va['f1']:>6.4f} | {va['iou']:>6.4f} | {va['auc']:>6.4f}",
            flush=True,
        )

        history.append({
            "epoch":          epoch,
            "train_loss":     tr["loss"],     "train_accuracy": tr["accuracy"],
            "train_f1":       tr["f1"],       "train_iou":      tr["iou"],
            "val_loss":       va["loss"],     "val_accuracy":   va["accuracy"],
            "val_f1":         va["f1"],       "val_iou":        va["iou"],
            "val_auc":        va["auc"],
        })

        # ── Save best ─────────────────────────────────────────────────
        if va["f1"] > best_f1:
            best_f1 = va["f1"]
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer":        optimizer.state_dict(),
                "val_metrics":      va,
            }, os.path.join(Config.SAVE_DIR, "best_localizer.pth"))
            print(f"         ✔ Saved best (val_F1={best_f1:.4f})", flush=True)

        # ── Periodic checkpoint ───────────────────────────────────────
        if epoch % 10 == 0 or epoch == Config.LOC_EPOCHS:
            plot_roc_curve(vp, vm, epoch, Config.LOG_DIR, split="val")
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                os.path.join(Config.SAVE_DIR, f"localizer_ep{epoch:03d}.pth")
            )

        # ── Save history after every epoch ────────────────────────────
        with open(os.path.join(Config.LOG_DIR, "localizer_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    # ── Final training curves ─────────────────────────────────────────────
    plot_training_curves(history, Config.LOG_DIR, mode="localizer")

    # ── Test ──────────────────────────────────────────────────────────────
    print("\n── Test evaluation ───────────────────────────────────────────")
    ckpt = torch.load(os.path.join(Config.SAVE_DIR, "best_localizer.pth"),
                      map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    te, tp, tm = evaluate_localizer(model, test_loader, device, criterion,
                                    show_progress=True, epoch="Test")
    plot_roc_curve(tp, tm, ckpt["epoch"], Config.LOG_DIR, split="test")

    print("\n── Test Results ──────────────────────────────────────────────")
    for k, v in te.items():
        print(f"  {k:12s}: {v}", flush=True)

    with open(os.path.join(Config.LOG_DIR, "localizer_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  Weights → {Config.SAVE_DIR}/best_localizer.pth")
    print(f"  Logs    → {Config.LOG_DIR}/")


if __name__ == "__main__":
    main()
