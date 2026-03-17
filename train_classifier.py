"""
train_classifier.py
===================
Trains ResNet50 on 9-channel (RGB+ELA+Noise) input for 5-class classification.
Run after precompute_maps.py and split_dataset.py.

    python train_classifier.py
"""

import os
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from config2 import Config
from dataset import get_classifier_loaders
from models2 import ResNet50Classifier
from evaluate import evaluate_classifier, plot_training_curves


def cosine_schedule(optimizer, warmup, total):
    def fn(ep):
        if ep < warmup:
            return ep / max(1, warmup)
        p = (ep - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    return LambdaLR(optimizer, fn)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch:03d} [Train]",
        unit="batch",
        dynamic_ncols=True,
        leave=False,        # clears bar after epoch finishes
    )

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast("cuda", enabled=Config.AMP):
            logits = model(images)
            loss   = criterion(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        # ── Live update inside the progress bar ───────────────────────
        running_loss = total_loss / (pbar.n + 1)
        running_acc  = correct / total
        pbar.set_postfix({
            "loss": f"{running_loss:.4f}",
            "acc":  f"{running_acc:.4f}",
        }, refresh=True)

    pbar.close()

    return {
        "loss":     round(total_loss / len(loader), 5),
        "accuracy": round(correct / total, 4),
    }


def main():
    torch.manual_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Classes: {Config.CLASS_NAMES}\n", flush=True)

    train_loader, val_loader, test_loader = get_classifier_loaders()

    model     = ResNet50Classifier(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                            lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = cosine_schedule(optimizer, Config.WARMUP_EPOCHS, Config.EPOCHS)
    scaler    = GradScaler("cuda", enabled=Config.AMP)

    best_acc = 0.0
    history  = []

    print(f"{'Epoch':>6} | {'TrLoss':>7} | {'TrAcc':>6} | {'VaLoss':>7} | {'VaAcc':>6}")
    print("─" * 50, flush=True)

    for epoch in range(1, Config.EPOCHS + 1):

        # ── Train with live progress bar ──────────────────────────────
        tr = train_one_epoch(model, train_loader, optimizer,
                             criterion, scaler, device, epoch)

        # ── Validate with progress bar ────────────────────────────────
        va = evaluate_classifier(model, val_loader, device, criterion,
                                 show_progress=True, epoch=epoch)

        scheduler.step()

        # ── Print epoch summary line ──────────────────────────────────
        print(
            f"{epoch:>6} | {tr['loss']:>7.4f} | {tr['accuracy']:>6.4f} | "
            f"{va['loss']:>7.4f} | {va['accuracy']:>6.4f}",
            flush=True,
        )

        history.append({
            "epoch":          epoch,
            "train_loss":     tr["loss"],
            "train_accuracy": tr["accuracy"],
            "val_loss":       va["loss"],
            "val_accuracy":   va["accuracy"],
        })

        if va["accuracy"] > best_acc:
            best_acc = va["accuracy"]
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer":        optimizer.state_dict(),
                "val_accuracy":     best_acc,
                "class_names":      Config.CLASS_NAMES,
            }, os.path.join(Config.SAVE_DIR, "best_classifier.pth"))
            print(f"         ✔ Saved best (val_acc={best_acc:.4f})", flush=True)

        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                os.path.join(Config.SAVE_DIR, f"classifier_ep{epoch:03d}.pth")
            )

    # ── Test ──────────────────────────────────────────────────────────────
    print("\n── Test evaluation ───────────────────────────────────────────")
    ckpt = torch.load(os.path.join(Config.SAVE_DIR, "best_classifier.pth"),
                      map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m = evaluate_classifier(model, test_loader, device, criterion,
                                 show_progress=True, epoch="Test")
    print(f"  Test Accuracy : {test_m['accuracy']}")
    print(f"  Test Loss     : {test_m['loss']}")

    plot_training_curves(history, Config.LOG_DIR, mode="classifier")

    with open(os.path.join(Config.LOG_DIR, "classifier_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  Weights → {Config.SAVE_DIR}/best_classifier.pth")
    print(f"  Logs    → {Config.LOG_DIR}/")
    print("\nNext → python train_localizer.py")


if __name__ == "__main__":
    main()
