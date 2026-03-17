import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def compute_seg_metrics(preds, targets, threshold=0.5):
    """
    preds  : float tensor in [0,1]  — sigmoid already applied
    targets: float tensor in {0,1}  — binary mask
    """
    # Force float32 numpy — prevents any dtype issues
    pn = preds.detach().float().cpu().numpy().flatten()
    tn = targets.detach().float().cpu().numpy().flatten()

    # Clamp to valid range just in case
    pn = np.clip(pn, 0.0, 1.0)
    tn = np.clip(tn, 0.0, 1.0)

    tn_int = (tn >= 0.5).astype(int)
    bn     = (pn >= threshold).astype(int)

    tp  = ((bn==1)&(tn_int==1)).sum()
    fp  = ((bn==1)&(tn_int==0)).sum()
    fn  = ((bn==0)&(tn_int==1)).sum()
    tnn = ((bn==0)&(tn_int==0)).sum()

    total = tp + fp + fn + tnn
    prec  = tp / (tp + fp  + 1e-8)
    rec   = tp / (tp + fn  + 1e-8)
    f1    = 2*prec*rec / (prec + rec + 1e-8)
    iou   = tp / (tp + fp + fn + 1e-8)
    acc   = (tp + tnn) / (total + 1e-8)

    try:    auc = roc_auc_score(tn_int, pn)
    except: auc = 0.0

    return {
        "f1":        round(float(f1),   4),
        "iou":       round(float(iou),  4),
        "auc":       round(float(auc),  4),
        "accuracy":  round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
    }


@torch.no_grad()
def evaluate_classifier(model, loader, device, criterion=None,
                        show_progress=False, epoch=""):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    pbar = tqdm(loader,
                desc=f"Epoch {epoch:>3} [ Val ]",
                unit="batch",
                dynamic_ncols=True,
                leave=False) if show_progress else loader

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)

        if criterion:
            total_loss += criterion(logits, labels).item()

        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)

        if show_progress:
            pbar.set_postfix({
                "loss": f"{total_loss/(pbar.n+1):.4f}",
                "acc":  f"{correct/total:.4f}",
            }, refresh=True)

    if show_progress:
        pbar.close()

    return {
        "accuracy": round(correct / total, 4),
        "loss":     round(total_loss / len(loader), 5),
    }


@torch.no_grad()
def evaluate_localizer(model, loader, device, criterion=None,
                       show_progress=False, epoch=""):
    model.eval()
    all_preds, all_masks = [], []
    total_loss = 0.0

    pbar = tqdm(loader,
                desc=f"Epoch {epoch:>3} [ Val ]",
                unit="batch",
                dynamic_ncols=True,
                leave=False) if show_progress else loader

    for images, masks, class_idx in pbar:
        images    = images.to(device)
        masks     = masks.to(device)
        class_idx = class_idx.to(device)

        logits = model(images, class_idx)
        preds  = torch.sigmoid(logits.float())   # float32 probs

        if criterion:
            loss, _ = criterion(logits, masks.float())
            total_loss += loss.item()

        all_preds.append(preds.float().cpu())
        all_masks.append(masks.float().cpu())

        if show_progress:
            pbar.set_postfix({
                "loss": f"{total_loss/(pbar.n+1):.4f}",
            }, refresh=True)

    if show_progress:
        pbar.close()

    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)
    metrics   = compute_seg_metrics(all_preds, all_masks)
    metrics["loss"] = round(total_loss / len(loader), 5)
    return metrics, all_preds, all_masks


def plot_roc_curve(all_preds, all_masks, epoch, save_dir, split="val"):
    pn = all_preds.float().numpy().flatten()
    tn = all_masks.float().numpy().flatten()
    tn_int = (tn >= 0.5).astype(int)
    pn = np.clip(pn, 0.0, 1.0)

    try:
        fpr, tpr, _ = roc_curve(tn_int, pn)
        auc         = roc_auc_score(tn_int, pn)
    except:
        print("[WARNING] Could not compute ROC curve.")
        return

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0,1],[0,1], "navy", lw=1.5, linestyle="--", label="Random")
    plt.fill_between(fpr, tpr, alpha=0.1, color="darkorange")
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate",  fontsize=13)
    plt.title(f"ROC — {split.upper()} (Epoch {epoch})", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3); plt.tight_layout()
    path = os.path.join(save_dir, f"roc_{split}_ep{epoch:03d}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  📈 ROC → {path}", flush=True)


def plot_training_curves(history, save_dir, mode="localizer"):
    epochs = [h["epoch"] for h in history]

    if mode == "classifier":
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Classifier Training", fontsize=15, fontweight="bold")
        axes[0].plot(epochs, [h["train_loss"] for h in history], "b-o", ms=3, label="Train")
        axes[0].plot(epochs, [h["val_loss"]   for h in history], "r-o", ms=3, label="Val")
        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(epochs, [h["train_accuracy"] for h in history], "b-o", ms=3, label="Train")
        axes[1].plot(epochs, [h["val_accuracy"]   for h in history], "r-o", ms=3, label="Val")
        axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
        axes[1].legend(); axes[1].grid(alpha=0.3)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Localizer Training", fontsize=15, fontweight="bold")
        axes[0,0].plot(epochs,[h["train_loss"] for h in history],"b-o",ms=3,label="Train")
        axes[0,0].plot(epochs,[h["val_loss"]   for h in history],"r-o",ms=3,label="Val")
        axes[0,0].set_title("Loss"); axes[0,0].set_xlabel("Epoch")
        axes[0,0].legend(); axes[0,0].grid(alpha=0.3)
        axes[0,1].plot(epochs,[h["train_accuracy"] for h in history],"b-o",ms=3,label="Train")
        axes[0,1].plot(epochs,[h["val_accuracy"]   for h in history],"r-o",ms=3,label="Val")
        axes[0,1].set_title("Pixel Accuracy"); axes[0,1].set_xlabel("Epoch")
        axes[0,1].legend(); axes[0,1].grid(alpha=0.3)
        axes[1,0].plot(epochs,[h["val_f1"]  for h in history],"g-o",ms=3,label="Val F1")
        axes[1,0].plot(epochs,[h["val_iou"] for h in history],"m-o",ms=3,label="Val IoU")
        axes[1,0].set_title("F1 & IoU"); axes[1,0].set_xlabel("Epoch")
        axes[1,0].legend(); axes[1,0].grid(alpha=0.3)
        axes[1,1].plot(epochs,[h["val_auc"] for h in history],"c-o",ms=3,label="Val AUC")
        axes[1,1].set_title("AUC-ROC"); axes[1,1].set_xlabel("Epoch")
        axes[1,1].set_ylim([0,1]); axes[1,1].legend()
        axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{mode}_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  📊 Curves → {path}", flush=True)
