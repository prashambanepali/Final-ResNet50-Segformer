"""
test_evaluate.py
================
Evaluates best_localizer.pth on the test set.
Generates:
  - Overall metrics (F1, IoU, AUC, Accuracy, Precision, Recall)
  - Per-class metrics
  - ROC curve with EER marked
  - IoU bar chart per class
  - EER bar chart per class
  - Combined summary plot

    python test_evaluate.py
"""

import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from config import Config
from dataset import get_localizer_loaders
from models import LocalizerPipeline
from losses import CombinedSegLoss
from evaluate import compute_seg_metrics

CLASSIFIER_WEIGHTS = os.path.join(Config.SAVE_DIR, "best_classifier.pth")
LOCALIZER_WEIGHTS  = os.path.join(Config.SAVE_DIR, "best_localizer.pth")


# ─────────────────────────────────────────────────────────────────────────────
# EER computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_eer(preds_np, targets_np):
    """
    Compute Equal Error Rate (EER).
    EER = point where FAR (False Accept Rate) == FRR (False Reject Rate)
    In pixel-level terms:
      FAR = FPR (False Positive Rate)
      FRR = FNR = 1 - TPR (False Negative Rate)
    Returns: eer_value, eer_threshold, fpr, tpr, thresholds
    """
    targets_int = (targets_np >= 0.5).astype(int)
    preds_np    = np.clip(preds_np, 0.0, 1.0)

    try:
        fpr, tpr, thresholds = roc_curve(targets_int, preds_np)
    except Exception as e:
        print(f"[WARNING] ROC failed: {e}")
        return None, None, None, None, None

    fnr = 1.0 - tpr   # False Negative Rate = 1 - TPR

    # Find index where |FPR - FNR| is minimized
    diff    = np.abs(fpr - fnr)
    eer_idx = np.argmin(diff)
    eer     = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_thr = float(thresholds[eer_idx])

    return eer, eer_thr, fpr, tpr, thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_test(model, loader, device, criterion):
    model.eval()
    all_preds, all_masks = [], []
    class_preds = {i: [] for i in range(Config.NUM_CLASSES)}
    class_masks = {i: [] for i in range(Config.NUM_CLASSES)}
    total_loss  = 0.0

    pbar = tqdm(loader, desc="Evaluating test set",
                unit="batch", dynamic_ncols=True)

    for images, masks, class_idx in pbar:
        images    = images.to(device)
        masks     = masks.to(device)
        class_idx = class_idx.to(device)

        logits = model(images, class_idx)
        preds  = torch.sigmoid(logits.float())

        if criterion:
            loss, _ = criterion(logits, masks.float())
            total_loss += loss.item()

        pc = preds.float().cpu()
        mc = masks.float().cpu()
        all_preds.append(pc)
        all_masks.append(mc)

        for i, cls in enumerate(class_idx.cpu().tolist()):
            class_preds[cls].append(pc[i:i+1])
            class_masks[cls].append(mc[i:i+1])

    pbar.close()

    all_preds = torch.clamp(torch.cat(all_preds), 0.0, 1.0)
    all_masks = torch.clamp(torch.cat(all_masks), 0.0, 1.0)
    test_loss = round(total_loss / len(loader), 5)

    return all_preds, all_masks, class_preds, class_masks, test_loss


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — ROC curve with EER marked
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_with_eer(preds, masks, ckpt_epoch, save_dir):
    pn      = preds.numpy().flatten()
    tn      = masks.numpy().flatten()
    tn_int  = (tn >= 0.5).astype(int)
    pn      = np.clip(pn, 0.0, 1.0)

    eer, eer_thr, fpr, tpr, thresholds = compute_eer(pn, tn)
    if fpr is None:
        print("[WARNING] Could not compute ROC/EER.")
        return None

    auc = roc_auc_score(tn_int, pn)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0,1],[0,1], "navy", lw=1.5, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color="darkorange")

    # ── EER point ─────────────────────────────────────────────────────
    fnr     = 1.0 - tpr
    diff    = np.abs(fpr - fnr)
    eer_idx = np.argmin(diff)
    ax.plot(fpr[eer_idx], tpr[eer_idx], "ro", ms=10, zorder=5,
            label=f"EER = {eer:.4f}  (threshold={eer_thr:.3f})")
    ax.axvline(fpr[eer_idx], color="red", lw=1, linestyle=":",  alpha=0.5)
    ax.axhline(tpr[eer_idx], color="red", lw=1, linestyle=":",  alpha=0.5)
    ax.annotate(f"EER={eer:.4f}",
                xy=(fpr[eer_idx], tpr[eer_idx]),
                xytext=(fpr[eer_idx]+0.05, tpr[eer_idx]-0.08),
                fontsize=10, color="red",
                arrowprops=dict(arrowstyle="->", color="red"))

    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=13)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=13)
    ax.set_title(f"ROC Curve with EER — Test Set (Epoch {ckpt_epoch})",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "test_roc_eer.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  📈 ROC + EER → {path}")
    return eer


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — IoU per class bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_iou_per_class(class_metrics, save_dir):
    names  = []
    ious   = []
    f1s    = []

    for i in range(Config.NUM_CLASSES):
        if class_metrics.get(i) is not None:
            names.append(Config.CLASS_NAMES[i])
            ious.append(class_metrics[i]["iou"])
            f1s.append(class_metrics[i]["f1"])

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - w/2, ious, w, label="IoU",
                   color="steelblue",  alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + w/2, f1s,  w, label="F1",
                   color="darkorange", alpha=0.85, edgecolor="white")

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.015,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.015,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("IoU and F1 per Class — Test Set",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim([0, 1.12])
    ax.set_ylabel("Score")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=np.mean(ious), color="steelblue",
               linestyle="--", lw=1.5, alpha=0.7,
               label=f"Mean IoU = {np.mean(ious):.3f}")
    ax.axhline(y=np.mean(f1s),  color="darkorange",
               linestyle="--", lw=1.5, alpha=0.7,
               label=f"Mean F1  = {np.mean(f1s):.3f}")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(save_dir, "test_iou_per_class.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  📊 IoU per class → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — EER per class bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_eer_per_class(class_metrics, save_dir):
    names = []
    eers  = []

    for i in range(Config.NUM_CLASSES):
        m = class_metrics.get(i)
        if m is None:
            continue
        eer_val = m.get("eer")
        # Skip None, NaN, or authentic (no positive mask pixels)
        if eer_val is None or (isinstance(eer_val, float) and np.isnan(eer_val)):
            continue
        names.append(Config.CLASS_NAMES[i])
        eers.append(float(eer_val))

    if not eers:
        print("  [SKIP] No valid EER data for per-class plot.")
        return

    # Filter out any remaining NaN/Inf just in case
    valid = [(n, e) for n, e in zip(names, eers)
             if np.isfinite(e)]
    if not valid:
        print("  [SKIP] All EER values are NaN/Inf.")
        return
    names, eers = zip(*valid)
    names = list(names)
    eers  = list(eers)

    x      = np.arange(len(names))
    colors = ["#2196F3","#FF5722","#4CAF50","#FF9800","#9C27B0"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, eers, color=colors[:len(names)],
                  alpha=0.85, edgecolor="white", width=0.5)

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    mean_eer = np.mean(eers)
    ax.axhline(y=mean_eer, color="red", linestyle="--",
               lw=2, label=f"Mean EER = {mean_eer:.4f}")

    ax.set_title("Equal Error Rate (EER) per Class — Test Set\n"
                 "(Lower is better — perfect=0.0, random=0.5)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    y_max = min(max(eers) * 1.5 + 0.02, 1.0)
    ax.set_ylim([0, y_max])
    ax.set_ylabel("EER")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "test_eer_per_class.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  📊 EER per class → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Summary dashboard
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary_dashboard(class_metrics, overall, ckpt_epoch, save_dir):
    names  = [Config.CLASS_NAMES[i] for i in range(Config.NUM_CLASSES)
              if class_metrics.get(i) is not None]
    f1s    = [class_metrics[i]["f1"]  for i in range(Config.NUM_CLASSES)
              if class_metrics.get(i) is not None]
    ious   = [class_metrics[i]["iou"] for i in range(Config.NUM_CLASSES)
              if class_metrics.get(i) is not None]
    aucs   = [class_metrics[i]["auc"] for i in range(Config.NUM_CLASSES)
              if class_metrics.get(i) is not None]
    eers   = [class_metrics[i]["eer"] for i in range(Config.NUM_CLASSES)
              if class_metrics.get(i) is not None
              and class_metrics[i].get("eer") is not None
              and class_metrics[i]["eer"] is not None
              and np.isfinite(float(class_metrics[i]["eer"]))]

    x = np.arange(len(names))
    w = 0.22

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Test Set Evaluation Dashboard  —  Best Model (Epoch {ckpt_epoch})\n"
        f"Overall  F1={overall['f1']:.4f}  IoU={overall['iou']:.4f}  "
        f"AUC={overall['auc']:.4f}  Acc={overall['accuracy']:.4f}",
        fontsize=14, fontweight="bold"
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── F1 per class ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(x, f1s, color="steelblue", alpha=0.85, edgecolor="white")
    ax1.axhline(np.mean(f1s), color="red", lw=1.5, linestyle="--",
                label=f"Mean={np.mean(f1s):.3f}")
    for b in bars:
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                 f"{b.get_height():.3f}", ha="center", fontsize=8)
    ax1.set_title("F1 per Class", fontsize=12, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9, rotation=15)
    ax1.set_ylim([0, 1.15]); ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)

    # ── IoU per class ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(x, ious, color="darkorange", alpha=0.85, edgecolor="white")
    ax2.axhline(np.mean(ious), color="red", lw=1.5, linestyle="--",
                label=f"Mean={np.mean(ious):.3f}")
    for b in bars:
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                 f"{b.get_height():.3f}", ha="center", fontsize=8)
    ax2.set_title("IoU per Class", fontsize=12, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=9, rotation=15)
    ax2.set_ylim([0, 1.15]); ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)

    # ── AUC per class ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    bars = ax3.bar(x, aucs, color="#4CAF50", alpha=0.85, edgecolor="white")
    ax3.axhline(np.mean(aucs), color="red", lw=1.5, linestyle="--",
                label=f"Mean={np.mean(aucs):.3f}")
    for b in bars:
        ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
                 f"{b.get_height():.3f}", ha="center", fontsize=8)
    ax3.set_title("AUC-ROC per Class", fontsize=12, fontweight="bold")
    ax3.set_xticks(x); ax3.set_xticklabels(names, fontsize=9, rotation=15)
    ax3.set_ylim([0, 1.08]); ax3.legend(fontsize=9); ax3.grid(axis="y", alpha=0.3)

    # ── EER per class ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if eers:
        bars = ax4.bar(x[:len(eers)], eers, color="#FF5722",
                       alpha=0.85, edgecolor="white")
        ax4.axhline(np.mean(eers), color="red", lw=1.5, linestyle="--",
                    label=f"Mean={np.mean(eers):.4f}")
        for b in bars:
            ax4.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
                     f"{b.get_height():.4f}", ha="center", fontsize=8)
        ax4.set_ylim([0, min(max(eers)*1.5, 1.0)])
        ax4.legend(fontsize=9)
    ax4.set_title("EER per Class\n(Lower = Better)",
                  fontsize=12, fontweight="bold")
    ax4.set_xticks(x); ax4.set_xticklabels(names, fontsize=9, rotation=15)
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "test_summary_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  📊 Summary dashboard → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    if not os.path.exists(LOCALIZER_WEIGHTS):
        raise FileNotFoundError(
            f"No localizer checkpoint: {LOCALIZER_WEIGHTS}\n"
            f"Train first with train_localizer.py"
        )

    # ── Load model ────────────────────────────────────────────────────────
    ckpt  = torch.load(LOCALIZER_WEIGHTS, map_location=device)
    model = LocalizerPipeline(CLASSIFIER_WEIGHTS).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ckpt_epoch  = ckpt["epoch"]
    ckpt_val_f1 = ckpt["val_metrics"]["f1"]
    print(f"Loaded  : epoch {ckpt_epoch}  (val_F1={ckpt_val_f1:.4f})\n")

    # ── Data ──────────────────────────────────────────────────────────────
    _, _, test_loader = get_localizer_loaders(include_authentic=True)
    criterion = CombinedSegLoss(0.4, 0.4, 0.2).to(device)

    # ── Evaluate ──────────────────────────────────────────────────────────
    all_preds, all_masks, class_preds, \
    class_masks, test_loss = evaluate_test(
        model, test_loader, device, criterion
    )

    # ── Overall metrics ───────────────────────────────────────────────────
    overall       = compute_seg_metrics(all_preds, all_masks)
    overall["loss"] = test_loss

    pn_all     = all_preds.numpy().flatten()
    tn_all     = all_masks.numpy().flatten()
    eer_overall, eer_thr, _, _, _ = compute_eer(pn_all, tn_all)
    overall["eer"] = round(float(eer_overall), 6) if eer_overall else None

    print("\n── Overall Test Results ──────────────────────────────────────")
    for k, v in overall.items():
        print(f"  {k:12s}: {v}")

    # ── Per-class metrics ─────────────────────────────────────────────────
    print("\n── Per-Class Test Results ────────────────────────────────────")
    print(f"  {'Class':20s} {'F1':>7} {'IoU':>7} {'AUC':>7} "
          f"{'Acc':>7} {'EER':>8}")
    print("  " + "─" * 60)

    class_metrics = {}
    for cls_idx in range(Config.NUM_CLASSES):
        cls_name = Config.CLASS_NAMES[cls_idx]
        if not class_preds[cls_idx]:
            print(f"  {cls_name:20s}  no samples")
            class_metrics[cls_idx] = None
            continue

        cp  = torch.clamp(torch.cat(class_preds[cls_idx]), 0.0, 1.0)
        cm  = torch.clamp(torch.cat(class_masks[cls_idx]), 0.0, 1.0)
        m   = compute_seg_metrics(cp, cm)

        pn  = cp.numpy().flatten()
        tn  = cm.numpy().flatten()
        eer_val, eer_thr_cls, _, _, _ = compute_eer(pn, tn)
        m["eer"] = round(float(eer_val), 6) if eer_val is not None else None

        class_metrics[cls_idx] = m
        eer_str = f"{m['eer']:.4f}" if m["eer"] is not None else "  N/A "
        print(f"  {cls_name:20s} "
              f"{m['f1']:>7.4f} {m['iou']:>7.4f} "
              f"{m['auc']:>7.4f} {m['accuracy']:>7.4f} {eer_str:>8}")

    # ── Plots ─────────────────────────────────────────────────────────────
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    print("\n── Generating plots ──────────────────────────────────────────")

    plot_roc_with_eer(all_preds, all_masks, ckpt_epoch, Config.LOG_DIR)
    plot_iou_per_class(class_metrics, Config.LOG_DIR)
    plot_eer_per_class(class_metrics, Config.LOG_DIR)
    plot_summary_dashboard(class_metrics, overall, ckpt_epoch, Config.LOG_DIR)

    # ── Save JSON ─────────────────────────────────────────────────────────
    results = {
        "checkpoint_epoch": ckpt_epoch,
        "val_f1_at_save":   ckpt_val_f1,
        "overall":          overall,
        "per_class": {
            Config.CLASS_NAMES[i]: class_metrics[i]
            for i in range(Config.NUM_CLASSES)
            if class_metrics[i] is not None
        }
    }
    save_path = os.path.join(Config.LOG_DIR, "test_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  💾 Results  → {save_path}")
    print(f"  📁 All plots → {Config.LOG_DIR}/")
    print("\n── Files Generated ───────────────────────────────────────────")
    print("  test_roc_eer.png            ← ROC curve with EER marked")
    print("  test_iou_per_class.png      ← IoU + F1 bar chart per class")
    print("  test_eer_per_class.png      ← EER bar chart per class")
    print("  test_summary_dashboard.png  ← All metrics in one figure")
    print("  test_results.json           ← All numbers saved")


if __name__ == "__main__":
    main()
