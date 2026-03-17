"""
plot_history.py
===============
Plot training curves from saved history JSON.
Can be run WHILE training is still running — reads whatever epochs are saved.

    python plot_history.py
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Config ────────────────────────────────────────────────────────────────
LOG_DIR          = "logs"
HISTORY_FILE     = os.path.join(LOG_DIR, "localizer_history.json")
OUTPUT_FILE      = os.path.join(LOG_DIR, "training_progress.png")
# ─────────────────────────────────────────────────────────────────────────


def load_history(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"History file not found: {path}\n"
            f"Make sure training has started and saved at least 1 epoch."
        )
    with open(path) as f:
        return json.load(f)


def plot(history, save_path):
    epochs   = [h["epoch"]          for h in history]
    tr_loss  = [h["train_loss"]     for h in history]
    va_loss  = [h["val_loss"]       for h in history]
    tr_acc   = [h["train_accuracy"] for h in history]
    va_acc   = [h["val_accuracy"]   for h in history]
    tr_f1    = [h["train_f1"]       for h in history]
    va_f1    = [h["val_f1"]         for h in history]
    va_iou   = [h["val_iou"]        for h in history]
    va_auc   = [h["val_auc"]        for h in history]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Localizer Training Progress  (epochs 1–{epochs[-1]})",
        fontsize=16, fontweight="bold", y=1.01
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. Loss ───────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, tr_loss, "b-o", ms=4, lw=1.5, label="Train Loss")
    ax1.plot(epochs, va_loss, "r-o", ms=4, lw=1.5, label="Val Loss")
    ax1.set_title("Loss vs Epoch",     fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch");           ax1.set_ylabel("Loss")
    ax1.legend(fontsize=10);           ax1.grid(alpha=0.3)
    ax1.set_xlim(left=1)

    # ── 2. Accuracy ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, tr_acc, "b-o", ms=4, lw=1.5, label="Train Accuracy")
    ax2.plot(epochs, va_acc, "r-o", ms=4, lw=1.5, label="Val Accuracy")
    ax2.set_title("Accuracy vs Epoch", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch");           ax2.set_ylabel("Pixel Accuracy")
    ax2.legend(fontsize=10);           ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1]);              ax2.set_xlim(left=1)

    # ── 3. F1 Score ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, tr_f1, "b-o", ms=4, lw=1.5, label="Train F1")
    ax3.plot(epochs, va_f1, "g-o", ms=4, lw=1.5, label="Val F1")
    ax3.set_title("F1 Score vs Epoch", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Epoch");           ax3.set_ylabel("F1")
    ax3.legend(fontsize=10);           ax3.grid(alpha=0.3)
    ax3.set_ylim([0, 1]);              ax3.set_xlim(left=1)

    # ── 4. IoU ────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, va_iou, "m-o", ms=4, lw=1.5, label="Val IoU")
    ax4.set_title("IoU vs Epoch",      fontsize=13, fontweight="bold")
    ax4.set_xlabel("Epoch");           ax4.set_ylabel("IoU")
    ax4.legend(fontsize=10);           ax4.grid(alpha=0.3)
    ax4.set_ylim([0, 1]);              ax4.set_xlim(left=1)

    # ── 5. AUC ────────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, va_auc, "c-o", ms=4, lw=1.5, label="Val AUC")
    ax5.set_title("AUC-ROC vs Epoch",  fontsize=13, fontweight="bold")
    ax5.set_xlabel("Epoch");           ax5.set_ylabel("AUC")
    ax5.legend(fontsize=10);           ax5.grid(alpha=0.3)
    ax5.set_ylim([0, 1]);              ax5.set_xlim(left=1)

    # ── 6. All val metrics together ───────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(epochs, va_f1,  "g-o", ms=4, lw=1.5, label="Val F1")
    ax6.plot(epochs, va_iou, "m-o", ms=4, lw=1.5, label="Val IoU")
    ax6.plot(epochs, va_auc, "c-o", ms=4, lw=1.5, label="Val AUC")
    ax6.plot(epochs, va_acc, "r-o", ms=4, lw=1.5, label="Val Acc")
    ax6.set_title("All Val Metrics",   fontsize=13, fontweight="bold")
    ax6.set_xlabel("Epoch");           ax6.set_ylabel("Score")
    ax6.legend(fontsize=9);            ax6.grid(alpha=0.3)
    ax6.set_ylim([0, 1]);              ax6.set_xlim(left=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Plot saved → {save_path}")

    # ── Print summary table ───────────────────────────────────────────────
    print(f"\n── Summary (last 5 epochs) ───────────────────────────────────")
    print(f"{'Epoch':>6} | {'TrLoss':>7} | {'TrAcc':>6} | {'TrF1':>6} | "
          f"{'VaLoss':>7} | {'VaAcc':>6} | {'VaF1':>6} | "
          f"{'VaIoU':>6} | {'VaAUC':>6}")
    print("─" * 82)
    for h in history[-5:]:
        print(f"{h['epoch']:>6} | {h['train_loss']:>7.4f} | "
              f"{h['train_accuracy']:>6.4f} | {h['train_f1']:>6.4f} | "
              f"{h['val_loss']:>7.4f} | {h['val_accuracy']:>6.4f} | "
              f"{h['val_f1']:>6.4f} | {h['val_iou']:>6.4f} | "
              f"{h['val_auc']:>6.4f}")

    best = max(history, key=lambda x: x["val_f1"])
    print(f"\n── Best epoch: {best['epoch']}  "
          f"val_F1={best['val_f1']:.4f}  "
          f"val_IoU={best['val_iou']:.4f}  "
          f"val_AUC={best['val_auc']:.4f}")


if __name__ == "__main__":
    history = load_history(HISTORY_FILE)
    print(f"Loaded {len(history)} epochs from {HISTORY_FILE}")
    plot(history, OUTPUT_FILE)
