"""
evaluate_classifier.py
=======================
Evaluates best_classifier.pth on the test set.
Generates:
  - Overall accuracy, F1, Precision, Recall
  - Per-class metrics
  - Confusion matrix
  - ROC curves (all classes + per class)
  - Per-class F1 + Accuracy bar charts
  - EER per class
  - predictions.csv + top_errors.csv

Usage:
    python evaluate_classifier.py
"""

import os
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)

from config import Config
from dataset import get_classifier_loaders
from models import ResNet50Classifier

CLASSIFIER_WEIGHTS = os.path.join(Config.SAVE_DIR, "best_classifier.pth")
OUTPUT_DIR         = "eval_classifier"
TOP_K_ERRORS       = 20
USE_TTA            = True    # Test-Time Augmentation (4 flips)


# ─────────────────────────────────────────────────────────────────────────────
# TTA — 4 flip augmentations averaged
# ─────────────────────────────────────────────────────────────────────────────
def tta_predict(model, images, device):
    """Average predictions over 4 flips."""
    preds = []
    for flip_h, flip_v in [(False,False),(True,False),(False,True),(True,True)]:
        x = images.clone()
        if flip_h: x = torch.flip(x, dims=[3])
        if flip_v: x = torch.flip(x, dims=[2])
        with torch.no_grad():
            logits = model(x.to(device))
            preds.append(torch.softmax(logits, dim=1).cpu())
    return torch.stack(preds).mean(0)   # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# EER computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_eer(y_true_binary, y_score):
    try:
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
    except Exception:
        return None, None
    fnr     = 1.0 - tpr
    diff    = np.abs(fpr - fnr)
    idx     = np.argmin(diff)
    eer     = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_thr = float(thresholds[idx])
    return eer, eer_thr


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_evaluation(model, loader, device, use_tta=True):
    model.eval()
    all_probs   = []
    all_preds   = []
    all_labels  = []
    all_fnames  = []

    pbar = tqdm(loader, desc="Evaluating", unit="batch", dynamic_ncols=True)

    for batch in pbar:
        # dataset returns (images, labels) — no filenames in loader
        images, labels = batch
        images = images.to(device)

        if use_tta:
            probs = tta_predict(model, images, device)
        else:
            with torch.no_grad():
                logits = model(images)
                probs  = torch.softmax(logits, dim=1).cpu()

        preds = probs.argmax(dim=1)
        all_probs.append(probs.numpy())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    all_probs  = np.vstack(all_probs)      # (N, C)
    all_preds  = np.array(all_preds)       # (N,)
    all_labels = np.array(all_labels)      # (N,)

    return all_probs, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# Compute metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(all_probs, all_preds, all_labels):
    n_classes   = len(Config.CLASS_NAMES)
    overall_acc = (all_preds == all_labels).mean()

    # Per-class metrics
    report = classification_report(
        all_labels, all_preds,
        target_names=Config.CLASS_NAMES,
        output_dict=True, zero_division=0
    )

    per_class = {}
    for cls in Config.CLASS_NAMES:
        r = report[cls]
        per_class[cls] = {
            "precision": round(r["precision"], 4),
            "recall":    round(r["recall"],    4),
            "f1":        round(r["f1-score"],  4),
            "support":   int(r["support"]),
            "accuracy":  round((all_preds[all_labels == Config.CLASS_NAMES.index(cls)]
                                == Config.CLASS_NAMES.index(cls)).mean(), 4)
            if (all_labels == Config.CLASS_NAMES.index(cls)).sum() > 0 else 0.0,
        }

    # AUC + EER per class (one-vs-rest)
    auc_dict = {}
    eer_dict = {}
    fpr_dict = {}
    tpr_dict = {}

    for i, cls in enumerate(Config.CLASS_NAMES):
        y_bin   = (all_labels == i).astype(int)
        y_score = all_probs[:, i]

        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            auc_dict[cls] = 0.0
            eer_dict[cls] = None
            fpr_dict[cls] = np.array([0.0, 1.0])
            tpr_dict[cls] = np.array([0.0, 1.0])
            continue

        try:
            fpr, tpr, thr = roc_curve(y_bin, y_score)
            auc           = roc_auc_score(y_bin, y_score)
        except Exception:
            auc = 0.0
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])

        auc_dict[cls] = round(float(auc), 4)
        fpr_dict[cls] = fpr
        tpr_dict[cls] = tpr

        eer, eer_thr  = compute_eer(y_bin, y_score)
        eer_dict[cls] = round(float(eer), 6) if eer is not None else None
        per_class[cls]["auc"] = auc_dict[cls]
        per_class[cls]["eer"] = eer_dict[cls]

    macro_f1        = np.mean([per_class[c]["f1"]        for c in Config.CLASS_NAMES])
    macro_precision = np.mean([per_class[c]["precision"] for c in Config.CLASS_NAMES])
    macro_recall    = np.mean([per_class[c]["recall"]    for c in Config.CLASS_NAMES])
    macro_auc       = np.mean([auc_dict[c]               for c in Config.CLASS_NAMES])

    return {
        "overall_acc":      round(float(overall_acc), 4),
        "macro_f1":         round(float(macro_f1),        4),
        "macro_precision":  round(float(macro_precision), 4),
        "macro_recall":     round(float(macro_recall),    4),
        "macro_auc":        round(float(macro_auc),       4),
        "per_class":        per_class,
        "auc_dict":         auc_dict,
        "eer_dict":         eer_dict,
        "fpr_dict":         fpr_dict,
        "tpr_dict":         tpr_dict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(all_preds, all_labels, output_dir):
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")

    for ax, data, title, fmt in [
        (axes[0], cm,      "Raw Counts",        "d"),
        (axes[1], cm_norm, "Normalised (Recall)", ".2f"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=Config.CLASS_NAMES,
                    yticklabels=Config.CLASS_NAMES,
                    ax=ax, linewidths=0.5)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True",      fontsize=11)
        ax.set_title(title,        fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — ROC curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curves(metrics, all_probs, all_labels, output_dir):
    COLORS = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6"]
    fpr_d  = metrics["fpr_dict"]
    tpr_d  = metrics["tpr_dict"]
    auc_d  = metrics["auc_dict"]

    # ── Combined ROC ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#f8f9fa")
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="Random (AUC=0.50)")
    for i, cls in enumerate(Config.CLASS_NAMES):
        ax.plot(fpr_d[cls], tpr_d[cls], color=COLORS[i], lw=2,
                label=f"{cls}  (AUC={auc_d[cls]:.4f})")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"ROC Curves — All Classes\nMacro AUC={metrics['macro_auc']:.4f}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(output_dir, "roc_combined.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC combined      → {path1}")

    # ── Per-class ROC with EER ─────────────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))
    fig2.suptitle("Per-Class ROC with EER Point",
                  fontsize=14, fontweight="bold")
    axes2_flat = axes2.flatten()

    for i, cls in enumerate(Config.CLASS_NAMES):
        ax2 = axes2_flat[i]
        ax2.set_facecolor("#f8f9fa")
        ax2.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
        ax2.fill_between(fpr_d[cls], tpr_d[cls],
                         alpha=0.12, color=COLORS[i])
        ax2.plot(fpr_d[cls], tpr_d[cls], color=COLORS[i], lw=2.5,
                 label=f"AUC={auc_d[cls]:.4f}")

        # EER point
        eer = metrics["eer_dict"].get(cls)
        if eer is not None:
            fnr     = 1.0 - tpr_d[cls]
            diff    = np.abs(fpr_d[cls] - fnr)
            eer_idx = np.argmin(diff)
            ax2.plot(fpr_d[cls][eer_idx], tpr_d[cls][eer_idx],
                     "ro", ms=8, zorder=5,
                     label=f"EER={eer:.4f}")
            ax2.axvline(fpr_d[cls][eer_idx], color="red",
                        lw=1, linestyle=":", alpha=0.5)

        # Optimal point (closest to top-left)
        dist    = np.sqrt(fpr_d[cls]**2 + (1-tpr_d[cls])**2)
        opt_idx = np.argmin(dist)
        ax2.plot(fpr_d[cls][opt_idx], tpr_d[cls][opt_idx],
                 "b^", ms=7, zorder=5,
                 label=f"Optimal FPR={fpr_d[cls][opt_idx]:.3f}")

        ax2.set_title(cls, fontsize=11, fontweight="bold", color=COLORS[i])
        ax2.set_xlabel("FPR", fontsize=9)
        ax2.set_ylabel("TPR", fontsize=9)
        ax2.legend(loc="lower right", fontsize=8)
        ax2.set_xlim([-0.01,1.01]); ax2.set_ylim([-0.01,1.05])
        ax2.grid(True, alpha=0.3)

    axes2_flat[5].set_visible(False)
    plt.tight_layout()
    path2 = os.path.join(output_dir, "roc_per_class.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC per-class     → {path2}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Per-class F1, Accuracy, AUC, EER dashboard
# ─────────────────────────────────────────────────────────────────────────────
def plot_class_metrics_dashboard(metrics, output_dir):
    names  = Config.CLASS_NAMES
    f1s    = [metrics["per_class"][c]["f1"]       for c in names]
    accs   = [metrics["per_class"][c]["accuracy"] for c in names]
    aucs   = [metrics["auc_dict"][c]              for c in names]
    eers   = [metrics["eer_dict"].get(c, 0) or 0  for c in names]

    x = np.arange(len(names))
    COLORS = ["#3498db","#e74c3c","#2ecc71","#f39c12","#9b59b6"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"Classifier Test Set Dashboard\n"
        f"Accuracy={metrics['overall_acc']:.4f}  "
        f"Macro F1={metrics['macro_f1']:.4f}  "
        f"Macro AUC={metrics['macro_auc']:.4f}",
        fontsize=14, fontweight="bold"
    )

    for ax, vals, title, ylabel, add_mean in [
        (axes[0,0], f1s,  "F1 Score per Class",  "F1",       True),
        (axes[0,1], accs, "Accuracy per Class",  "Accuracy", True),
        (axes[1,0], aucs, "AUC-ROC per Class",   "AUC",      True),
        (axes[1,1], eers, "EER per Class\n(Lower = Better)", "EER", True),
    ]:
        bars = ax.bar(x, vals, color=COLORS, alpha=0.85,
                      edgecolor="white", width=0.6)
        if add_mean:
            mean_val = np.mean(vals)
            ax.axhline(mean_val, color="red", lw=1.5, linestyle="--",
                       label=f"Mean={mean_val:.4f}")
            ax.legend(fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.005,
                    f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10, rotation=15)
        ax.set_title(title,  fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim([0, min(max(vals)*1.3+0.05, 1.15)])
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "class_metrics_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Metrics dashboard → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Save CSVs
# ─────────────────────────────────────────────────────────────────────────────
def save_csvs(metrics, all_probs, all_preds, all_labels, output_dir):
    # Per-class summary CSV
    summary_csv = os.path.join(output_dir, "class_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class","accuracy","f1","precision",
                         "recall","auc","eer","support"])
        for cls in Config.CLASS_NAMES:
            m = metrics["per_class"][cls]
            writer.writerow([
                cls,
                m.get("accuracy",  ""),
                m.get("f1",        ""),
                m.get("precision", ""),
                m.get("recall",    ""),
                m.get("auc",       ""),
                m.get("eer",       ""),
                m.get("support",   ""),
            ])
        writer.writerow([
            "MACRO",
            metrics["overall_acc"],
            metrics["macro_f1"],
            metrics["macro_precision"],
            metrics["macro_recall"],
            metrics["macro_auc"],
            "",
            sum(metrics["per_class"][c]["support"] for c in Config.CLASS_NAMES),
        ])
    print(f"  Class summary CSV → {summary_csv}")

    # Per-sample predictions CSV
    pred_csv = os.path.join(output_dir, "predictions.csv")
    with open(pred_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["idx","true_label","true_class",
                  "pred_label","pred_class","correct"] + \
                 [f"prob_{c}" for c in Config.CLASS_NAMES]
        writer.writerow(header)
        for i in range(len(all_labels)):
            row = [
                i,
                all_labels[i],
                Config.CLASS_NAMES[all_labels[i]],
                all_preds[i],
                Config.CLASS_NAMES[all_preds[i]],
                int(all_preds[i] == all_labels[i]),
            ] + [f"{all_probs[i,c]:.6f}" for c in range(len(Config.CLASS_NAMES))]
            writer.writerow(row)
    print(f"  Predictions CSV   → {pred_csv}")

    # Top errors CSV
    wrong_idx  = np.where(all_preds != all_labels)[0]
    wrong_conf = all_probs[wrong_idx, all_preds[wrong_idx]]
    top_idx    = wrong_idx[np.argsort(-wrong_conf)[:TOP_K_ERRORS]]

    errors_csv = os.path.join(output_dir, "top_errors.csv")
    with open(errors_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank","idx","true_class","pred_class","confidence"])
        for rank, idx in enumerate(top_idx, 1):
            writer.writerow([
                rank, idx,
                Config.CLASS_NAMES[all_labels[idx]],
                Config.CLASS_NAMES[all_preds[idx]],
                f"{all_probs[idx, all_preds[idx]]:.4f}",
            ])
    print(f"  Top errors CSV    → {errors_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    if not os.path.exists(CLASSIFIER_WEIGHTS):
        raise FileNotFoundError(f"No classifier: {CLASSIFIER_WEIGHTS}")

    ckpt  = torch.load(CLASSIFIER_WEIGHTS, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model = ResNet50Classifier(pretrained=False).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_acc   = ckpt.get("val_accuracy", "?")
    print(f"Loaded : epoch {ckpt_epoch}  val_acc={ckpt_acc}")
    print(f"TTA    : {USE_TTA}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    _, _, test_loader = get_classifier_loaders()

    # ── Evaluate ──────────────────────────────────────────────────────────
    all_probs, all_preds, all_labels = run_evaluation(
        model, test_loader, device, use_tta=USE_TTA
    )

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = compute_metrics(all_probs, all_preds, all_labels)

    # ── Print results ─────────────────────────────────────────────────────
    print("\n── Overall Results ───────────────────────────────────────────")
    print(f"  Overall Accuracy : {metrics['overall_acc']*100:.2f}%")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision  : {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall     : {metrics['macro_recall']:.4f}")
    print(f"  Macro AUC        : {metrics['macro_auc']:.4f}")

    print("\n── Per-Class Results ─────────────────────────────────────────")
    print(f"  {'Class':20s} {'Acc':>7} {'F1':>7} {'Prec':>7} "
          f"{'Rec':>7} {'AUC':>7} {'EER':>8} {'N':>6}")
    print("  " + "─" * 72)
    for cls in Config.CLASS_NAMES:
        m = metrics["per_class"][cls]
        eer_str = f"{m['eer']:.4f}" if m.get("eer") else "  N/A "
        print(f"  {cls:20s} "
              f"{m['accuracy']:>7.4f} {m['f1']:>7.4f} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m.get('auc',0):>7.4f} {eer_str:>8} "
              f"{m['support']:>6}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n── Generating plots ──────────────────────────────────────────")
    plot_confusion_matrix(all_preds, all_labels, OUTPUT_DIR)
    plot_roc_curves(metrics, all_probs, all_labels, OUTPUT_DIR)
    plot_class_metrics_dashboard(metrics, OUTPUT_DIR)

    # ── CSVs ──────────────────────────────────────────────────────────────
    save_csvs(metrics, all_probs, all_preds, all_labels, OUTPUT_DIR)

    # ── Save JSON ─────────────────────────────────────────────────────────
    results = {
        "checkpoint_epoch": ckpt_epoch,
        "val_accuracy":     ckpt_acc,
        "overall":          {
            "accuracy":   metrics["overall_acc"],
            "macro_f1":   metrics["macro_f1"],
            "macro_prec": metrics["macro_precision"],
            "macro_rec":  metrics["macro_recall"],
            "macro_auc":  metrics["macro_auc"],
        },
        "per_class": metrics["per_class"],
    }
    json_path = os.path.join(OUTPUT_DIR, "test_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n── Files Generated ───────────────────────────────────────────")
    print(f"  confusion_matrix.png       ← raw + normalised heatmap")
    print(f"  roc_combined.png           ← all classes on one plot")
    print(f"  roc_per_class.png          ← individual ROC + EER per class")
    print(f"  class_metrics_dashboard.png← F1/Acc/AUC/EER bar charts")
    print(f"  class_summary.csv          ← all metrics per class")
    print(f"  predictions.csv            ← per-sample results")
    print(f"  top_errors.csv             ← worst {TOP_K_ERRORS} mistakes")
    print(f"  test_results.json          ← all numbers")
    print(f"\n  Output folder: {os.path.abspath(OUTPUT_DIR)}/")

    print(f"\n{'='*54}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*54}")
    print(f"  Accuracy  : {metrics['overall_acc']*100:.2f}%")
    print(f"  Macro F1  : {metrics['macro_f1']:.4f}")
    print(f"  Macro AUC : {metrics['macro_auc']:.4f}")
    print(f"{'='*54}\n")


if __name__ == "__main__":
    main()
