"""
infer.py
========
Run inference on a single external image.
Outputs:
  1. Predicted class (from ResNet50 classifier)
  2. Localization mask (from SegFormer localizer)
  3. GradCAM heatmap (from ResNet50 classifier)
  4. Combined visualization saved as PNG

Usage:
    python infer.py --image path/to/image.jpg
    python infer.py --image path/to/image.jpg --save results/output.png
    python infer.py --image path/to/image.jpg --threshold 0.4
"""

import os
import cv2
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from config import Config
from models import LocalizerPipeline, ResNet50Classifier
from precompute_maps import compute_ela_3ch, compute_noise_3ch

CLASSIFIER_WEIGHTS = os.path.join(Config.SAVE_DIR, "best_classifier.pth")
LOCALIZER_WEIGHTS  = os.path.join(Config.SAVE_DIR, "best_localizer.pth")

# Normalisation (same as training)
_M = [0.485, 0.456, 0.406] * 3
_S = [0.229, 0.224, 0.225] * 3
MEAN_T = torch.tensor(_M).view(9, 1, 1)
STD_T  = torch.tensor(_S).view(9, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    GradCAM for ResNet50.
    Hooks into the last conv layer (layer4) to get gradients and activations.
    """
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hooks      = []
        self._register_hooks()

    def _register_hooks(self):
        # Target: last conv block of ResNet50
        target_layer = self.model.model.layer4[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(
            target_layer.register_forward_hook(forward_hook))
        self._hooks.append(
            target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def generate(self, x, class_idx=None):
        """
        x         : (1, 9, H, W) tensor
        class_idx : int or None (uses predicted class)
        Returns   : heatmap (H, W) numpy array in [0, 1]
        """
        self.model.eval()
        x = x.requires_grad_(True)

        logits = self.model(x)                        # (1, num_classes)
        probs  = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backprop w.r.t. predicted class score
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Grad-weighted activations
        grads = self.gradients                         # (1, C, h, w)
        acts  = self.activations                       # (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True) # (1, C, 1, 1)

        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx, probs.detach().cpu().numpy()[0]


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(image_path):
    """
    Load image → compute ELA + Noise → stack 9ch tensor.
    Returns:
        tensor_9ch : (1, 9, 224, 224) normalised
        rgb_orig   : (224, 224, 3) uint8 numpy for display
    """
    img_pil = Image.open(image_path).convert("RGB")

    # Resize for display
    img_resized = img_pil.resize(
        (Config.IMAGE_SIZE, Config.IMAGE_SIZE), Image.BILINEAR)
    rgb_orig = np.array(img_resized)

    # Compute maps
    ela_pil   = compute_ela_3ch(img_pil)
    noise_pil = compute_noise_3ch(img_pil)

    size = (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    rgb_t   = TF.to_tensor(TF.resize(img_pil,   size, InterpolationMode.BILINEAR))
    ela_t   = TF.to_tensor(TF.resize(ela_pil,   size, InterpolationMode.BILINEAR))
    noise_t = TF.to_tensor(TF.resize(noise_pil, size, InterpolationMode.BILINEAR))

    x = torch.cat([rgb_t, ela_t, noise_t], dim=0)   # (9, H, W)
    x = (x - MEAN_T) / STD_T                        # normalise
    x = x.unsqueeze(0)                              # (1, 9, H, W)

    return x, rgb_orig


# ─────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ─────────────────────────────────────────────────────────────────────────────
def apply_colormap(heatmap_np, colormap=cv2.COLORMAP_JET):
    """Convert [0,1] float heatmap → BGR uint8 colormap image."""
    h = (heatmap_np * 255).astype(np.uint8)
    return cv2.applyColorMap(h, colormap)


def overlay_heatmap(rgb_np, heatmap_np, alpha=0.45):
    """Blend RGB image with heatmap colormap."""
    heatmap_bgr = apply_colormap(heatmap_np)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return (rgb_np * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)


def overlay_mask(rgb_np, mask_np, color=(255, 50, 50), alpha=0.45):
    """Overlay binary mask on RGB image in red."""
    overlay = rgb_np.copy().astype(np.float32)
    mask_3ch = np.stack([mask_np]*3, axis=2)
    colored  = np.zeros_like(overlay)
    colored[:,:,0] = color[0]
    colored[:,:,1] = color[1]
    colored[:,:,2] = color[2]
    overlay = np.where(mask_3ch > 0,
                       overlay * (1-alpha) + colored * alpha,
                       overlay)
    return overlay.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Main inference
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(image_path, threshold=0.5, save_path=None, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Image      : {image_path}")
    print(f"Threshold  : {threshold}\n")

    # ── Load models ───────────────────────────────────────────────────────
    if not os.path.exists(CLASSIFIER_WEIGHTS):
        raise FileNotFoundError(f"Classifier not found: {CLASSIFIER_WEIGHTS}")
    if not os.path.exists(LOCALIZER_WEIGHTS):
        raise FileNotFoundError(f"Localizer not found: {LOCALIZER_WEIGHTS}")

    # Full pipeline (classifier + localizer)
    pipeline = LocalizerPipeline(CLASSIFIER_WEIGHTS).to(device)
    ckpt     = torch.load(LOCALIZER_WEIGHTS, map_location=device)
    pipeline.load_state_dict(ckpt["model_state_dict"])
    pipeline.eval()

    # Standalone classifier for GradCAM
    clf_ckpt  = torch.load(CLASSIFIER_WEIGHTS, map_location=device)
    clf_state = clf_ckpt.get("model_state_dict",
                clf_ckpt.get("state_dict", clf_ckpt))
    classifier = ResNet50Classifier(pretrained=False).to(device)
    classifier.load_state_dict(clf_state, strict=False)
    classifier.eval()

    # ── Preprocess ────────────────────────────────────────────────────────
    print("Computing ELA and Noise maps ...")
    x, rgb_orig = preprocess_image(image_path)
    x_dev = x.to(device)

    # ── Classification ────────────────────────────────────────────────────
    with torch.no_grad():
        logits     = classifier(x_dev)
        probs      = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx   = int(probs.argmax())
        pred_class = Config.CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

    print(f"Predicted  : {pred_class.upper()}  (confidence={confidence:.4f})")
    print("Class probabilities:")
    for i, (name, p) in enumerate(zip(Config.CLASS_NAMES, probs)):
        bar = "█" * int(p * 30)
        marker = " ←" if i == pred_idx else ""
        print(f"  {name:20s}: {p:.4f}  {bar}{marker}")

    # ── GradCAM ───────────────────────────────────────────────────────────
    print("\nGenerating GradCAM ...")
    gradcam    = GradCAM(classifier)
    cam, _, _  = gradcam.generate(x_dev.clone(), class_idx=pred_idx)
    gradcam.remove_hooks()

    # ── Localization mask ─────────────────────────────────────────────────
    print("Generating localization mask ...")
    with torch.no_grad():
        class_tensor = torch.tensor([pred_idx], dtype=torch.long).to(device)
        logits_loc   = pipeline(x_dev, class_tensor)
        prob_map     = torch.sigmoid(logits_loc.float()).squeeze().cpu().numpy()

    binary_mask = (prob_map >= threshold).astype(np.uint8)
    tampered_pct = binary_mask.mean() * 100

    print(f"\nTampered region: {tampered_pct:.1f}% of image pixels")
    print(f"Mask threshold : {threshold}")

    # ── Build visualization ───────────────────────────────────────────────
    print("\nBuilding visualization ...")

    gradcam_overlay = overlay_heatmap(rgb_orig, cam, alpha=0.5)
    mask_overlay    = overlay_mask(rgb_orig, binary_mask)
    prob_colored    = apply_colormap(prob_map)
    prob_rgb        = cv2.cvtColor(prob_colored, cv2.COLOR_BGR2RGB)

    # ELA channel for display (channel 3 = first ELA)
    ela_display = (x[0, 3].numpy() * 0.229 + 0.485)
    ela_display = np.clip(ela_display, 0, 1)

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("#1a1a2e")

    is_authentic = (pred_class == "authentic")
    status_color = "#00ff88" if is_authentic else "#ff4444"
    status_text  = "✓ AUTHENTIC" if is_authentic else f"⚠ TAMPERED — {pred_class.upper()}"

    fig.suptitle(
        f"{status_text}   |   Confidence: {confidence:.1%}   |   "
        f"Tampered pixels: {tampered_pct:.1f}%",
        fontsize=16, fontweight="bold",
        color=status_color, y=0.98
    )

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.35, wspace=0.25,
                           left=0.04, right=0.96,
                           top=0.92, bottom=0.08)

    panels = [
        (gs[0, 0], rgb_orig,         "Original Image",         None),
        (gs[0, 1], gradcam_overlay,   "GradCAM Heatmap",        None),
        (gs[0, 2], mask_overlay,      f"Localization Mask\n(threshold={threshold})", None),
        (gs[0, 3], prob_rgb,          "Tampering Probability\nHeatmap", "jet"),
        (gs[1, 0], binary_mask*255,   "Binary Mask",            "gray"),
        (gs[1, 1], (cam*255).astype(np.uint8), "GradCAM Raw",  "hot"),
        (gs[1, 2], (ela_display*255).astype(np.uint8), "ELA Map (q70)", "gray"),
    ]

    for spec, img, title, cmap in panels:
        ax = fig.add_subplot(spec)
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10, fontweight="bold",
                     color="white", pad=6)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    # ── Probability bar chart ─────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, 3])
    colors = ["#00ff88" if i == pred_idx else "#4466aa"
              for i in range(len(Config.CLASS_NAMES))]
    bars = ax_bar.barh(Config.CLASS_NAMES, probs * 100,
                       color=colors, edgecolor="#222244", height=0.6)
    ax_bar.set_xlim([0, 100])
    ax_bar.set_xlabel("Confidence (%)", color="white", fontsize=9)
    ax_bar.set_title("Class Probabilities", fontsize=10,
                     fontweight="bold", color="white", pad=6)
    ax_bar.tick_params(colors="white", labelsize=9)
    ax_bar.set_facecolor("#1a1a2e")
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#444466")
    for bar, p in zip(bars, probs):
        ax_bar.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{p:.1%}", va="center", color="white", fontsize=8)

    # ── Save / show ───────────────────────────────────────────────────────
    if save_path is None:
        base     = os.path.splitext(os.path.basename(image_path))[0]
        save_dir = "inference_results"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{base}_result.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"\n✅ Result saved → {save_path}")
    print("\n── Summary ───────────────────────────────────────────────────")
    print(f"  Predicted class : {pred_class.upper()}")
    print(f"  Confidence      : {confidence:.4f}  ({confidence:.1%})")
    print(f"  Tampered pixels : {tampered_pct:.2f}%")
    print(f"  EER threshold   : {threshold}")
    print(f"  Result image    : {save_path}")

    return {
        "predicted_class": pred_class,
        "confidence":       confidence,
        "probabilities":    {n: float(p) for n, p in
                             zip(Config.CLASS_NAMES, probs)},
        "tampered_pixels_pct": tampered_pct,
        "binary_mask":      binary_mask,
        "prob_map":         prob_map,
        "gradcam":          cam,
        "save_path":        save_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forgery detection inference — class + mask + GradCAM"
    )
    parser.add_argument("--image",     required=True,
                        help="Path to input image (.jpg / .png)")
    parser.add_argument("--save",      default=None,
                        help="Path to save output PNG (default: inference_results/<name>_result.png)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Mask binarization threshold (default: 0.5)")
    parser.add_argument("--device",    default="cuda",
                        help="cuda or cpu (default: cuda)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    run_inference(
        image_path = args.image,
        threshold  = args.threshold,
        save_path  = args.save,
        device     = args.device,
    )
