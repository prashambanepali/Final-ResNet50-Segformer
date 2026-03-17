"""
models.py
=========
1. ResNet50Classifier   — 9ch input, 5-class output
2. ForgeryLocalizer     — SegFormer-B2, 9ch input, binary mask output
3. LocalizerPipeline    — frozen classifier + localizer combined
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel
import torchvision.models as models
from config import Config


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ResNet50 Classifier  (9-channel input)
# ─────────────────────────────────────────────────────────────────────────────
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super().__init__()
        weights  = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        old_conv       = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            Config.IN_CHANNELS, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            w     = old_conv.weight.data
            new_w = backbone.conv1.weight.data
            for i in range(Config.IN_CHANNELS):
                new_w[:, i, :, :] = w[:, i % 3, :, :] / (Config.IN_CHANNELS / 3)

        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(backbone.fc.in_features, num_classes),
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        return self.forward(x).argmax(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Input projection  9ch → 3ch for SegFormer
# ─────────────────────────────────────────────────────────────────────────────
class InputProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(Config.IN_CHANNELS, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1, bias=False),
            nn.BatchNorm2d(3),
        )
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.proj(x) + self.skip_scale * x[:, :3]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SegFormer-B2 Localizer
# ─────────────────────────────────────────────────────────────────────────────
class ConvBNReLU(nn.Module):
    def __init__(self, ic, oc, k=3, p=1):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(ic, oc, k, padding=p, bias=False),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.b(x)


class ForgeryLocalizer(nn.Module):
    """
    9ch → InputProjection(3ch) → SegFormer-B2 →
    class-conditioned decoder → binary mask (B, 1, H, W)

    All skip connections are resized to match decoder spatial dims
    to avoid any size mismatch regardless of transformers version.
    """
    ENC_CH = [64, 128, 320, 512]

    def __init__(self):
        super().__init__()
        self.input_proj = InputProjection()
        self.encoder    = SegformerModel.from_pretrained(
            "nvidia/mit-b2",
            output_hidden_states=True,
        )

        D = 256
        self.proj = nn.ModuleList([nn.Conv2d(c, D, 1) for c in self.ENC_CH])

        self.class_embed = nn.Embedding(Config.NUM_CLASSES + 1, D)

        fused     = D * 4 + D    # 1280
        self.dec4 = ConvBNReLU(fused,    256)
        self.dec3 = ConvBNReLU(256 + D,  128)
        self.dec2 = ConvBNReLU(128 + D,   64)
        self.dec1 = ConvBNReLU(64  + D,   32)
        # No Sigmoid here — BCEWithLogitsLoss applies it internally (AMP safe)
        # At inference, apply torch.sigmoid(output) to get mask probabilities
        self.head = nn.Sequential(nn.Conv2d(32, 1, 1))
        self.drop = nn.Dropout2d(0.1)

    def _to_spatial(self, h, B):
        """Convert hidden state to (B, C, H, W) regardless of format."""
        if h.dim() == 3:
            # Old transformers: (B, seq_len, C)
            seq_len = h.shape[1]
            c       = h.shape[2]
            side    = int(seq_len ** 0.5)
            return h.permute(0, 2, 1).reshape(B, c, side, side)
        else:
            # New transformers: already (B, C, H, W)
            return h

    def forward(self, x9, class_idx):
        B = x9.size(0)
        H = W = Config.IMAGE_SIZE

        # Project 9ch → 3ch
        x3     = self.input_proj(x9)

        # Encoder
        hidden = self.encoder(pixel_values=x3).hidden_states

        # Project all stages to D=256 and upsample to same size (H/4, W/4)
        th, tw = H // 4, W // 4
        feats  = []
        for h, proj in zip(hidden, self.proj):
            h_spatial = self._to_spatial(h, B)          # (B, C, h, w)
            h_proj    = proj(h_spatial)                  # (B, D, h, w)
            # Always interpolate to same target size — fixes all size mismatches
            h_up = F.interpolate(h_proj, size=(th, tw),
                                 mode="bilinear", align_corners=False)
            feats.append(h_up)                           # all (B, D, H/4, W/4)

        # Class conditioning
        ce = self.class_embed(class_idx)[:, :, None, None].expand(B, -1, th, tw)

        # Fuse all 4 stages + class embed
        x = self.drop(torch.cat([*feats, ce], dim=1))   # (B, 1280, H/4, W/4)

        # ── Decode — all skips resized to match x at each stage ───────────
        # Stage 4 → H/2
        x = self.dec4(x)                                 # (B, 256, H/4, W/4)
        x = F.interpolate(x, scale_factor=2,
                          mode="bilinear", align_corners=False)  # (B,256,H/2,W/2)

        # Stage 3 — resize feats[1] to match x
        skip3 = F.interpolate(feats[1], size=x.shape[2:],
                              mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, skip3], dim=1))     # (B, 128, H/2, W/2)
        x = F.interpolate(x, scale_factor=2,
                          mode="bilinear", align_corners=False)  # (B,128,H,W)

        # Stage 2 — resize feats[0] to match x
        skip2 = F.interpolate(feats[0], size=x.shape[2:],
                              mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, skip2], dim=1))     # (B, 64, H, W)
        x = F.interpolate(x, size=(H, W),
                          mode="bilinear", align_corners=False)

        # Stage 1 — resize feats[0] to final H,W
        skip1 = F.interpolate(feats[0], size=(H, W),
                              mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, skip1], dim=1))     # (B, 32, H, W)

        return self.head(x)                              # (B, 1, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────
class LocalizerPipeline(nn.Module):
    def __init__(self, classifier_weights):
        super().__init__()
        clf  = ResNet50Classifier(pretrained=False)
        ckpt = torch.load(classifier_weights, map_location="cpu")
        state = ckpt.get("model_state_dict",
                ckpt.get("state_dict", ckpt))
        clf.load_state_dict(state, strict=False)
        for p in clf.parameters():
            p.requires_grad = False
        clf.eval()
        self.classifier = clf
        self.localizer  = ForgeryLocalizer()

    @torch.no_grad()
    def _get_class(self, x9):
        return self.classifier.predict(x9)

    def forward(self, x9, class_idx=None):
        if class_idx is None:
            class_idx = self._get_class(x9)
        return self.localizer(x9, class_idx)
