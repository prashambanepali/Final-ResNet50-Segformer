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
from config2 import Config


# ─────────────────────────────────────────────────────────────────────────
# 1.  ResNet50 Classifier  (9-channel input)
# ─────────────────────────────────────────────────────────────────────────
class ResNet50Classifier(nn.Module):
    """
    ResNet50 with first conv replaced to accept 9-channel input.
    Output: (B, NUM_CLASSES) logits.
    """
    def __init__(self, num_classes: int = Config.NUM_CLASSES,
                 pretrained: bool = True):
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Replace conv1: 3ch → 9ch
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            Config.IN_CHANNELS, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # Initialise new conv: copy ImageNet weights into first 3 channels,
        # replicate across remaining 6 channels and scale down
        with torch.no_grad():
            w = old_conv.weight.data          # (64, 3, 7, 7)
            new_w = backbone.conv1.weight.data
            for i in range(Config.IN_CHANNELS):
                new_w[:, i, :, :] = w[:, i % 3, :, :] / (Config.IN_CHANNELS / 3)
            backbone.conv1.weight.data = new_w

        # Replace final FC
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(backbone.fc.in_features, num_classes),
        )

        self.model = backbone

    def forward(self, x):
        return self.model(x)    # (B, num_classes)

    def predict(self, x):
        """Returns predicted class indices (B,)."""
        return self.forward(x).argmax(dim=1)


# ─────────────────────────────────────────────────────────────────────────
# 2.  Input projection  9ch → 3ch for SegFormer
# ─────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────
# 3.  SegFormer-B2 Localizer
# ─────────────────────────────────────────────────────────────────────────
class ConvBNReLU(nn.Module):
    def __init__(self, ic, oc, k=3, p=1):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(ic, oc, k, padding=p, bias=False),
            nn.BatchNorm2d(oc), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)


class ForgeryLocalizer(nn.Module):
    """
    9ch → InputProjection(3ch) → SegFormer-B2 →
    class-conditioned decoder → binary mask (B, 1, H, W)
    """
    ENC_CH = [64, 128, 320, 512]

    def __init__(self):
        super().__init__()
        self.input_proj = InputProjection()
        self.encoder    = SegformerModel.from_pretrained(
            "nvidia/mit-b2", output_hidden_states=True)

        D = 256
        self.proj = nn.ModuleList([nn.Conv2d(c, D, 1) for c in self.ENC_CH])

        # +1 for authentic token (index = NUM_CLASSES - 1 + 1)
        self.class_embed = nn.Embedding(Config.NUM_CLASSES + 1, D)

        fused = D * 4 + D   # 1280
        self.dec4 = ConvBNReLU(fused,    256)
        self.dec3 = ConvBNReLU(256 + D,  128)
        self.dec2 = ConvBNReLU(128 + D,   64)
        self.dec1 = ConvBNReLU(64  + D,   32)
        self.head = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x9, class_idx):
        B = x9.size(0)
        H = W = Config.IMAGE_SIZE

        x3     = self.input_proj(x9)
        hidden = self.encoder(pixel_values=x3).hidden_states

        th, tw = H // 4, W // 4
        feats  = []
        for h, proj in zip(hidden, self.proj):
            s  = int(h.shape[1] ** 0.5)
            hs = h.permute(0, 2, 1).reshape(B, h.shape[2], s, s)
            feats.append(F.interpolate(proj(hs), (th, tw),
                                       mode="bilinear", align_corners=False))

        ce  = self.class_embed(class_idx)[:, :, None, None].expand(B, -1, th, tw)
        x   = self.drop(torch.cat([*feats, ce], dim=1))

        x = self.dec4(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, feats[1]], 1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, feats[0]], 1))
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x,
              F.interpolate(feats[0], (H, W),
                            mode="bilinear", align_corners=False)], 1))
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────
# 4.  Full localizer pipeline  (frozen classifier + SegFormer)
# ─────────────────────────────────────────────────────────────────────────
class LocalizerPipeline(nn.Module):
    def __init__(self, classifier_weights: str):
        super().__init__()

        # Load frozen classifier
        clf = ResNet50Classifier(pretrained=False)
        ckpt = torch.load(classifier_weights, map_location="cpu")
        state = ckpt.get("model_state_dict",
                ckpt.get("state_dict", ckpt))
        clf.load_state_dict(state, strict=False)
        for p in clf.parameters():
            p.requires_grad = False
        clf.eval()
        self.classifier = clf

        self.localizer = ForgeryLocalizer()

    @torch.no_grad()
    def _get_class(self, x9):
        return self.classifier.predict(x9)

    def forward(self, x9, class_idx=None):
        if class_idx is None:
            class_idx = self._get_class(x9)
        return self.localizer(x9, class_idx)
