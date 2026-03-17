import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        # Apply sigmoid here since model no longer has sigmoid in head
        pred  = torch.sigmoid(logits).float()
        t     = target.float()
        p, t  = pred.view(-1), t.view(-1)
        inter = (p * t).sum()
        return 1 - (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                          dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("sx", sx)
        self.register_buffer("sy", sx.transpose(2,3))

    def _edges(self, x):
        x  = x.float()
        gx = F.conv2d(x, self.sx, padding=1)
        gy = F.conv2d(x, self.sy, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def forward(self, logits, target):
        pred = torch.sigmoid(logits).float()
        return F.l1_loss(self._edges(pred), self._edges(target.float()))


class CombinedSegLoss(nn.Module):
    """
    BCEWithLogitsLoss + Dice + Edge.
    Takes RAW LOGITS (no sigmoid) — AMP safe.
    Sigmoid is applied internally in Dice and Edge.
    """
    def __init__(self, bce_w=0.4, dice_w=0.4, edge_w=0.2):
        super().__init__()
        self.bce_w  = bce_w
        self.dice_w = dice_w
        self.edge_w = edge_w
        # BCEWithLogitsLoss = sigmoid + BCE in one op — AMP safe
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.edge = EdgeLoss()

    def forward(self, logits, target):
        target = target.float()
        b = self.bce(logits,  target)
        d = self.dice(logits, target)
        e = self.edge(logits, target)
        total = self.bce_w*b + self.dice_w*d + self.edge_w*e
        return total, {"bce": b.item(), "dice": d.item(), "edge": e.item()}
