"""U-Net building blocks and experimental objectives."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, ci, co, residual):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False),
                                  nn.GroupNorm(8, co), nn.SiLU(),
                                  nn.Conv2d(co, co, 3, padding=1, bias=False),
                                  nn.GroupNorm(8, co))
        self.skip = nn.Conv2d(ci, co, 1) if residual else None

    def forward(self, x):
        y = self.conv(x)
        return F.silu(y if self.skip is None else y + self.skip(x))


class UNet(nn.Module):
    def __init__(self, channels, residual=False, auxiliary=False, context=False):
        super().__init__()
        self.down = nn.ModuleList([Block(channels, 24, residual), Block(24, 48, residual),
                                   Block(48, 96, residual), Block(96, 128, residual)])
        self.up = nn.ModuleList([Block(128+96, 96, residual), Block(96+48, 48, residual),
                                 Block(48+24, 24, residual)])
        self.head = nn.Conv2d(24, 1, 1)
        self.aux = nn.Conv2d(48, 1, 1) if auxiliary else None
        self.context = nn.Conv2d(128, 128, 1) if context else None

    def forward(self, x):
        skips = []
        for block in self.down[:-1]:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.down[-1](x)
        if self.context is not None:
            x = x + self.context(F.adaptive_avg_pool2d(x, 1))
        aux = None
        for i, (block, skip) in enumerate(zip(self.up, reversed(skips))):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], 1))
            if i == 1 and self.aux is not None:
                aux = self.aux(x)
        return self.head(x), aux


def loss_value(outputs, target, loss, auxiliary_core):
    logits, aux = outputs
    p = logits.float().sigmoid()
    dimensions = (1, 2, 3)
    tp = (p * target).sum(dimensions)
    fp = (p * (1 - target)).sum(dimensions)
    fn = ((1 - p) * target).sum(dimensions)
    alpha, beta = (.4, .6) if loss == "tversky" else (.5, .5)
    overlap = ((tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)).mean()
    bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    if loss == "focal":
        pt = torch.where(target > .5, p, 1 - p)
        bce = bce * (1 - pt)
    total = bce.mean() + 1 - overlap
    if aux is not None:
        aux_target = target
        if auxiliary_core:
            aux_target = 1 - F.max_pool2d(1 - target, 3, stride=1, padding=1)
        aux = F.interpolate(aux, size=target.shape[-2:], mode="bilinear", align_corners=False)
        total += .25 * F.binary_cross_entropy_with_logits(aux.float(), aux_target)
    return total
