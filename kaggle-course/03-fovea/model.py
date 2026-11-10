"""Center-point estimators. V12 architecture is preserved from the original script."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


def expectation(logits, temperature=1.):
    batch, _, h, w = logits.shape
    probability = (logits.float().reshape(batch, -1) / temperature).softmax(1)
    y, x = torch.meshgrid(torch.linspace(0, 1, h, device=logits.device),
                          torch.linspace(0, 1, w, device=logits.device), indexing="ij")
    return torch.stack(((probability * x.flatten()).sum(1),
                        (probability * y.flatten()).sum(1)), dim=1)


class ConvBlock(nn.Sequential):
    def __init__(self, ci, co):
        super().__init__(nn.Conv2d(ci, co, 3, padding=1, bias=False),
                         nn.GroupNorm(8, co), nn.SiLU(inplace=True),
                         nn.Conv2d(co, co, 3, padding=1, bias=False),
                         nn.GroupNorm(8, co), nn.SiLU(inplace=True))


class V12UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e0 = ConvBlock(6, 16)
        self.e1 = ConvBlock(16, 32)
        self.e2 = ConvBlock(32, 64)
        self.e3 = ConvBlock(64, 96)
        self.mid = ConvBlock(96, 128)
        self.d3 = ConvBlock(128 + 96, 96)
        self.d2 = ConvBlock(96 + 64, 64)
        self.heat = nn.Conv2d(64, 1, 1)
        self.global_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                         nn.Linear(128, 64), nn.SiLU(),
                                         nn.Linear(64, 2), nn.Sigmoid())

    def forward(self, x):
        a = self.e0(x)
        b = self.e1(F.avg_pool2d(a, 2))
        c = self.e2(F.avg_pool2d(b, 2))
        d = self.e3(F.avg_pool2d(c, 2))
        m = self.mid(F.avg_pool2d(d, 2))
        z = self.d3(torch.cat([resize(m, d), d], dim=1))
        z = self.d2(torch.cat([resize(z, c), c], dim=1))
        heat = self.heat(z)
        glob = self.global_head(m)
        return .8 * expectation(heat) + .2 * glob, heat, glob


def resize(x, target):
    return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)


class ResNetHeatmap(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        net = models.resnet18(weights="DEFAULT" if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        self.decode = nn.ModuleList([ConvBlock(512 + 256, 128),
                                     ConvBlock(128 + 128, 64), ConvBlock(64 + 64, 32)])
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        features = [self.stem(x)]
        for stage in self.stages:
            features.append(stage(features[-1]))
        z = features.pop()
        for block, skip in zip(self.decode, reversed(features[1:])):
            z = block(torch.cat([resize(z, skip), skip], 1))
        heat = self.head(z)
        return expectation(heat, .1), heat, None


class LocalRegressor(nn.Module):
    def __init__(self, deep, pretrained):
        super().__init__()
        model = models.get_model("resnet34" if deep else "resnet18",
                                 weights="DEFAULT" if pretrained else None)
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 2), nn.Sigmoid())
        self.net = model

    def forward(self, x):
        return self.net(x), None, None


def objective(outputs, truth, version):
    xy, heat, glob = outputs
    coord = F.smooth_l1_loss(xy, truth, beta=.02)
    if heat is None:
        return coord
    h, w = heat.shape[-2:]
    y, x = torch.meshgrid(torch.linspace(0, 1, h, device=heat.device),
                          torch.linspace(0, 1, w, device=heat.device), indexing="ij")
    distance = (x - truth[:, 0, None, None])**2 + (y - truth[:, 1, None, None])**2
    sigma = (2.5 if version != "v8" else 3.5) / w
    gaussian = -distance.flatten(1) / (2 * sigma**2)
    if glob is None:
        target = gaussian.exp()
        heat_loss = ((1 + 30 * target) * (heat.float().flatten(1) - target).square()).mean()
        return heat_loss + 10 * F.smooth_l1_loss(xy, truth, beta=.05)
    heat_loss = -(gaussian.softmax(1) * heat.float().flatten(1).log_softmax(1)).sum(1).mean()
    return heat_loss + 12 * coord + 3 * F.smooth_l1_loss(glob, truth, beta=.02)
