"""Residual U-Net built on the downloaded Pytorch-UNet model."""

import torch
import torch.nn as nn

from unet import UNet
from unet.unet_parts import DoubleConv


class ResDoubleConv(nn.Module):
    """Double convolution with a learnable residual projection."""

    def __init__(self, base: DoubleConv, residual_scale: float = 0.0):
        super().__init__()
        self.double_conv = base.double_conv
        in_channels = base.double_conv[0].in_channels
        out_channels = base.double_conv[3].out_channels
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x):
        output = self.double_conv(x)
        return output + self.residual_scale * self.residual(x)


def _replace_double_convs(module, residual_scale):
    for name, child in module.named_children():
        if isinstance(child, DoubleConv):
            setattr(module, name, ResDoubleConv(child, residual_scale=residual_scale))
        else:
            _replace_double_convs(child, residual_scale)


def create_resunet(n_channels=3, n_classes=2, bilinear=False, residual_scale=0.0):
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    _replace_double_convs(model, residual_scale=residual_scale)
    return model
