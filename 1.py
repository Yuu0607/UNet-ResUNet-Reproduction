import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "Pytorch-UNet-3.0"
WEIGHT = ROOT / "unet_carvana_scale0.5_epoch2.pth"
INPUT = ROOT / "carvana_sample.jpg"
MASK_OUT = ROOT / "carvana_sample_mask_255.png"
PREVIEW_OUT = ROOT / "carvana_unet_result.jpg"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
(ROOT / ".matplotlib").mkdir(exist_ok=True)
sys.path.insert(0, str(REPO))

from predict import predict_img  # noqa: E402
from unet import UNet  # noqa: E402


def make_preview(image, mask, output):
    panel_w, panel_h, top = 640, 427, 42
    image = image.resize((panel_w, panel_h), Image.BICUBIC)
    mask_small = mask.resize((panel_w, panel_h), Image.NEAREST)
    mask_rgb = mask_small.convert("RGB")
    red = Image.new("RGB", image.size, (255, 45, 45))
    overlay = Image.composite(Image.blend(image, red, 0.45), image, mask_small)

    canvas = Image.new("RGB", (panel_w * 3, panel_h + top), "white")
    for index, (panel, label) in enumerate([
        (image, "Input image"),
        (mask_rgb, "UNet mask"),
        (overlay, "Segmentation overlay"),
    ]):
        canvas.paste(panel, (index * panel_w, top))
        ImageDraw.Draw(canvas).text((index * panel_w + 12, 12), label, fill="black")
    canvas.save(output, quality=95)


def main():
    if not WEIGHT.is_file() or not INPUT.is_file():
        raise FileNotFoundError("Weight or sample image is missing")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(n_channels=3, n_classes=2, bilinear=False).to(device).eval()
    state_dict = torch.load(WEIGHT, map_location=device)
    net.load_state_dict(state_dict)

    image = Image.open(INPUT).convert("RGB")
    start = time.perf_counter()
    one_hot_mask = predict_img(net, image, device, scale_factor=0.5, out_threshold=0.5)
    elapsed = time.perf_counter() - start

    foreground = one_hot_mask[1].astype(bool)
    binary_mask = Image.fromarray((foreground * 255).astype(np.uint8), mode="L")
    binary_mask.save(MASK_OUT)
    make_preview(image, binary_mask, PREVIEW_OUT)

    print("Repository: https://github.com/milesial/Pytorch-UNet")
    print("Model: U-Net (Carvana pretrained)")
    print("Device:", device)
    print("Input image:", INPUT)
    print("Image size:", image.size)
    print("Masked pixels:", int(foreground.sum()))
    print("Masked ratio: {:.2%}".format(float(foreground.mean())))
    print("Inference time: {:.2f} s".format(elapsed))
    print("Mask:", MASK_OUT)
    print("Preview:", PREVIEW_OUT)


if __name__ == "__main__":
    main()

