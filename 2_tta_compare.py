import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "Pytorch-UNet-3.0"
WEIGHT = ROOT / "unet_carvana_scale0.5_epoch2.pth"
INPUT = ROOT / "carvana_train_image.jpg"
GROUND_TRUTH = ROOT / "carvana_train_gt_mask.png"

BASELINE_MASK = ROOT / "tta_baseline_mask.png"
TTA_MASK = ROOT / "tta_result_mask.png"
GT_MASK = ROOT / "tta_ground_truth_mask.png"
COMPARISON = ROOT / "tta_comparison.jpg"
METRICS = ROOT / "tta_metrics.json"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
(ROOT / ".matplotlib").mkdir(exist_ok=True)
sys.path.insert(0, str(REPO))

from unet import UNet  # noqa: E402


def preprocess(image, scale=0.5):
    resized = image.resize(
        (int(image.width * scale), int(image.height * scale)),
        Image.BICUBIC,
    )
    array = np.asarray(resized).transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def predict_probabilities(net, image, device):
    tensor = preprocess(image).to(device)
    logits = net(tensor)
    probabilities = torch.softmax(logits, dim=1)
    return F.interpolate(
        probabilities,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )[0]


def segmentation_metrics(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    eps = 1e-9
    return {
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "pixel_accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_ratio": float(prediction.mean()),
        "ground_truth_ratio": float(target.mean()),
    }


def to_mask_image(mask):
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def make_comparison(image, gt, baseline, tta, baseline_metrics, tta_metrics, output):
    panel_w, panel_h, top = 480, 320, 48
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()

    input_panel = image.resize((panel_w, panel_h), Image.BICUBIC)
    gt_panel = to_mask_image(gt).convert("RGB").resize((panel_w, panel_h), Image.NEAREST)
    base_panel = to_mask_image(baseline).convert("RGB").resize((panel_w, panel_h), Image.NEAREST)
    tta_panel = to_mask_image(tta).convert("RGB").resize((panel_w, panel_h), Image.NEAREST)

    diff = np.zeros((*baseline.shape, 3), dtype=np.uint8)
    diff[baseline & gt] = (50, 160, 90)
    diff[(~baseline) & gt] = (220, 70, 70)
    diff[baseline & (~gt)] = (230, 170, 40)
    diff[tta & (~baseline) & gt] = (40, 230, 255)
    diff[(~tta) & baseline & gt] = (240, 50, 220)
    diff_panel = Image.fromarray(diff).resize((panel_w, panel_h), Image.NEAREST)

    panels = [
        (input_panel, "Input"),
        (gt_panel, "Ground truth"),
        (base_panel, f"Baseline  Dice {baseline_metrics['dice']:.4f}"),
        (tta_panel, f"Flip TTA  Dice {tta_metrics['dice']:.4f}"),
        (diff_panel, "TTA change: cyan better / magenta worse"),
    ]
    canvas = Image.new("RGB", (panel_w * len(panels), panel_h + top), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(panels):
        canvas.paste(panel, (index * panel_w, top))
        draw.text((index * panel_w + 10, 12), label, fill="black", font=font)
    canvas.save(output, quality=95)


def main():
    required = [WEIGHT, INPUT, GROUND_TRUTH]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(n_channels=3, n_classes=2, bilinear=False).to(device).eval()
    net.load_state_dict(torch.load(WEIGHT, map_location=device))

    image = Image.open(INPUT).convert("RGB")
    ground_truth = np.asarray(Image.open(GROUND_TRUTH).convert("L")) > 0

    with torch.inference_mode():
        _ = predict_probabilities(net, image, device)  # warmup

        start = time.perf_counter()
        baseline_probs = predict_probabilities(net, image, device)
        baseline_time = time.perf_counter() - start
        baseline_mask = baseline_probs.argmax(0).cpu().numpy().astype(bool)

        flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        start = time.perf_counter()
        original_probs = predict_probabilities(net, image, device)
        flipped_probs = predict_probabilities(net, flipped, device)
        flipped_probs = torch.flip(flipped_probs, dims=(2,))
        tta_probs = (original_probs + flipped_probs) / 2.0
        tta_time = time.perf_counter() - start
        tta_mask = tta_probs.argmax(0).cpu().numpy().astype(bool)

    baseline_metrics = segmentation_metrics(baseline_mask, ground_truth)
    tta_metrics = segmentation_metrics(tta_mask, ground_truth)

    to_mask_image(baseline_mask).save(BASELINE_MASK)
    to_mask_image(tta_mask).save(TTA_MASK)
    to_mask_image(ground_truth).save(GT_MASK)
    make_comparison(image, ground_truth, baseline_mask, tta_mask, baseline_metrics, tta_metrics, COMPARISON)

    improved_pixels = int((tta_mask & ~baseline_mask & ground_truth).sum())
    worsened_pixels = int((baseline_mask & ~tta_mask & ground_truth).sum())
    result = {
        "dataset": "Carvana",
        "sample": INPUT.name,
        "device": str(device),
        "baseline": {**baseline_metrics, "inference_seconds": baseline_time},
        "flip_tta": {**tta_metrics, "inference_seconds": tta_time},
        "dice_gain": tta_metrics["dice"] - baseline_metrics["dice"],
        "iou_gain": tta_metrics["iou"] - baseline_metrics["iou"],
        "pixels_corrected_by_tta": improved_pixels,
        "pixels_harmed_by_tta": worsened_pixels,
        "comparison_image": str(COMPARISON),
    }
    METRICS.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Baseline mask:", BASELINE_MASK)
    print("TTA mask:", TTA_MASK)
    print("Ground truth:", GT_MASK)
    print("Comparison:", COMPARISON)


if __name__ == "__main__":
    main()

