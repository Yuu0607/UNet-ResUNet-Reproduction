import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "Pytorch-UNet-3.0"
WEIGHT = ROOT / "unet_carvana_scale0.5_epoch2.pth"
DATA_DIR = ROOT / "resunet_data"
IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"
OUT_METRICS = ROOT / "resunet_metrics.json"
OUT_MONTAGE = ROOT / "resunet_validation_montage.jpg"
OUT_COMPARISON = ROOT / "resunet_comparison.jpg"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
(ROOT / ".matplotlib").mkdir(exist_ok=True)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))

from unet import UNet  # noqa: E402
from resunet_model import create_resunet  # noqa: E402

SEED = 42
IMAGE_SIZE = (192, 128)
EPOCHS = 5
BATCH_SIZE = 4
LR = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CarvanaSubset(Dataset):
    def __init__(self, files, train):
        self.files = files
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_path = self.files[index]
        mask_path = MASK_DIR / f"{image_path.stem}.png"
        image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE, Image.BICUBIC)
        mask = Image.open(mask_path).convert("L").resize(IMAGE_SIZE, Image.NEAREST)

        array = np.asarray(image)
        target = (np.asarray(mask) > 0).astype(np.uint8)

        if self.train and random.random() < 0.5:
            array = np.ascontiguousarray(array[:, ::-1])
            target = np.ascontiguousarray(target[:, ::-1])

        if self.train and random.random() < 0.3:
            factor = random.uniform(0.9, 1.1)
            array = np.clip(array.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        image_tensor = torch.from_numpy(array.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(target).long()
        return image_tensor, mask_tensor


def soft_dice_loss(logits, target):
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    target = target.float()
    intersection = (probabilities * target).sum(dim=(1, 2))
    union = probabilities.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2 * intersection + 1e-7) / (union + 1e-7)
    return 1.0 - dice.mean()


def train_one_epoch(model, loader, optimizer):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks) + soft_dice_loss(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.inference_mode()
def evaluate(model, files):
    model.eval()
    scores = []
    for path in files:
        image = Image.open(path).convert("RGB").resize(IMAGE_SIZE, Image.BICUBIC)
        mask = Image.open(MASK_DIR / f"{path.stem}.png").convert("L").resize(IMAGE_SIZE, Image.NEAREST)
        image_tensor = torch.from_numpy(np.asarray(image).transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(DEVICE)
        target = torch.from_numpy((np.asarray(mask) > 0).astype(np.uint8)).unsqueeze(0).to(DEVICE)
        probabilities = torch.softmax(model(image_tensor), dim=1)[:1, 1]
        prediction = probabilities > 0.5
        tp = torch.logical_and(prediction, target.bool()).sum().item()
        fp = torch.logical_and(prediction, ~target.bool()).sum().item()
        fn = torch.logical_and(~prediction, target.bool()).sum().item()
        dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
        iou = tp / (tp + fp + fn + 1e-9)
        scores.append({
            "sample": path.stem,
            "dice": dice,
            "iou": iou,
            "prediction_ratio": prediction.float().mean().item(),
            "target_ratio": target.float().mean().item(),
        })
    return {
        "dice": float(np.mean([item["dice"] for item in scores])),
        "iou": float(np.mean([item["iou"] for item in scores])),
        "samples": scores,
    }


def fine_tune(model, train_files, val_files, name):
    dataset = CarvanaSubset(train_files, train=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    best_dice = -1.0
    best_state = None
    history = []
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, loader, optimizer)
        metrics = evaluate(model, val_files)
        history.append({"epoch": epoch, "loss": train_loss, "val_dice": metrics["dice"], "val_iou": metrics["iou"]})
        print(f"{name} epoch {epoch}/{EPOCHS} loss={train_loss:.6f} val_dice={metrics['dice']:.6f}", flush=True)
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            best_state = copy.deepcopy(model.state_dict())
    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)
    final = evaluate(model, val_files)
    return final, history, elapsed


def to_mask_image(mask):
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def overlay(image, mask):
    image = image.convert("RGB")
    red = Image.new("RGB", image.size, (255, 45, 45))
    return Image.composite(Image.blend(image, red, 0.45), image, to_mask_image(mask))


def get_font(size):
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def predict_mask(model, path):
    model.eval()
    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE, Image.BICUBIC)
    mask = Image.open(MASK_DIR / f"{path.stem}.png").convert("L").resize(IMAGE_SIZE, Image.NEAREST)
    tensor = torch.from_numpy(np.asarray(image).transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        prediction = (torch.softmax(model(tensor), dim=1)[0, 1] > 0.5).cpu().numpy()
    target = np.asarray(mask) > 0
    return image, target, prediction


def make_detailed_comparison(path, unet, resunet, baseline_item, resunet_item):
    image, target, base_mask = predict_mask(unet, path)
    _, _, res_mask = predict_mask(resunet, path)
    panel_w, panel_h, top = 480, 320, 48
    font = get_font(20)

    diff = np.zeros((*target.shape, 3), dtype=np.uint8)
    diff[base_mask & target] = (50, 160, 90)
    diff[(~base_mask) & target] = (220, 70, 70)
    diff[base_mask & (~target)] = (230, 170, 40)
    diff[res_mask & (~base_mask) & target] = (40, 230, 255)
    diff[(~res_mask) & base_mask & target] = (240, 50, 220)

    panels = [
        (image, "Input"),
        (to_mask_image(target).convert("RGB"), "Ground truth"),
        (to_mask_image(base_mask).convert("RGB"), f"UNet  Dice {baseline_item['dice']:.4f}"),
        (to_mask_image(res_mask).convert("RGB"), f"ResUNet  Dice {resunet_item['dice']:.4f}"),
        (Image.fromarray(diff), "Cyan: ResUNet better / Magenta: worse"),
    ]
    canvas = Image.new("RGB", (panel_w * 5, panel_h + top), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(panels):
        canvas.paste(panel.resize((panel_w, panel_h), Image.NEAREST), (index * panel_w, top))
        draw.text((index * panel_w + 10, 12), label, fill="black", font=font)
    canvas.save(OUT_COMPARISON, quality=95)


def make_montage(val_files, unet, resunet, baseline_samples, resunet_samples):
    cell_w, cell_h, row_h = 220, 147, 182
    font = get_font(15)
    canvas = Image.new("RGB", (cell_w * 5, row_h * len(val_files) + 30), "white")
    draw = ImageDraw.Draw(canvas)
    headers = ["Input", "Ground truth", "UNet mask", "ResUNet mask", "Difference"]
    for index, label in enumerate(headers):
        draw.text((index * cell_w + 8, 6), label, fill="black", font=font)

    for row, path in enumerate(val_files):
        image, target, base_mask = predict_mask(unet, path)
        _, _, res_mask = predict_mask(resunet, path)
        diff = np.zeros((*target.shape, 3), dtype=np.uint8)
        diff[(base_mask == target) & target] = (50, 160, 90)
        diff[(~base_mask) & target] = (220, 70, 70)
        diff[base_mask & (~target)] = (230, 170, 40)
        diff[res_mask & (~base_mask) & target] = (40, 230, 255)
        diff[(~res_mask) & base_mask & target] = (240, 50, 220)
        panels = [
            image,
            to_mask_image(target).convert("RGB"),
            to_mask_image(base_mask).convert("RGB"),
            to_mask_image(res_mask).convert("RGB"),
            Image.fromarray(diff),
        ]
        y = 30 + row * row_h
        for col, panel in enumerate(panels):
            canvas.paste(panel.resize((cell_w, cell_h), Image.NEAREST), (col * cell_w, y))
        draw.text((8, y + cell_h + 3), f"{path.stem}  U {baseline_samples[row]['dice']:.4f}  R {resunet_samples[row]['dice']:.4f}", fill="black", font=font)
    canvas.save(OUT_MONTAGE, quality=95)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(8)

    files = sorted(IMAGE_DIR.glob("*.jpg"))
    if len(files) != 40:
        raise RuntimeError(f"Expected 40 images, found {len(files)}")
    indices = np.random.default_rng(SEED).permutation(len(files))
    val_files = [files[i] for i in indices[:8]]
    train_files = [files[i] for i in indices[8:]]

    pretrained = torch.load(WEIGHT, map_location=DEVICE)

    unet = UNet(n_channels=3, n_classes=2, bilinear=False).to(DEVICE)
    unet.load_state_dict(pretrained)

    resunet = create_resunet(n_channels=3, n_classes=2, bilinear=False, residual_scale=0.01).to(DEVICE)
    load_result = resunet.load_state_dict(pretrained, strict=False)
    if load_result.unexpected_keys:
        raise RuntimeError(load_result.unexpected_keys)

    baseline_cache = ROOT / "unet_baseline_metrics.json"
    baseline_weight = ROOT / "unet_finetuned.pth"
    if baseline_cache.exists() and baseline_weight.exists():
        print("Loading cached fair UNet baseline...", flush=True)
        cached = json.loads(baseline_cache.read_text(encoding="utf-8"))
        unet.load_state_dict(torch.load(baseline_weight, map_location=DEVICE))
        baseline_metrics = cached["baseline"]
        baseline_history = cached["history"]["unet"]
        baseline_time = cached["training_seconds"]["unet"]
    else:
        print("Training UNet...", flush=True)
        baseline_metrics, baseline_history, baseline_time = fine_tune(unet, train_files, val_files, "UNet")
    print("Training ResUNet...", flush=True)
    resunet_metrics, resunet_history, resunet_time = fine_tune(resunet, train_files, val_files, "ResUNet")

    baseline_samples = baseline_metrics["samples"]
    resunet_samples = resunet_metrics["samples"]
    gains = [r["dice"] - b["dice"] for b, r in zip(baseline_samples, resunet_samples)]
    representative_index = int(np.argsort(gains)[len(gains) // 2])

    make_detailed_comparison(
        val_files[representative_index], unet, resunet,
        baseline_samples[representative_index], resunet_samples[representative_index],
    )
    make_montage(val_files, unet, resunet, baseline_samples, resunet_samples)

    scales = [float(p.detach().cpu()) for name, p in resunet.named_parameters() if name.endswith("residual_scale")]
    result = {
        "architecture": "UNet vs residual double-conv UNet (ResUNet)",
        "dataset": "Carvana subset",
        "train_images": len(train_files),
        "validation_images": len(val_files),
        "image_size_wh": IMAGE_SIZE,
        "epochs": EPOCHS,
        "device": str(DEVICE),
        "baseline": baseline_metrics,
        "resunet": resunet_metrics,
        "mean_dice_gain": resunet_metrics["dice"] - baseline_metrics["dice"],
        "mean_iou_gain": resunet_metrics["iou"] - baseline_metrics["iou"],
        "training_seconds": {"unet": baseline_time, "resunet": resunet_time},
        "history": {"unet": baseline_history, "resunet": resunet_history},
        "residual_scales": scales,
        "representative_sample": val_files[representative_index].stem,
        "comparison_image": str(OUT_COMPARISON),
        "validation_montage": str(OUT_MONTAGE),
    }
    OUT_METRICS.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(unet.state_dict(), ROOT / "unet_finetuned.pth")
    torch.save(resunet.state_dict(), ROOT / "resunet_finetuned.pth")

    print(json.dumps({k: result[k] for k in [
        "mean_dice_gain", "mean_iou_gain", "training_seconds", "residual_scales", "representative_sample"
    ]}, ensure_ascii=False, indent=2))
    print("UNet Dice:", baseline_metrics["dice"])
    print("ResUNet Dice:", resunet_metrics["dice"])
    print("Metrics:", OUT_METRICS)
    print("Comparison:", OUT_COMPARISON)
    print("Montage:", OUT_MONTAGE)


if __name__ == "__main__":
    main()
