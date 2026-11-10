"""Shared image-folder classification trainer and probability ensemble."""
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms as T
from torchvision.datasets import ImageFolder

from common import device, loader, load_state, save_state, seed_all, write_json


@dataclass
class Recipe:
    model: str
    size: int
    epochs: int
    lr: float = 1e-4
    wd: float = 1e-4
    crop: float = 0.85
    smoothing: float = 0.0
    pretrained: bool = True
    source: str = ""
    weighted: bool = False
    head_lr: float = 0.0
    schedule: str = "cosine"
    batch: int = 32
    patience: int = 0


class Images(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as im:
            return self.transform(im.convert("RGB")), label


def preprocessing(recipe, task, training, mean, std):
    if training:
        if recipe.crop < 1:
            resize = T.RandomResizedCrop(recipe.size, scale=(recipe.crop, 1.0),
                                         ratio=(0.9, 1.1))
        else:
            resize = T.Resize((recipe.size, recipe.size))
        if task == "plants":
            spatial = [resize, T.RandomHorizontalFlip(.5), T.RandomVerticalFlip(.2),
                       T.RandomRotation(15), T.ColorJitter(.2, .2, .15, .03)]
        else:
            spatial = [resize, T.RandomHorizontalFlip(), T.RandomRotation(8),
                       T.ColorJitter(brightness=.1, contrast=.1)]
    else:
        spatial = [T.Resize((recipe.size, recipe.size))]
    return T.Compose([*spatial, T.ToTensor(), T.Normalize(mean, std)])


def network(recipe, classes, initialize):
    name = recipe.model
    n = len(classes)
    if name == "hf_vit":
        from transformers import AutoConfig, ViTForImageClassification

        mapping = {i: c for i, c in enumerate(classes)}
        config = AutoConfig.from_pretrained(recipe.source, num_labels=n,
                                            id2label=mapping,
                                            label2id={c: i for i, c in mapping.items()})
        model = (ViTForImageClassification.from_pretrained(
            recipe.source, config=config, ignore_mismatched_sizes=True)
            if initialize and recipe.pretrained else ViTForImageClassification(config))
        return model, model.classifier

    weights = "DEFAULT" if initialize and recipe.pretrained else None
    model = models.get_model(name, weights=weights)
    if name == "resnet18":
        if recipe.size == 48:
            model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, n)
        return model, model.fc
    if name == "vit_b_16":
        model.heads.head = nn.Linear(model.heads.head.in_features, n)
        return model, model.heads
    last = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last.in_features, n)
    return model, model.classifier


def logits(model, x):
    output = model(x)
    return output if torch.is_tensor(output) else output.logits


@torch.inference_mode()
def probabilities(model, batches, target_device, tta):
    model.eval()
    labels, values = [], []
    for x, y in batches:
        x = x.to(target_device)
        views = [x]
        if tta in ("horizontal", "four"):
            views.append(x.flip(-1))
        if tta == "four":
            views.extend([x.flip(-2), x.flip((-2, -1))])
        values.append(torch.stack([logits(model, view).softmax(1)
                                   for view in views]).mean(0).cpu().numpy())
        labels.extend(y.tolist())
    return np.array(labels), np.concatenate(values)


def train(args, recipes, task):
    seed_all(args.seed)
    recipe = recipes[args.recipe]
    epochs = args.epochs or recipe.epochs
    batch = args.batch or recipe.batch
    dataset = ImageFolder(args.train_dir)
    classes = dataset.classes
    if len(classes) != (5 if task == "plants" else 6):
        raise ValueError(f"Unexpected classes: {classes}")
    indices = np.arange(len(dataset))
    train_ids, val_ids = train_test_split(
        indices, test_size=args.val_fraction, random_state=args.seed, stratify=dataset.targets)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mean, std = (.485, .456, .406), (.229, .224, .225)
    if recipe.source:
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(recipe.source)
        mean, std = tuple(processor.image_mean), tuple(processor.image_std)
    tr = Images([dataset.samples[i] for i in train_ids],
                preprocessing(recipe, task, True, mean, std))
    va = Images([dataset.samples[i] for i in val_ids],
                preprocessing(recipe, task, False, mean, std))
    train_batches, val_batches = loader(tr, batch, True), loader(va, batch)
    target_device = device()
    model, head = network(recipe, classes, not args.scratch and not args.init)
    model.to(target_device)
    if args.init:
        previous = load_state(args.init)
        if previous["classes"] != classes:
            raise ValueError("Checkpoint class order differs from dataset")
        model.load_state_dict(previous["state"])
    weights = None
    if recipe.weighted:
        counts = np.bincount(np.array(dataset.targets)[train_ids], minlength=len(classes))
        weights = torch.as_tensor(counts.sum() / (len(classes) * counts),
                                  dtype=torch.float32, device=target_device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=recipe.smoothing)
    if recipe.head_lr:
        head_params = list(head.parameters())
        head_ids = {id(p) for p in head_params}
        groups = [{"params": [p for p in model.parameters() if id(p) not in head_ids]},
                  {"params": head_params, "lr": recipe.head_lr}]
    else:
        groups = model.parameters()
    opt = torch.optim.AdamW(groups, lr=recipe.lr, weight_decay=recipe.wd)
    if recipe.schedule == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=recipe.lr, epochs=epochs, steps_per_epoch=len(train_batches))
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = -1.0
    wait = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_batches:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(logits(model, x.to(target_device)), y.to(target_device))
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if recipe.schedule == "onecycle":
                sched.step()
        if recipe.schedule != "onecycle":
            sched.step()
        y, p = probabilities(model, val_batches, target_device, "none")
        predicted = p.argmax(1)
        f1 = f1_score(y, predicted, labels=np.arange(len(classes)), average="macro", zero_division=0)
        acc = accuracy_score(y, predicted)
        score = f1 if task == "plants" else acc
        metrics = dict(epoch=epoch, loss=float(np.mean(losses)), accuracy=float(acc), macro_f1=float(f1))
        history.append(metrics)
        print(metrics, flush=True)
        if score > best:
            best, wait = score, 0
            save_state(out / "best.pth", model, classes=classes, recipe=asdict(recipe),
                       mean=list(mean), std=list(std), epoch=epoch, task=task)
        else:
            wait += 1
            if recipe.patience and wait >= recipe.patience:
                break
    write_json(out / "history.json", history)
    model.load_state_dict(load_state(out / "best.pth")["state"])
    for tta in dict.fromkeys(("none", args.tta)):
        y, p = probabilities(model, val_batches, target_device, tta)
        table = pd.DataFrame(p, columns=classes)
        table.insert(0, "true", [classes[i] for i in y])
        table.insert(0, "path", [str(Path(x[0]).relative_to(args.train_dir)) for x in va.samples])
        table.to_csv(out / f"validation_{tta}.csv", index=False)
        wrong = table.loc[y != p.argmax(1), ["path", "true"]].copy()
        wrong["predicted"] = np.array(classes)[p.argmax(1)[y != p.argmax(1)]]
        wrong.to_csv(out / f"errors_{tta}.csv", index=False)
        write_json(out / f"metrics_{tta}.json", {
            "accuracy": float(accuracy_score(y, p.argmax(1))),
            "macro_f1": float(f1_score(y, p.argmax(1), average="macro", zero_division=0)),
            "confusion": confusion_matrix(y, p.argmax(1), labels=np.arange(len(classes))).tolist(),
            "classes": classes,
        })


def predict(args, task):
    template = pd.read_csv(args.template, dtype={args.id_column: str})
    if template[args.id_column].isna().any() or not template[args.id_column].is_unique:
        raise ValueError("Template IDs must be present and unique")
    images = [p for p in Path(args.test_dir).rglob("*")
              if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
    by_name = {p.name: p for p in images}
    by_stem = {p.stem: p for p in images}
    if len(by_name) != len(images) or len(by_stem) != len(images):
        raise ValueError("Ambiguous test filenames or stems")
    samples = []
    for name in template[args.id_column]:
        path = by_name.get(name) or by_stem.get(name)
        if path is None:
            raise FileNotFoundError(f"Template image missing: {name}")
        samples.append((path, 0))
    weights = np.array(args.weights or [1.] * len(args.checkpoints))
    if len(weights) != len(args.checkpoints) or not np.isfinite(weights).all() or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("Supply a non-negative weight for each checkpoint")
    target_device = device()
    classes, ensemble = None, None
    for path, weight in zip(args.checkpoints, weights / weights.sum()):
        saved = load_state(path)
        if saved["task"] != task:
            raise ValueError("Checkpoint belongs to another task")
        recipe = Recipe(**saved["recipe"])
        model, _ = network(recipe, saved["classes"], False)
        model.load_state_dict(saved["state"])
        model.to(target_device)
        test = Images(samples, preprocessing(recipe, task, False, saved["mean"], saved["std"]))
        _, p = probabilities(model, loader(test, args.batch), target_device, args.tta)
        if classes is None:
            classes = saved["classes"]
            ensemble = np.zeros_like(p)
        if set(classes) != set(saved["classes"]):
            raise ValueError("Incompatible checkpoint classes")
        ensemble += weight * p[:, [saved["classes"].index(c) for c in classes]]
        del model
    labels = np.array(classes)[ensemble.argmax(1)]
    if args.label_order:
        if set(args.label_order) != set(classes):
            raise ValueError("Submission label order must contain every class exactly once")
        labels = [args.label_order.index(label) for label in labels]
    template[args.target_column] = labels
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(args.out, index=False)


def compare_validation(args):
    expected = None
    averaged = None
    weights = np.asarray(args.weights or [1.] * len(args.inputs))
    if len(weights) != len(args.inputs) or not np.isfinite(weights).all() or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("Supply a non-negative weight for each validation CSV")
    for path, weight in zip(args.inputs, weights / weights.sum()):
        table = pd.read_csv(path).set_index("path", verify_integrity=True)
        if expected is None:
            expected = table["true"]
            classes = sorted(set(table.columns) - {"true"})
        if set(expected.index) != set(table.index) or set(table.columns) != set(classes) | {"true"}:
            raise ValueError("Validation splits and classes must match")
        aligned = table.loc[expected.index]
        if not aligned["true"].equals(expected):
            raise ValueError("Validation labels differ")
        values = aligned[classes].to_numpy()
        averaged = weight*values if averaged is None else averaged + weight*values
    prediction = np.array(classes)[averaged.argmax(1)]
    result = {"accuracy": float(accuracy_score(expected, prediction)),
              "macro_f1": float(f1_score(expected, prediction, average="macro"))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(result)


def main(recipes, task, defaults):
    parser = argparse.ArgumentParser(description=task)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--recipe", choices=recipes, default=next(iter(recipes)))
    fit.add_argument("--train-dir", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--val-fraction", type=float, default=defaults[0])
    fit.add_argument("--epochs", type=int)
    fit.add_argument("--batch", type=int)
    fit.add_argument("--init", help="Compatible checkpoint for fine-tuning")
    fit.add_argument("--scratch", action="store_true")
    fit.add_argument("--tta", choices=("none", "horizontal", "four"), default=defaults[1])
    test = sub.add_parser("predict")
    test.add_argument("--test-dir", required=True)
    test.add_argument("--template", required=True)
    test.add_argument("--checkpoints", required=True, nargs="+")
    test.add_argument("--weights", nargs="+", type=float)
    test.add_argument("--label-order", nargs="+", help="Class names in official integer-ID order")
    test.add_argument("--id-column", default=defaults[2])
    test.add_argument("--target-column", default=defaults[3])
    test.add_argument("--batch", type=int, default=32)
    test.add_argument("--tta", choices=("none", "horizontal", "four"), default=defaults[1])
    test.add_argument("--out", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--inputs", nargs="+", required=True)
    compare.add_argument("--weights", nargs="+", type=float)
    compare.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "compare":
        compare_validation(args)
    elif args.command == "train":
        train(args, recipes, task)
    else:
        predict(args, task)
