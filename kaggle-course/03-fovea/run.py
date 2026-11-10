"""Train point localization folds, compare OOF, and export template-ordered CSV."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import device, loader, save_state, seed_all
from model import LocalRegressor, ResNetHeatmap, V12UNet, objective

VERSIONS = ("mean", "v1", "v3", "v4", "v8", "v9", "v11", "v12")
LOCAL = ("v9", "v11")
MEAN = np.array([.485, .456, .406], dtype=np.float32)
STD = np.array([.229, .224, .225], dtype=np.float32)


def images(directory):
    paths = sorted(p for p in Path(directory).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".bmp"})
    result = {str(int(p.stem)) if p.stem.isdigit() else p.stem: p for p in paths}
    if not result or len(result) != len(paths):
        raise ValueError(f"Missing images or duplicate IDs in {directory}")
    return result


def image_id(value):
    name = Path(str(value)).stem
    return str(int(name)) if name.isdigit() else name


class RetinaPoints(Dataset):
    def __init__(self, rows, paths, args, training=False):
        self.rows = rows.to_dict("records")
        self.paths = paths
        self.args = args
        self.training = training
        self.clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))
        self.grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, args.size, dtype=np.float32)] * 2), -1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        key = row["ID"]
        raw = cv2.imread(str(self.paths[key]))
        if raw is None:
            raise ValueError(f"Unable to read {self.paths[key]}")
        h, w = raw.shape[:2]
        x, y, width, height = 0, 0, w, h
        if self.args.version in LOCAL:
            cx, cy = row["Coarse_X"], row["Coarse_Y"]
            width, height = min(self.args.crop, w), min(self.args.crop, h)
            x = int(np.clip(round(cx - width / 2), 0, w - width))
            y = int(np.clip(round(cy - height / 2), 0, h - height))
        image = raw[y:y+height, x:x+width, ::-1]
        size = self.args.size
        if self.args.version == "v3":
            scale = size / max(width, height)
            fit_w, fit_h = max(1, round(width*scale)), max(1, round(height*scale))
            left, top = (size-fit_w)//2, (size-fit_h)//2
            canvas = np.zeros((size, size, 3), dtype=np.uint8)
            canvas[top:top+fit_h, left:left+fit_w] = cv2.resize(
                image, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
            image = canvas
            width, height = w * size / fit_w, h * size / fit_h
            x, y = -left*w/fit_w, -top*h/fit_h
        else:
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        rgb = (image.astype(np.float32) / 255. - MEAN) / STD
        if self.args.version == "v12":
            green = (self.clahe.apply(image[..., 1]).astype(np.float32) / 255. - .5) / .25
            rgb = np.concatenate((rgb, green[..., None]), axis=2)
        truth = np.array([(row.get("True_X", 0) - x) / width,
                          (row.get("True_Y", 0) - y) / height], dtype=np.float32)
        if self.training and np.random.random() < .5:
            rgb = rgb[:, ::-1]
            truth[0] = 1 - truth[0]
        if self.training and np.random.random() < .5:
            rgb = rgb * np.random.uniform(.9, 1.1)
        if self.args.version == "v12":
            rgb = np.concatenate((rgb, self.grid), axis=2)
        if self.args.version in LOCAL:
            truth = np.clip(truth, 0., 1.)
        geometry = torch.tensor([x, y, width, height, w, h], dtype=torch.float32)
        coarse = torch.tensor([row.get("Coarse_X", 0), row.get("Coarse_Y", 0)])
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)), truth, geometry, coarse, key


def network(version, pretrained):
    if version == "v12":
        return V12UNet()
    if version in LOCAL:
        return LocalRegressor(version == "v11", pretrained)
    return ResNetHeatmap(pretrained)


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def update_ema(ema, model):
    for a, b in zip(ema.parameters(), model.parameters()):
        a.lerp_(b.detach(), .005)
    for a, b in zip(ema.buffers(), model.buffers()):
        a.copy_(b)


@torch.inference_mode()
def predict(model, batches, args, dev, prior=None):
    if model is not None:
        model.eval()
    rows = []
    for x, _, geometry, coarse, ids in batches:
        if model is None:
            xy = torch.as_tensor(prior, dtype=torch.float32).repeat(len(ids), 1)
        else:
            x = x.to(dev)
            xy = model(x)[0].float()
            if args.tta:
                flipped = x.flip(-1)
                if args.version == "v12":
                    flipped[:, 4].neg_()
                flipped_xy = model(flipped)[0].float()
                flipped_xy[:, 0] = 1 - flipped_xy[:, 0]
                xy = (xy + flipped_xy) / 2
            xy = xy.cpu()
        xy = xy * geometry[:, 2:4] + geometry[:, :2]
        if args.version == "v11":
            delta = xy - coarse
            scale = (args.limit / delta.norm(dim=1, keepdim=True).clamp_min(1e-8)).clamp(max=1.)
            xy = coarse + delta * scale
        xy = torch.minimum(xy.clamp_min(0), geometry[:, 4:6] - 1)
        rows.extend((key, *point) for key, point in zip(ids, xy.tolist()))
    return pd.DataFrame(rows, columns=["ID", "Pred_X", "Pred_Y"])


def score(points, truth):
    merged = points.merge(truth, on="ID", validate="one_to_one")
    if len(merged) != len(points):
        raise ValueError("OOF IDs do not match annotations")
    return float(np.sqrt(np.mean((merged[["Pred_X", "Pred_Y"]].to_numpy()
                                  - merged[["True_X", "True_Y"]].to_numpy())**2)))


def run_fold(args, seed, fold, train, val, test, train_paths, test_paths, out):
    seed_all(seed * 10 + fold)
    dev = device()
    train_set = RetinaPoints(train, train_paths, args, True)
    val_batches = loader(RetinaPoints(val, train_paths, args), args.batch)
    test_batches = loader(RetinaPoints(test, test_paths, args), args.batch)
    if args.version == "mean":
        prior = []
        for row in train.itertuples():
            image = cv2.imread(str(train_paths[row.ID]))
            h, w = image.shape[:2]
            prior.append([row.True_X / w, row.True_Y / h])
        center = np.mean(prior, axis=0)
        return predict(None, val_batches, args, dev, center), predict(None, test_batches, args, dev, center)

    net = network(args.version, not args.scratch).to(dev)
    ema = deepcopy(net).eval() if args.version == "v12" else None
    batches = loader(train_set, args.batch, True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * .05)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    best, state, snapshots, history = float("inf"), None, [], []
    for epoch in range(1, args.epochs + 1):
        net.train()
        losses = []
        for x, y, _, _, _ in batches:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                loss = objective(net(x.to(dev)), y.to(dev), args.version)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                update_ema(ema, net)
            losses.append(float(loss.detach()))
        scheduler.step()
        current = ema if ema is not None else net
        rmse = score(predict(current, val_batches, args, dev), val)
        history.append(dict(epoch=epoch, loss=np.mean(losses), rmse=rmse))
        print(f"seed={seed} fold={fold} epoch={epoch} RMSE={rmse:.4f}", flush=True)
        if rmse < best:
            best, state = rmse, clone_state(current)
        if args.strategy == "snapshots" and epoch in args.snapshots:
            snapshots.append(clone_state(current))
    if args.strategy == "snapshots":
        state = {k: torch.stack([s[k] for s in snapshots]).mean(0) if v.is_floating_point()
                 else snapshots[-1][k] for k, v in snapshots[-1].items()}
    net.load_state_dict(state)
    save_state(out / f"seed{seed}_fold{fold}.pth", net, version=args.version, fold=fold, seed=seed)
    pd.DataFrame(history).to_csv(out / f"seed{seed}_fold{fold}_history.csv", index=False)
    return predict(net, val_batches, args, dev), predict(net, test_batches, args, dev)


def add_coarse(table, path):
    coarse = pd.read_csv(path, dtype={"ID": str})[["ID", "Pred_X", "Pred_Y"]]
    coarse["ID"] = coarse.ID.map(image_id)
    result = table.merge(coarse.rename(columns={"Pred_X": "Coarse_X", "Pred_Y": "Coarse_Y"}),
                         on="ID", validate="one_to_one")
    if len(result) != len(table):
        raise ValueError("Missing coarse predictions")
    return result


def submission(points, template_path, out):
    lookup = dict(zip(points.ID, zip(points.Pred_X, points.Pred_Y)))
    template = pd.read_csv(template_path)
    if not template.ImageID.is_unique or len(template) != 2 * len(lookup):
        raise ValueError("Expected exactly two unique template rows per image")
    values = []
    seen = set()
    for name in template.ImageID:
        key, axis = name.rsplit("_Fovea_", 1)
        if axis not in ("X", "Y"):
            raise ValueError(name)
        key = image_id(key)
        if (key, axis) in seen:
            raise ValueError(f"Duplicate normalized ID: {name}")
        seen.add((key, axis))
        values.append(lookup[key][axis == "Y"])
    if not np.isfinite(values).all():
        raise ValueError("Non-finite prediction")
    template["value"] = values
    template.to_csv(out / "submission.csv", index=False)


def average(parts):
    return pd.concat(parts).groupby("ID", as_index=False)[["Pred_X", "Pred_Y"]].mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=VERSIONS, default="v12")
    p.add_argument("--train-dir", required=True)
    p.add_argument("--test-dir", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fold-csv", help="Optional original ID/fold assignments")
    p.add_argument("--coarse-oof")
    p.add_argument("--coarse-test")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--folds", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--crop", type=int, default=384)
    p.add_argument("--limit", type=float, default=20)
    p.add_argument("--strategy", choices=("best", "snapshots"))
    p.add_argument("--snapshots", nargs="+", type=int, default=[60, 70, 80])
    p.add_argument("--scratch", action="store_true")
    p.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    args.seeds = args.seeds or ([42, 2026, 3407] if args.version == "v12" else [42])
    args.folds = args.folds or (1 if args.version in ("v1", "v3") else 5)
    args.epochs = args.epochs or (80 if args.version == "v12" else 35)
    args.lr = 3e-4 if args.version == "v12" else 1e-4
    args.strategy = args.strategy or ("snapshots" if args.version == "v12" else "best")
    if args.strategy == "snapshots" and (args.version != "v12" or
            not all(1 <= e <= args.epochs for e in args.snapshots)):
        p.error("Snapshot averaging requires V12 and epochs covering every snapshot")
    cv2.setNumThreads(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_paths, test_paths = images(args.train_dir), images(args.test_dir)
    truth = pd.read_csv(args.labels, dtype=str).rename(
        columns={"data": "ID", "Fovea_X": "True_X", "Fovea_Y": "True_Y"})
    truth = truth[["ID", "True_X", "True_Y"]].copy()
    truth["ID"] = truth.ID.map(image_id)
    truth[["True_X", "True_Y"]] = truth[["True_X", "True_Y"]].astype(float)
    if not truth.ID.is_unique or set(truth.ID) != set(train_paths):
        raise ValueError("Training labels must match image IDs one-to-one")
    if not np.isfinite(truth[["True_X", "True_Y"]]).all().all():
        raise ValueError("Non-finite training coordinates")
    invisible = (truth[["True_X", "True_Y"]] == 0).all(axis=1)
    if invisible.any():
        print(f"Excluding {invisible.sum()} invisible points from coordinate training")
        truth = truth.loc[~invisible].reset_index(drop=True)
    test = pd.DataFrame({"ID": list(test_paths)})
    if args.version in LOCAL:
        if not args.coarse_oof or not args.coarse_test:
            p.error("Local correction requires external fold-safe coarse OOF and test predictions")
        truth, test = add_coarse(truth, args.coarse_oof), add_coarse(test, args.coarse_test)
    if args.fold_csv:
        folds = pd.read_csv(args.fold_csv, dtype={"ID": str})[["ID", "fold"]]
        folds["ID"] = folds.ID.map(image_id)
        truth = truth.merge(folds, on="ID", validate="one_to_one", how="left")
        if truth.fold.isna().any() or truth.fold.nunique() < 2:
            raise ValueError("Invalid original fold assignments")
        partitions = [(int(f), np.flatnonzero(truth.fold != f), np.flatnonzero(truth.fold == f))
                      for f in sorted(truth.fold.unique())]
    elif args.folds == 1:
        tr, va = train_test_split(np.arange(len(truth)), test_size=.20, random_state=42)
        partitions = [(0, tr, va)]
    else:
        partitions = [(f, tr, va) for f, (tr, va) in enumerate(
            KFold(args.folds, shuffle=True, random_state=42).split(truth))]
    oof, tests = [], []
    for seed in args.seeds:
        for fold, tr, va in partitions:
            val_pred, test_pred = run_fold(args, seed, fold, truth.iloc[tr], truth.iloc[va],
                                          test, train_paths, test_paths, out)
            oof.append(val_pred.assign(seed=seed, fold=fold))
            tests.append(test_pred.assign(seed=seed, fold=fold))
    pd.concat(oof).to_csv(out / "oof_by_model.csv", index=False)
    pd.concat(tests).to_csv(out / "test_by_model.csv", index=False)
    average(oof).to_csv(out / "oof.csv", index=False)
    test_points = average(tests)
    test_points.to_csv(out / "test_predictions.csv", index=False)
    submission(test_points, args.template, out)
    print("Coordinate RMSE:", score(average(oof), truth))


if __name__ == "__main__":
    main()
