"""Image-level cross-validation, sliding-window inference and OOF comparisons."""
import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import device, loader, save_state, seed_all, write_json
from metrics import overlap, rle_decode, rle_encode
from model import UNet, loss_value


@dataclass
class Recipe:
    size: int = 512
    patch: int = 128
    stride: int = 64
    channels: int = 3
    residual: bool = True
    ema: bool = False
    aux: bool = False
    aux_core: bool = False
    context: bool = False
    thin: bool = False
    official: bool = True
    loss: str = "dice"
    trees: bool = False


V5 = Recipe()
RECIPES = {
    "v1": Recipe(patch=0, residual=False, official=False),
    "v2": replace(V5, official=False, patch=256, stride=128),
    "v3": Recipe(size=0, patch=0, residual=False, official=False, ema=True, aux=True),
    "v4": replace(V5, channels=1, official=False),
    "v5": V5,
    "v7-tversky": replace(V5, loss="tversky"),
    "v8-core": replace(V5, aux=True, aux_core=True),
    "v9-stride32": replace(V5, stride=32),
    "v10-thin": replace(V5, thin=True),
    "v11-focal-trial": replace(V5, loss="focal"),
    "v12-context": replace(V5, context=True),
    "extra-trees": Recipe(size=0, patch=0, official=False, trees=True),
}


def key(name):
    name = Path(str(name)).stem
    return str(int(name)) if name.isdigit() else name


def indexed(directory):
    paths = sorted(p for p in Path(directory).iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".bmp"})
    mapping = {key(p.name): p for p in paths}
    if not mapping or len(mapping) != len(paths):
        raise ValueError(f"No images or ambiguous IDs in {directory}")
    return mapping


def read(path, mode):
    im = cv2.imread(str(path), mode)
    if im is None:
        raise ValueError(f"Cannot read {path}")
    return im


def image(path, recipe):
    bgr = read(path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if recipe.channels == 1:
        rgb = rgb[..., 1:2]
    return rgb.astype(np.float32) / 255


def mask(path, foreground):
    raw = read(path, cv2.IMREAD_GRAYSCALE)
    if not np.isin(raw, [0, 1, 255]).all():
        raise ValueError(f"Expected a binary mask: {path}")
    return raw == 0 if foreground == "dark" else raw > 0


def prepare_masks(args):
    images, masks = indexed(args.train_dir), indexed(args.mask_dir)
    if set(images) != set(masks):
        raise ValueError("Image and mask IDs differ")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, path in images.items():
        h, w = read(path, cv2.IMREAD_COLOR).shape[:2]
        normalized = mask(masks[k], args.foreground).astype(np.uint8) * 255
        scaled = cv2.resize(normalized, (w,h), interpolation=cv2.INTER_LINEAR)
        threshold = 128 if args.core else 1
        converted = (scaled >= threshold).astype(np.uint8) * 255
        if not cv2.imwrite(str(out / f"{k}.png"), converted):
            raise OSError(f"Failed to write prepared mask {k}")


def resized(x, h, w, nearest=False):
    result = cv2.resize(x.astype(np.float32), (w, h),
                        interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    if x.ndim == 3 and result.ndim == 2:
        return result[..., None]
    return result


class TrainPatches(Dataset):
    def __init__(self, items, recipe, samples):
        self.recipe = recipe
        self.samples = samples if recipe.patch else 1
        self.items = []
        for x, y in items:
            if recipe.size:
                x = resized(x, recipe.size, recipe.size)
                y = resized(y, recipe.size, recipe.size, nearest=True) > .5
            if recipe.patch:
                h, w = y.shape
                pad_h, pad_w = max(0, recipe.patch-h), max(0, recipe.patch-w)
                x = np.pad(x, ((0,pad_h),(0,pad_w),(0,0)), mode="edge")
                y = np.pad(y, ((0,pad_h),(0,pad_w)), mode="constant")
            core = cv2.erode(y.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            candidates = np.argwhere(y & ~core if recipe.thin else y)
            self.items.append((x, y, candidates))

    def __len__(self):
        return len(self.items) * self.samples

    def __getitem__(self, i):
        x, y, candidates = self.items[i // self.samples]
        patch = self.recipe.patch
        if patch:
            h, w = y.shape
            if len(candidates) and np.random.rand() < (.85 if self.recipe.thin else .55):
                row, col = candidates[np.random.randint(len(candidates))]
                row = int(np.clip(row - np.random.randint(patch), 0, h-patch))
                col = int(np.clip(col - np.random.randint(patch), 0, w-patch))
            else:
                row, col = np.random.randint(h-patch+1), np.random.randint(w-patch+1)
            x, y = x[row:row+patch, col:col+patch], y[row:row+patch, col:col+patch]
        if np.random.rand() < .5:
            x, y = x[:, ::-1], y[:, ::-1]
        if np.random.rand() < .5:
            x, y = x[::-1], y[::-1]
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))),
                torch.from_numpy(np.ascontiguousarray(y[None], dtype=np.float32)))


def starts(length, patch, stride):
    return sorted(set([*range(0, length-patch+1, stride), length-patch]))


@torch.inference_mode()
def infer(net, x, recipe, dev, batch, tta):
    net.eval()
    h, w = x.shape[:2]
    if recipe.size:
        x = resized(x, recipe.size, recipe.size)
    x = np.pad(x, ((0, max(0, recipe.patch-x.shape[0])),
                   (0, max(0, recipe.patch-x.shape[1])), (0,0)), mode="edge")
    hh, ww = x.shape[:2]
    patch = recipe.patch
    if patch:
        tiles = [(r,c,patch,patch) for r in starts(hh,patch,recipe.stride)
                 for c in starts(ww,patch,recipe.stride)]
    else:
        tiles = [(0,0,hh,ww)]
    total, count = np.zeros((hh,ww), np.float32), np.zeros((hh,ww), np.float32)
    for i in range(0, len(tiles), batch):
        group = tiles[i:i+batch]
        windows = np.stack([x[r:r+ph,c:c+pw].transpose(2,0,1) for r,c,ph,pw in group])
        x_gpu = torch.from_numpy(windows).to(dev)
        p = net(x_gpu)[0].sigmoid()
        if tta:
            views = [p]
            for dims in [(-1,), (-2,), (-2,-1)]:
                views.append(net(x_gpu.flip(dims))[0].sigmoid().flip(dims))
            p = torch.stack(views).mean(0)
        for (r,c,ph,pw), tile in zip(group, p[:,0].cpu().numpy()):
            total[r:r+ph,c:c+pw] += tile
            count[r:r+ph,c:c+pw] += 1
    probability = total / count
    if recipe.size:
        probability = resized(probability[:recipe.size,:recipe.size], h, w)
    return probability[:h,:w]


def features(x):
    planes = [x[..., 1]]
    green = planes[0]
    for width in (3, 5, 9, 15):
        smooth = cv2.GaussianBlur(green, (width, width), 0)
        planes.extend([smooth, green-smooth, cv2.Laplacian(smooth, cv2.CV_32F),
                       cv2.Sobel(smooth, cv2.CV_32F, 1, 0),
                       cv2.Sobel(smooth, cv2.CV_32F, 0, 1),
                       cv2.blur((green-smooth)**2, (width,width))])
    return np.stack(planes, -1)


def fit_trees(train, valid, test):
    data, labels = [], []
    for x, y in train:
        matrix = features(x).reshape(-1,25)
        ix = np.random.choice(len(matrix), min(10000,len(matrix)), replace=False)
        data.append(matrix[ix])
        labels.append(y.ravel()[ix])
    estimator = ExtraTreesClassifier(n_estimators=128, random_state=42, n_jobs=-1)
    estimator.fit(np.concatenate(data), np.concatenate(labels))
    if True not in estimator.classes_:
        raise ValueError("No foreground pixels in the sampled training set")

    def predict(item):
        x = item[0]
        proba = estimator.predict_proba(features(x).reshape(-1,25))
        return proba[:, list(estimator.classes_).index(True)].reshape(x.shape[:2])

    return [predict(item) for item in valid], [predict((x,)) for x in test]


@torch.no_grad()
def update_ema(ema, model):
    for dst, src in zip(ema.parameters(), model.parameters()):
        dst.lerp_(src, .005)
    for dst, src in zip(ema.buffers(), model.buffers()):
        dst.copy_(src)


def fit_net(args, recipe, train, valid, test, fold, out):
    dev = device()
    model = UNet(recipe.channels, recipe.residual, recipe.aux, recipe.context).to(dev)
    ema = deepcopy(model).eval() if recipe.ema else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    batches = loader(TrainPatches(train, recipe, args.patches), 1 if recipe.size == 0 else args.batch, True)
    best, wait, state = -1., 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in batches:
            opt.zero_grad(set_to_none=True)
            loss = loss_value(model(x.to(dev)), y.to(dev), recipe.loss, recipe.aux_core)
            loss.backward()
            opt.step()
            if ema is not None:
                update_ema(ema, model)
            losses.append(float(loss.detach()))
        scheduler.step()
        chosen = ema if ema is not None else model
        dice = np.mean([overlap(infer(chosen, x, recipe, dev, args.batch, False) >= .5, y)["dice"]
                        for x, y in valid])
        history.append(dict(epoch=epoch, loss=np.mean(losses), dice=dice))
        print(f"fold={fold} epoch={epoch} Dice@0.50={dice:.5f}", flush=True)
        if dice > best:
            best, wait = dice, 0
            state = {k: v.detach().cpu().clone() for k, v in chosen.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(state)
    save_state(out / f"fold{fold}.pth", model, recipe=asdict(recipe), fold=fold)
    pd.DataFrame(history).to_csv(out / f"fold{fold}_history.csv", index=False)
    return ([infer(model, x, recipe, dev, args.batch, args.tta) for x, _ in valid],
            [infer(model, x, recipe, dev, args.batch, args.tta) for x in test])


def archive(path, values):
    np.savez_compressed(path, **{f"id_{k}": v for k,v in values.items()})


def evaluate(probabilities, targets, thresholds, out, folds=None):
    rows = []
    for threshold in thresholds:
        for image_id, p in probabilities.items():
            rows.append({"ID": image_id, "threshold": threshold,
                         "fold": None if folds is None else folds[image_id],
                         **overlap(p >= threshold, targets[image_id])})
    table = pd.DataFrame(rows)
    table.to_csv(out / "oof_metrics.csv", index=False)
    print(table.groupby("threshold")[["dice","precision","recall"]].mean().to_string())


def train(args):
    recipe = RECIPES[args.recipe]
    if recipe.official and not args.official_mask_dir:
        raise ValueError("This recipe requires the official-rule masks via --official-mask-dir")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = indexed(args.train_dir)
    targets = indexed(args.official_mask_dir if recipe.official else args.mask_dir)
    if set(paths) != set(targets):
        raise ValueError("Image and mask IDs differ")
    keys = sorted(paths)
    raw = {}
    for k in keys:
        x, y = image(paths[k], recipe), mask(targets[k], args.foreground)
        if x.shape[:2] != y.shape:
            raise ValueError(f"Target must have the same geometry as image {k}")
        raw[k] = (x,y)
    test = {k: image(p, recipe) for k,p in indexed(args.test_dir).items()}
    prediction, test_sum, fold_of = {}, {}, {}
    completed = 0
    for fold, (tr, va) in enumerate(KFold(args.folds, shuffle=True, random_state=args.seed).split(keys)):
        if args.only_folds is not None and fold not in args.only_folds:
            continue
        seed_all(args.seed + fold)
        train_items, val_items = ([raw[keys[i]] for i in indices] for indices in (tr,va))
        if recipe.trees:
            val_p, test_p = fit_trees(train_items, val_items, list(test.values()))
        else:
            val_p, test_p = fit_net(args, recipe, train_items, val_items, list(test.values()), fold, out)
        for i,p in zip(va,val_p):
            prediction[keys[i]], fold_of[keys[i]] = p,fold
        for k,p in zip(test,test_p):
            test_sum[k] = p if k not in test_sum else test_sum[k] + p
        completed += 1
    if completed == 0:
        raise ValueError("No selected folds were run")
    evaluate(prediction, {k:v[1] for k,v in raw.items()}, args.thresholds, out, fold_of)
    archive(out / "oof.npz", prediction)
    archive(out / "test.npz", {k:p/completed for k,p in test_sum.items()})
    write_json(out / "recipe.json", dict(recipe=asdict(recipe), seed=args.seed, folds=args.folds,
                                         folds_completed=completed, tta=args.tta))


def blend(paths, weights=None, common_ids=False):
    weights = np.asarray(weights if weights is not None else [1.] * len(paths))
    if len(weights) != len(paths) or not np.isfinite(weights).all() or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("Supply one non-negative weight per archive")
    result = {}
    allowed = None
    if common_ids:
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                ids = set(data.files)
                allowed = ids if allowed is None else allowed & ids
        if not allowed:
            raise ValueError("No shared prediction IDs")
    for i, (path, w) in enumerate(zip(paths, weights / weights.sum())):
        with np.load(path, allow_pickle=False) as data:
            if i and allowed is None and set(result) != {name[3:] for name in data.files}:
                raise ValueError("Prediction archives must cover identical IDs")
            for name in data.files:
                if allowed is not None and name not in allowed:
                    continue
                if not name.startswith("id_"):
                    raise ValueError(f"Unexpected array name {name}")
                k, p = name[3:], data[name]
                if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                    raise ValueError(f"Invalid probability map: {k}")
                if k in result and result[k].shape != p.shape:
                    raise ValueError(f"Shape differs for image {k}")
                result[k] = result[k] + w*p if k in result else w*p
    return result


def postprocess(binary, mode):
    if mode == "none":
        return binary
    if mode == "closing":
        closed = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)))
        return binary | (closed > 0)
    # Fill a single missing pixel only when both opposite neighbors are vessels.
    horizontal, vertical = np.zeros_like(binary), np.zeros_like(binary)
    horizontal[:, 1:-1] = binary[:, :-2] & binary[:, 2:]
    vertical[1:-1] = binary[:-2] & binary[2:]
    return binary | horizontal | vertical


def export(args):
    values = blend(args.inputs, args.weights)
    template = pd.read_csv(args.template, dtype={args.id_column:str})
    ids = template[args.id_column].map(key)
    if not ids.is_unique or set(ids) != set(values):
        raise ValueError("Template and prediction IDs differ")
    encoded = []
    for k in ids:
        binary = postprocess(values[k] >= args.threshold, args.postprocess)
        result = rle_encode(binary, args.order, args.start)
        if not np.array_equal(binary, rle_decode(result,binary.shape,args.order,args.start)):
            raise ValueError(f"RLE did not round-trip: {k}")
        encoded.append(result)
    template[args.target_column] = encoded
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(args.out,index=False)


def compare(args):
    values = blend(args.inputs, args.weights, args.common_ids)
    targets = indexed(args.mask_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k,p in values.items():
        truth = mask(targets[k], args.foreground)
        for threshold in args.thresholds:
            for operation in ("none","closing","short-gap"):
                result = postprocess(p >= threshold, operation)
                rows.append(dict(ID=k, threshold=threshold, operation=operation, **overlap(result, truth)))
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison.csv", index=False)
    archive(out / "blended.npz", values)
    print(table.groupby(["operation","threshold"])[["dice","precision","recall"]].mean().to_string())


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--recipe", choices=RECIPES, default="v5")
    for name in ("train-dir","test-dir","mask-dir","out"):
        fit.add_argument("--"+name, required=True)
    fit.add_argument("--official-mask-dir")
    fit.add_argument("--foreground", choices=("dark","bright"), required=True)
    fit.add_argument("--epochs",type=int,default=80)
    fit.add_argument("--patience",type=int,default=20)
    fit.add_argument("--patches",type=int,default=64)
    fit.add_argument("--batch",type=int,default=8)
    fit.add_argument("--lr",type=float,default=3e-4)
    fit.add_argument("--folds",type=int,default=5)
    fit.add_argument("--only-folds",nargs="+",type=int)
    fit.add_argument("--seed",type=int,default=42)
    fit.add_argument("--tta",action=argparse.BooleanOptionalAction,default=True)
    fit.add_argument("--thresholds",type=float,nargs="+",default=[.15,.25,.35,.45,.50,.60,.70])
    fit.set_defaults(handler=train)
    prepare = sub.add_parser("prepare-masks")
    for name in ("train-dir","mask-dir","out"):
        prepare.add_argument("--"+name,required=True)
    prepare.add_argument("--foreground",choices=("dark","bright"),required=True)
    prepare.add_argument("--core",action="store_true")
    prepare.set_defaults(handler=prepare_masks)
    assessment = sub.add_parser("compare")
    assessment.add_argument("--mask-dir",required=True)
    assessment.add_argument("--foreground",choices=("dark","bright"),required=True)
    assessment.add_argument("--thresholds",type=float,nargs="+",default=[.50])
    assessment.add_argument("--common-ids",action="store_true")
    assessment.set_defaults(handler=compare)
    submit = sub.add_parser("submit")
    submit.add_argument("--template",required=True)
    submit.add_argument("--order",choices=("C","F"),required=True)
    submit.add_argument("--start",type=int,choices=(0,1),required=True)
    submit.add_argument("--id-column",default="Id")
    submit.add_argument("--target-column",required=True)
    submit.add_argument("--threshold",type=float,default=.50)
    submit.add_argument("--postprocess",choices=("none","closing","short-gap"),default="none")
    submit.set_defaults(handler=export)
    for action in (assessment,submit):
        action.add_argument("--inputs",nargs="+",required=True)
        action.add_argument("--weights",nargs="+",type=float)
        action.add_argument("--out",required=True)
    args = p.parse_args()
    cv2.setNumThreads(0)
    args.handler(args)


if __name__ == "__main__":
    main()
