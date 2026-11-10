"""Dice counts, validated run-length encoding, and mask postprocessing."""
import numpy as np


def overlap(pred, truth):
    p, t = np.asarray(pred, dtype=bool), np.asarray(truth, dtype=bool)
    if p.shape != t.shape:
        raise ValueError(f"Mask shapes differ: {p.shape}, {t.shape}")
    tp = np.count_nonzero(p & t)
    fp = np.count_nonzero(p & ~t)
    fn = np.count_nonzero(~p & t)
    denominator = 2 * tp + fp + fn
    return dict(dice=2 * tp / denominator if denominator else 1.,
                precision=tp / (tp + fp) if tp + fp else 0.,
                recall=tp / (tp + fn) if tp + fn else 0., fp=fp, fn=fn)


def rle_encode(mask, order, start):
    values = np.asarray(mask, dtype=np.uint8).ravel(order=order)
    edges = np.flatnonzero(np.diff(np.pad(values, 1)))
    pairs = zip(edges[::2] + start, edges[1::2] - edges[::2])
    return " ".join(str(n) for pair in pairs for n in pair)


def rle_decode(rle, shape, order, start):
    mask = np.zeros(np.prod(shape), dtype=bool)
    values = [int(token) for token in rle.split()]
    if len(values) % 2:
        raise ValueError("Incomplete RLE run")
    previous_end = 0
    for beginning, length in zip(values[::2], values[1::2]):
        index = beginning - start
        if index < previous_end or length <= 0 or index + length > mask.size:
            raise ValueError("Invalid or overlapping RLE run")
        mask[index:index+length] = True
        previous_end = index + length
    return mask.reshape(shape, order=order)
