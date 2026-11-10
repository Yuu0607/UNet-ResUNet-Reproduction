"""Combine aligned OOF or test point predictions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def combine(inputs, weights):
    keys, total = None, None
    for path, weight in zip(inputs, weights):
        table = pd.read_csv(path, dtype={"ID": str}).set_index("ID", verify_integrity=True)
        if keys is None:
            keys = table.index
        if set(table.index) != set(keys):
            raise ValueError("Only blend files covering the same IDs")
        values = table.loc[keys, ["Pred_X", "Pred_Y"]].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite coordinates")
        total = weight * values if total is None else total + weight * values
    return pd.DataFrame(total, index=keys, columns=["Pred_X","Pred_Y"]).reset_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True, nargs="+")
    p.add_argument("--weights", type=float, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--template", help="Use with test coordinates to additionally export submission.csv")
    args = p.parse_args()
    w = np.array(args.weights or [1.] * len(args.inputs))
    if len(w) != len(args.inputs) or not np.isfinite(w).all() or min(w) < 0 or w.sum() <= 0:
        p.error("Supply one non-negative weight per file")
    table = combine(args.inputs, w/w.sum())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    if args.template:
        from run import submission
        submission(table, args.template, out.parent)


if __name__ == "__main__":
    main()
