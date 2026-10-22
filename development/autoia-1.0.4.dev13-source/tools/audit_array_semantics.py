#!/usr/bin/env python3
"""Independent sampled audit of Step-4 array semantics.

This is diagnostic evidence, not the proof itself.  Random affine/bilinear models
are executed through AutoIA and compared with many ordinary floating-point
samples of the same mathematical expressions.
"""

from __future__ import annotations

from pathlib import Path
import sys

import argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoia import autoia, Interval


def build_model(M, B, c):
    def model(p):
        x = np.array(p)
        # Broadcasting, matrix-vector multiplication, reduction, product, norm.
        y = np.matmul(M, x) + B[:, 0] * x[0] + c
        s = np.sum(y)
        m = np.mean(y)
        d = np.dot(y, np.arange(1.0, len(y) + 1.0))
        pr = np.prod(1.0 + 0.05 * y)
        nrm = np.linalg.norm(y)
        return np.array([s, m, d, pr, nrm])
    return model


def exact_eval(M, B, c, x):
    y = M @ x + B[:, 0] * x[0] + c
    return np.array([
        np.sum(y), np.mean(y), np.dot(y, np.arange(1.0, len(y)+1.0)),
        np.prod(1.0 + 0.05*y), np.linalg.norm(y),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=int, default=50)
    ap.add_argument('--samples', type=int, default=200)
    ap.add_argument('--seed', type=int, default=4701)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    checks = 0
    for case in range(args.cases):
        m = int(rng.integers(2, 5))
        n = int(rng.integers(2, 5))
        centers = rng.uniform(-0.4, 0.6, size=m)
        half = rng.uniform(0.02, 0.18, size=m)
        boxes = [Interval(float(c-h), float(c+h)) for c, h in zip(centers, half)]
        M = rng.normal(scale=0.35, size=(n, m))
        B = rng.normal(scale=0.1, size=(n, m))
        c = rng.normal(scale=0.2, size=n)
        model = build_model(M, B, c)
        R = autoia(model, boxes, order=4)
        for _ in range(args.samples):
            x = np.array([rng.uniform(I.lo, I.hi) for I in boxes])
            y = exact_eval(M, B, c, x)
            if not (np.all(R.lower <= y) and np.all(y <= R.upper)):
                raise AssertionError(
                    f'Containment failure case={case}: x={x}, y={y}, '
                    f'lower={R.lower}, upper={R.upper}'
                )
            checks += len(y)

    print(f'Array-semantics sampled checks passed: {checks:,}')
    print(f'Random model cases: {args.cases}')
    print(f'Samples per case: {args.samples}')


if __name__ == '__main__':
    main()
