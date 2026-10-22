#!/usr/bin/env python3
"""Randomized diagnostic audit for AutoIA path-aware execution.

This is sampled validation evidence, not the inclusion proof.  Each randomly
parameterized model is certified by AutoIA, then ordinary floating-point truth
is sampled densely enough to catch implementation/dispatch mistakes.
"""

from __future__ import annotations

from pathlib import Path
import sys

import argparse
import math
import random
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoia import autoia, Interval, BranchPolicy


def make_models(rng):
    a = rng.uniform(0.25, 0.8)
    b = rng.uniform(-0.6, 0.6)
    c = rng.uniform(-0.4, 0.4)
    d = rng.uniform(-0.2, 0.2)
    t1 = rng.uniform(-0.15, 0.25)
    t2 = rng.uniform(-0.25, 0.25)

    def validated(p):
        x, y = p
        g = a*x*x + b*y + c*x + d
        if g > t1:
            z = np.exp(0.2*x) + 0.3*y
            if x + y <= t2:
                return z + 0.1*x*y
            return z*z - 0.2*x
        else:
            z = np.cos(0.3*y) - 0.4*x
            if x - y >= -t2:
                return z + 0.15*y*y
            return z*z + 0.1*x

    def truth(x, y):
        g = a*x*x + b*y + c*x + d
        if g > t1:
            z = math.exp(0.2*x) + 0.3*y
            if x + y <= t2:
                return z + 0.1*x*y
            return z*z - 0.2*x
        else:
            z = math.cos(0.3*y) - 0.4*x
            if x - y >= -t2:
                return z + 0.15*y*y
            return z*z + 0.1*x

    return validated, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", type=int, default=12)
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--seed", type=int, default=41721)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    checks = 0
    total_branches = 0
    max_branches = 0
    for family in range(args.families):
        model, truth = make_models(rng)
        result = autoia(
            model,
            [Interval(-0.6, 0.6), Interval(-0.6, 0.6)],
            order=5,
            branch_policy=BranchPolicy(
                max_branches=128,
                max_path_decisions=24,
                max_split_depth_per_decision=1,
                max_total_splits=24,
            ),
        )
        nbranches = len(getattr(result, "branch_results", [result]))
        total_branches += nbranches
        max_branches = max(max_branches, nbranches)
        lo = float(result.lower[0])
        hi = float(result.upper[0])
        for _ in range(args.samples):
            x = rng.uniform(-0.6, 0.6)
            y = rng.uniform(-0.6, 0.6)
            f = truth(x, y)
            if not (lo <= f <= hi):
                raise AssertionError(
                    f"containment failure family={family} x={x} y={y} "
                    f"truth={f} enclosure=[{lo},{hi}]"
                )
            checks += 1

    print(f"Path-aware sampled checks passed: {checks:,}")
    print(f"Random model families: {args.families}")
    print(f"Samples per family: {args.samples}")
    print(f"Mean certified paths: {total_branches/args.families:.2f}")
    print(f"Maximum certified paths: {max_branches}")


if __name__ == "__main__":
    main()
