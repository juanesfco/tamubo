#!/usr/bin/env python3
"""Sampled diagnostic audit for the Step-7 shared-generator representation.

This is not the proof.  It independently evaluates ordinary floating-point truth
models at random physical parameter points and checks containment in AutoIA's
certified output hulls.  It also records the fan-out cancellation benefit of
stable generator provenance.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoia import autoia, Interval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", type=int, default=20)
    ap.add_argument("--samples", type=int, default=200)
    args = ap.parse_args()
    rng = np.random.default_rng(20260829)

    checks = 0
    cancellation_improvements = 0
    max_generators = 0
    for _ in range(args.families):
        a = rng.uniform(-0.5, 0.5)
        b = rng.uniform(-0.4, 0.4)
        c = rng.uniform(0.1, 0.6)

        def model(p):
            x, y = p
            z = x*x + a*x*y + b*y + c
            w = z*(1.0 + 0.2*x) + np.sin(z)
            return np.array([z, w, w-w], dtype=object)

        box = [Interval(-0.8, 0.9), Interval(-0.6, 0.7)]
        gen = autoia(model, box, order=2, representation="generator")
        poly = autoia(model, box, order=2, representation="polynomial")
        max_generators = max(max_generators, len(gen.generator_ids))
        if gen.width[2] < poly.width[2]:
            cancellation_improvements += 1
        if not (gen.lower[2] <= 0.0 <= gen.upper[2]):
            raise AssertionError("generator cancellation output does not contain zero")

        for _ in range(args.samples):
            x = rng.uniform(-0.8, 0.9)
            y = rng.uniform(-0.6, 0.7)
            z = x*x + a*x*y + b*y + c
            w = z*(1.0 + 0.2*x) + math.sin(z)
            truth = (z, w, 0.0)
            for j, value in enumerate(truth):
                if not (gen.lower[j] <= value <= gen.upper[j]):
                    raise AssertionError((j, value, gen.lower[j], gen.upper[j]))
                checks += 1

    print(f"Shared-generator sampled checks passed: {checks:,}")
    print(f"Random model families: {args.families}")
    print(f"Samples per family: {args.samples}")
    print(f"Fan-out cancellation improved vs P+R: {cancellation_improvements}/{args.families}")
    print(f"Maximum active generators observed: {max_generators}")


if __name__ == "__main__":
    main()
