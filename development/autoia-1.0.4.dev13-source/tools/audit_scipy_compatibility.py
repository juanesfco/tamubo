#!/usr/bin/env python3
"""Sampled compatibility audit for AutoIA's SciPy normal-distribution dispatch.

SciPy is used only to supply the familiar user-facing syntax and independent
numeric sample values.  The AutoIA proof path is routed to AutoIA's own trusted
normal/erf kernels.
"""

from __future__ import annotations

from pathlib import Path
import sys

import argparse
import random

import numpy as np
from scipy import special, stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoia import Interval, autoia


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    checks = 0

    for _ in range(args.cases):
        center = rng.uniform(-1.5, 1.5)
        half = rng.uniform(0.05, 0.8)
        lo, hi = center - half, center + half
        mu = rng.uniform(-0.8, 0.8)
        sigma = rng.uniform(0.5, 2.0)

        def model(p):
            x = p[0]
            rv = stats.norm(loc=mu, scale=sigma)
            return np.array([
                special.erf(x),
                special.ndtr(x),
                stats.norm.pdf(x, loc=mu, scale=sigma),
                stats.norm.cdf(x, loc=mu, scale=sigma),
                rv.pdf(x),
                rv.cdf(x),
            ])

        result = autoia(model, [Interval(lo, hi)], order=8)
        for x in np.linspace(lo, hi, args.samples):
            truth = np.array([
                special.erf(x),
                special.ndtr(x),
                stats.norm.pdf(x, loc=mu, scale=sigma),
                stats.norm.cdf(x, loc=mu, scale=sigma),
                stats.norm(loc=mu, scale=sigma).pdf(x),
                stats.norm(loc=mu, scale=sigma).cdf(x),
            ], dtype=float)
            if np.any(truth < result.lower) or np.any(truth > result.upper):
                raise AssertionError(
                    f"containment failure: box=({lo},{hi}), mu={mu}, sigma={sigma}, x={x}, "
                    f"truth={truth}, lower={result.lower}, upper={result.upper}"
                )
            checks += truth.size

    print(f"SciPy compatibility sampled checks passed: {checks:,}")
    print(f"Random boxes: {args.cases}")
    print(f"Samples per box: {args.samples}")


if __name__ == "__main__":
    main()
