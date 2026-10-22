#!/usr/bin/env python3
"""Independent high-precision audit for erf and standard-normal primitives."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mpmath as mp

from autoia.trusted import (
    RigorousBinary64Backend,
    erf_point_interval,
    standard_normal_pdf_point_interval,
    standard_normal_cdf_point_interval,
)


def mp_exact_float(x: float):
    n, d = float(x).as_integer_ratio()
    return mp.mpf(n) / mp.mpf(d)


def contains(I, y) -> bool:
    lo = mp_exact_float(I.lo) if math.isfinite(I.lo) else mp.ninf
    hi = mp_exact_float(I.hi) if math.isfinite(I.hi) else mp.inf
    return lo <= y <= hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--dps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--boxes", type=int, default=30)
    args = ap.parse_args()

    mp.mp.dps = args.dps
    rng = random.Random(args.seed)
    backend = RigorousBinary64Backend()
    passed = 0

    fixed = [-10.0, -8.0, -5.0, -4.0, -3.0, -1.0, -0.0, 0.0, 0.1, 1.0, 3.0, 4.0, 5.0, 8.0, 10.0]
    xs = fixed + [rng.uniform(-10.0, 10.0) for _ in range(args.cases)]

    sqrt2pi = mp.sqrt(2 * mp.pi)
    sqrt2 = mp.sqrt(2)
    for x in xs:
        xx = mp_exact_float(x)
        checks = (
            (erf_point_interval(x, backend), mp.erf(xx), "erf"),
            (standard_normal_pdf_point_interval(x, backend), mp.e**(-xx*xx/2) / sqrt2pi, "normal_pdf"),
            (standard_normal_cdf_point_interval(x, backend), (1 + mp.erf(xx/sqrt2))/2, "normal_cdf"),
        )
        for I, truth, name in checks:
            if not contains(I, truth):
                raise AssertionError(f"{name} containment failed at x={x!r}: {I}, truth={truth}")
            passed += 1

    # Representation-level audit of the Taylor lifting on random 1D boxes.
    import numpy as np
    from autoia import autoia, Interval as PublicInterval, erf as autoia_erf, standard_normal_pdf, standard_normal_cdf

    specs = (
        (autoia_erf, lambda z: mp.erf(z), "erf"),
        (standard_normal_pdf, lambda z: mp.e**(-z*z/2) / sqrt2pi, "normal_pdf"),
        (standard_normal_cdf, lambda z: (1 + mp.erf(z/sqrt2))/2, "normal_cdf"),
    )
    box_checks = 0
    for _ in range(args.boxes):
        a = rng.uniform(-3.0, 2.0)
        b = a + rng.uniform(0.02, 1.25)
        for fn, truth_fn, name in specs:
            def model(p, fn=fn):
                return np.array([fn(p[0])], dtype=object)
            result = autoia(model, [PublicInterval(a, b)], order=8)
            lo = mp_exact_float(float(result.lower[0]))
            hi = mp_exact_float(float(result.upper[0]))
            for j in range(17):
                xf = a + (b-a) * j / 16.0
                y = truth_fn(mp_exact_float(float(xf)))
                if not (lo <= y <= hi):
                    raise AssertionError(
                        f"polynomial {name} containment failed on [{a},{b}] at x={xf}: "
                        f"[{result.lower[0]}, {result.upper[0]}], truth={y}"
                    )
                box_checks += 1

    print(f"Special-function high-precision scalar checks passed: {passed}")
    print(f"Polynomial high-precision sample checks passed: {box_checks}")
    print(f"Random scalar points: {args.cases}")
    print(f"Random polynomial boxes: {args.boxes}")
    print(f"mpmath precision: {args.dps} decimal digits")


if __name__ == "__main__":
    main()
