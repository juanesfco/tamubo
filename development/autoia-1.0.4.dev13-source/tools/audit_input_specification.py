#!/usr/bin/env python3
"""Independent exact-rational audit for AutoIA 1.0.1 input specification.

This script does not participate in certification.  It checks the public
``Interval.enclosing`` constructor against Python's exact ``Fraction`` arithmetic
for deterministic edge cases and randomized rational endpoints.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import random
import sys

from autoia import Interval


def fq(x: float) -> Fraction:
    return Fraction.from_float(float(x))


def check(qlo: Fraction, qhi: Fraction):
    I = Interval.enclosing(qlo, qhi)
    if not (fq(I.lo) <= qlo <= qhi <= fq(I.hi)):
        raise AssertionError((qlo, qhi, I, fq(I.lo), fq(I.hi)))
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-cases", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    checks = 0
    edge = [
        Fraction(0, 1),
        Fraction(1, 10), Fraction(-1, 10),
        Fraction(3, 10), Fraction(-3, 10),
        Fraction(1, 3), Fraction(-1, 3),
        Fraction(1, 2), Fraction(-1, 2),
        Fraction(1, 10**324), Fraction(-1, 10**324),
        Fraction(sys.float_info.max.as_integer_ratio()[0], sys.float_info.max.as_integer_ratio()[1]),
    ]
    for q in edge:
        checks += check(q, q)
    for a in edge:
        for b in edge:
            if a <= b:
                checks += check(a, b)

    rng = random.Random(args.seed)
    maxq = Fraction.from_float(sys.float_info.max)
    for _ in range(args.random_cases):
        # Wide exponent spread, including subnormal-scale rationals, while staying
        # inside finite binary64 range.  Exact integer arithmetic defines every
        # target endpoint.
        exp = rng.randint(-1075, 1023)
        numerator = rng.randint(-10**6, 10**6)
        denominator = rng.randint(1, 10**6)
        if exp >= 0:
            q1 = Fraction(numerator * (2**exp), denominator)
        else:
            q1 = Fraction(numerator, denominator * (2**(-exp)))
        exp2 = rng.randint(-1075, 1023)
        numerator2 = rng.randint(-10**6, 10**6)
        denominator2 = rng.randint(1, 10**6)
        if exp2 >= 0:
            q2 = Fraction(numerator2 * (2**exp2), denominator2)
        else:
            q2 = Fraction(numerator2, denominator2 * (2**(-exp2)))
        if abs(q1) > maxq or abs(q2) > maxq:
            continue
        lo, hi = sorted((q1, q2))
        checks += check(lo, hi)

    print(f"PASS: {checks:,} exact-rational input-enclosure checks")


if __name__ == "__main__":
    main()
