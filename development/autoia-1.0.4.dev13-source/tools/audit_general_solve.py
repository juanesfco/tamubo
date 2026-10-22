#!/usr/bin/env python3
"""Sampled audit for AutoIA Step-5 polynomial semantic solves.

This script is diagnostic rather than part of the proof.  It generates stable
2x2 quadratic parameter-dependent systems, certifies each family with AutoIA,
and samples ordinary floating-point truth solutions throughout the input box to
check inclusion in the returned rigorous enclosure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

# Prefer the source tree containing this audit script over any previously
# installed AutoIA checkpoint.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoia import autoia, Interval


def make_model(A0, A1, A2, A11, A12, A22,
               b0, b1, b2, b11, b12, b22):
    def model(p):
        x, y = p
        A = (
            A0 + x*A1 + y*A2 + (x*x)*A11
            + (x*y)*A12 + (y*y)*A22
        )
        b = (
            b0 + x*b1 + y*b2 + (x*x)*b11
            + (x*y)*b12 + (y*y)*b22
        )
        return np.linalg.solve(A, b)
    return model


def eval_truth(x, y, A0, A1, A2, A11, A12, A22,
               b0, b1, b2, b11, b12, b22):
    A = A0 + x*A1 + y*A2 + x*x*A11 + x*y*A12 + y*y*A22
    b = b0 + x*b1 + y*b2 + x*x*b11 + x*y*b12 + y*y*b22
    return np.linalg.solve(A, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260829)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    box = [Interval(-0.3, 0.3), Interval(-0.3, 0.3)]
    checks = 0
    midpoint_selected = 0

    for case in range(args.cases):
        # Strongly diagonally dominant center plus modest polynomial variation.
        A0 = np.array([[2.4, 0.12], [-0.08, 1.9]])
        A0 = A0 + rng.normal(scale=0.03, size=(2, 2))
        scales = [0.10, 0.10, 0.08, 0.06, 0.08]
        mats = [rng.normal(scale=s, size=(2, 2)) for s in scales]
        A1, A2, A11, A12, A22 = mats

        b0 = np.array([1.0, 0.45]) + rng.normal(scale=0.03, size=2)
        vecs = [rng.normal(scale=s, size=2) for s in (0.10, 0.10, 0.06, 0.05, 0.06)]
        b1, b2, b11, b12, b22 = vecs

        model = make_model(
            A0, A1, A2, A11, A12, A22,
            b0, b1, b2, b11, b12, b22,
        )
        R = autoia(model, box, order=4)
        cert = R.certificates[0]
        if cert.get("solve_mode") != "polynomial-general":
            raise AssertionError(f"case {case}: did not use polynomial-general solve")
        if cert.get("selected_preconditioner") == "family-hull-midpoint":
            midpoint_selected += 1

        for _ in range(args.samples):
            x, y = rng.uniform(-0.3, 0.3, size=2)
            truth = eval_truth(
                x, y,
                A0, A1, A2, A11, A12, A22,
                b0, b1, b2, b11, b12, b22,
            )
            if np.any(truth < R.lower) or np.any(truth > R.upper):
                raise AssertionError(
                    f"case {case}, point {(x, y)}: truth {truth} not in "
                    f"[{R.lower}, {R.upper}]"
                )
            checks += len(truth)

    print(f"General polynomial solve sampled checks passed: {checks:,}")
    print(f"Random system families: {args.cases}")
    print(f"Samples per family: {args.samples}")
    print(f"Family-hull midpoint preconditioner selected: {midpoint_selected}")


if __name__ == "__main__":
    main()
