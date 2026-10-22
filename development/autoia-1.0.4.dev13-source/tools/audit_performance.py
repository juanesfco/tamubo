#!/usr/bin/env python3
"""Reproducible Step-9 performance and soundness audit."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoia import autoia, Interval, RigorousBinary64Backend
from autoia.trusted import rigorous_dot_point
from autoia.polynomial import zero_alpha, multi_indices_exact_degree
from autoia.polynomial.algebra import rigorous_poly_mul_truncated
from autoia.semantic.linear_solve import build_solution_polynomials_general
from tests.support.exact_binary64 import exact_float, interval_contains_exact


def exact_dot(a, b):
    return sum((exact_float(x) * exact_float(y) for x, y in zip(a, b)), Fraction(0, 1))


def best_time(fn, repeats=3):
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); vals.append(time.perf_counter() - t0)
    return min(vals)


class CountingSolve:
    def __init__(self): self.calls = 0
    def __call__(self, A, b):
        self.calls += 1
        return np.linalg.solve(A, b)


def predictor_batch_audit(rng):
    m, order, n = 5, 5, 80
    z = zero_alpha(m)
    M = rng.normal(size=(n, n))
    A0 = M.T @ M + 10.0 * np.eye(n)
    A = {z: A0}
    for i in range(m):
        alpha = tuple(1 if j == i else 0 for j in range(m))
        A[alpha] = 0.002 * rng.normal(size=(n, n))
    b = {}
    for degree in range(order + 1):
        for alpha in multi_indices_exact_degree(m, degree):
            b[alpha] = 0.01 * rng.normal(size=n)
    counter = CountingSolve()
    t0 = time.perf_counter()
    coeff = build_solution_polynomials_general(A, b, m, order, original_solve=counter)
    dt = time.perf_counter() - t0
    return counter.calls, len(coeff), dt


def polynomial_cache_benchmark(rng):
    m = 6; h = np.full(m, 0.5)
    alphas = []
    def rec(prefix, left, k):
        if k == 1:
            alphas.append(tuple(prefix + [left])); return
        for i in range(left + 1): rec(prefix + [i], left - i, k - 1)
    for degree in range(4): rec([], degree, m)
    rng.shuffle(alphas)
    p = {a: float(rng.normal()) for a in alphas[:55]}
    q = {a: float(rng.normal()) for a in alphas[10:65]}

    no_cache = RigorousBinary64Backend(reduction_method="reference")
    no_cache._monomial_magnitude_cache = None
    cached = RigorousBinary64Backend(reduction_method="reference")
    # Warm the support plan and cached monomial magnitudes.
    rigorous_poly_mul_truncated(p, q, 4, h, cached)
    t_no = best_time(lambda: rigorous_poly_mul_truncated(p, q, 4, h, no_cache), 3)
    t_cache = best_time(lambda: rigorous_poly_mul_truncated(p, q, 4, h, cached), 3)
    return t_no, t_cache


def make_solve_model(n):
    def model(p):
        x, y = p
        A = np.zeros((n, n)); b = np.zeros(n)
        for i in range(n):
            A[i, i] = 3.0 + 0.03*x*x + 0.01*(i + 1)/n*y
            if i + 1 < n:
                A[i, i + 1] = 0.02*x
                A[i + 1, i] = -0.015*y
            b[i] = 1.0 + 0.005*(i + 1)*x*y + 0.002*y*y
        return np.linalg.solve(A, b)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-cases", type=int, default=1000)
    parser.add_argument("--end-to-end-size", type=int, default=28)
    parser.add_argument("--seed", type=int, default=2026082913)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # Exact-rational vectorized reduction soundness.
    fast = RigorousBinary64Backend(reduction_method="vectorized")
    for _ in range(args.exact_cases):
        n = int(rng.integers(16, 513))
        a = rng.normal(size=n); b = rng.normal(size=n)
        got = rigorous_dot_point(a, b, fast)
        if not interval_contains_exact(got, exact_dot(a, b)):
            raise AssertionError(("vectorized dot containment", n, got))

    print(f"Vectorized exact-rational dot checks passed: {args.exact_cases:,}")
    print("Dot kernel speedups (reference/vectorized):")
    for n in (64, 256, 1024):
        a = rng.normal(size=n); b = rng.normal(size=n)
        ref = RigorousBinary64Backend(reduction_method="reference")
        vec = RigorousBinary64Backend(reduction_method="vectorized")
        reps = max(10, int(12000 / n))
        tr = best_time(lambda: [rigorous_dot_point(a, b, ref) for _ in range(reps)], 2)
        tv = best_time(lambda: [rigorous_dot_point(a, b, vec) for _ in range(reps)], 2)
        print(f"  n={n:4d}: {tr/tv:8.2f}x  reference={tr:.6f}s vectorized={tv:.6f}s")

    calls, coeffs, pred_time = predictor_batch_audit(rng)
    print(f"Batched predictor solves: {calls} factorization calls for {coeffs} coefficients; time={pred_time:.6f}s")

    t_no, t_cache = polynomial_cache_benchmark(rng)
    print(f"Repeated sparse polynomial multiply cache speedup: {t_no/t_cache:.2f}x  uncached={t_no:.6f}s cached={t_cache:.6f}s")

    n = args.end_to_end_size
    model = make_solve_model(n)
    box = [Interval(-0.5, 0.5), Interval(-0.4, 0.4)]
    bref = RigorousBinary64Backend(reduction_method="reference")
    bauto = RigorousBinary64Backend(reduction_method="auto")
    t0 = time.perf_counter(); rref = autoia(model, box, order=2, backend=bref); tref = time.perf_counter()-t0
    t0 = time.perf_counter(); rauto = autoia(model, box, order=2, backend=bauto); tauto = time.perf_counter()-t0
    # Both are independently rigorous.  The benchmark also checks that this
    # structured family happens to retain the same final endpoints.
    same = np.array_equal(rref.lower.view(np.uint64), rauto.lower.view(np.uint64)) and np.array_equal(rref.upper.view(np.uint64), rauto.upper.view(np.uint64))
    print(f"End-to-end {n}x{n} nonlinear solve: reference={tref:.6f}s auto={tauto:.6f}s speedup={tref/tauto:.2f}x")
    print(f"End-to-end endpoints bit-identical in this benchmark: {same}")
    print(f"Auto reduction stats: {bauto.reduction_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
