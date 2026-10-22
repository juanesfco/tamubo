#!/usr/bin/env python3
"""Exact-rational audit for Step 9d phase-3 packed Bernstein kernels."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fractions import Fraction
from math import comb
import itertools
import numpy as np

from autoia.trusted import RigorousBinary64Backend
from autoia.polynomial import (
    pack_point_polynomial_family,
    bernstein_axis_transform_endpoints,
    packed_bernstein_coefficients,
    range_packed_point_monotonicity,
    range_packed_point_reduced_bernstein,
)


def F(x):
    return Fraction.from_float(float(x))


def axis_weight(degree, h, gamma, alpha):
    H = F(h)
    total = Fraction(0)
    for beta in range(min(int(alpha), int(gamma)) + 1):
        total += (
            Fraction(comb(int(alpha), beta))
            * ((-H) ** (int(alpha) - beta))
            * ((2 * H) ** beta)
            * Fraction(comb(int(gamma), beta), comb(int(degree), beta))
        )
    return total


def exact_coefficient(coeff, support, degrees, h, gamma, fixed, unresolved):
    total = Fraction(0)
    for c, alpha in zip(coeff, support):
        term = F(c)
        for i, value in fixed.items():
            term *= F(value) ** int(alpha[i])
        for ui, i in enumerate(unresolved):
            term *= axis_weight(degrees[i], h[i], gamma[ui], alpha[i])
        total += term
    return total


def encloses(lo, hi, exact):
    return F(lo) <= exact <= F(hi)


def main():
    rng = np.random.default_rng(2026083013)
    backend = RigorousBinary64Backend(
        reduction_method="vectorized", vectorized_reduction_min_terms=1
    )
    axis_checks = 0
    full_coefficient_checks = 0
    reduced_coefficient_checks = 0
    reduced_range_checks = 0

    for degree in range(7):
        for h in (0.0, 0.07, 0.37, 0.91):
            lo, hi = bernstein_axis_transform_endpoints(degree, h, backend)
            for gamma in range(degree + 1):
                for alpha in range(degree + 1):
                    exact = axis_weight(degree, h, gamma, alpha)
                    if not encloses(lo[gamma, alpha], hi[gamma, alpha], exact):
                        raise AssertionError(("axis-transform", degree, h, gamma, alpha))
                    axis_checks += 1

    for m in (1, 2, 3):
        candidates = [
            a for a in itertools.product(range(4), repeat=m) if sum(a) <= 3
        ]
        support = sorted(candidates)
        h = rng.uniform(0.05, 0.5, size=m)
        coeff_arrays = {a: rng.normal(size=4) for a in support}
        packed = pack_point_polynomial_family(coeff_arrays, m)
        lo, hi, _ = packed_bernstein_coefficients(packed, h, backend)
        for j in range(4):
            coeff = [coeff_arrays[a][j] for a in packed.plan.support]
            for gamma in itertools.product(
                *[range(d + 1) for d in packed.plan.degree_vector]
            ):
                exact = exact_coefficient(
                    coeff, packed.plan.support, packed.plan.degree_vector,
                    h, gamma, {}, tuple(range(m)),
                )
                if not encloses(lo[(j,) + gamma], hi[(j,) + gamma], exact):
                    raise AssertionError(("full-coefficient", m, j, gamma))
                full_coefficient_checks += 1

        if m >= 2:
            fixed_index = tuple(range(0, m, 2))
            fixed = {
                i: (-float(h[i]) if i % 4 == 0 else float(h[i]))
                for i in fixed_index
            }
            unresolved = tuple(i for i in range(m) if i not in fixed)
            lo, hi, _ = packed_bernstein_coefficients(
                packed, h, backend, fixed=fixed, unresolved=unresolved
            )
            gamma_space = list(itertools.product(
                *[range(packed.plan.degree_vector[i] + 1) for i in unresolved]
            )) if unresolved else [tuple()]
            for j in range(4):
                coeff = [coeff_arrays[a][j] for a in packed.plan.support]
                for gamma in gamma_space:
                    exact = exact_coefficient(
                        coeff, packed.plan.support, packed.plan.degree_vector,
                        h, gamma, fixed, unresolved,
                    )
                    if not encloses(lo[(j,) + gamma], hi[(j,) + gamma], exact):
                        raise AssertionError(("reduced-coefficient", m, j, gamma))
                    reduced_coefficient_checks += 1

    # Orchestration audit with proven monotonicity: q0 and q1 are dominated by
    # linear terms, while q2 remains unresolved because of an even quadratic.
    h = np.array([0.2, 0.25, 0.3])
    support = [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0),
        (1, 0, 1), (0, 1, 1),
    ]
    n = 12
    coeff_arrays = {a: rng.uniform(-0.02, 0.02, size=n) for a in support}
    coeff_arrays[(1, 0, 0)] = np.where(np.arange(n) % 2 == 0, 2.0, -2.0)
    coeff_arrays[(0, 1, 0)] = np.where(np.arange(n) % 3 == 0, 1.5, -1.5)
    coeff_arrays[(0, 0, 1)] = np.zeros(n)
    coeff_arrays[(0, 0, 2)] = np.full(n, 0.4)
    packed = pack_point_polynomial_family(coeff_arrays, 3)
    _, _, mm = range_packed_point_monotonicity(packed, h, backend)
    rlo, rhi, _ = range_packed_point_reduced_bernstein(
        packed, h, backend, monotonicity_meta=mm
    )
    for j in range(n):
        signs = [int(v) for v in mm["signs"][j]]
        unresolved = tuple(i for i, sign in enumerate(signs) if sign == 0)
        lower_fixed = {}
        upper_fixed = {}
        for i, sign in enumerate(signs):
            if sign > 0:
                lower_fixed[i] = -float(h[i]); upper_fixed[i] = float(h[i])
            elif sign < 0:
                lower_fixed[i] = float(h[i]); upper_fixed[i] = -float(h[i])
        gamma_space = list(itertools.product(
            *[range(packed.plan.degree_vector[i] + 1) for i in unresolved]
        )) if unresolved else [tuple()]
        coeff = [coeff_arrays[a][j] for a in packed.plan.support]
        lower_exact = []
        upper_exact = []
        for gamma in gamma_space:
            lower_exact.append(exact_coefficient(
                coeff, packed.plan.support, packed.plan.degree_vector,
                h, gamma, lower_fixed, unresolved,
            ))
            upper_exact.append(exact_coefficient(
                coeff, packed.plan.support, packed.plan.degree_vector,
                h, gamma, upper_fixed, unresolved,
            ))
        if F(rlo[j]) > min(lower_exact):
            raise AssertionError(("reduced-lower-range", j, signs))
        if F(rhi[j]) < max(upper_exact):
            raise AssertionError(("reduced-upper-range", j, signs))
        reduced_range_checks += 2

    total = axis_checks + full_coefficient_checks + reduced_coefficient_checks + reduced_range_checks
    print(f"axis_transform_exact_entries_checked={axis_checks}")
    print(f"full_bernstein_exact_coefficients_checked={full_coefficient_checks}")
    print(f"reduced_bernstein_exact_coefficients_checked={reduced_coefficient_checks}")
    print(f"reduced_bernstein_exact_face_ranges_checked={reduced_range_checks}")
    print(f"total_phase3_exact_checks={total}")
    print("status=PASS")


if __name__ == "__main__":
    main()
