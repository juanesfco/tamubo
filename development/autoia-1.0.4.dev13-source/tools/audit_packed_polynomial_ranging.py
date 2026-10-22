#!/usr/bin/env python3
"""Independent exact-rational audit for Step-9d packed polynomial ranging."""

from __future__ import annotations

from fractions import Fraction
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoia.polynomial import (
    pack_point_polynomial_family,
    pack_interval_polynomial_family,
    range_packed_point_termwise,
    range_packed_interval_termwise,
)
from autoia.trusted import (
    Interval, RigorousBinary64Backend, rigorous_interval_subtract_point_arrays,
)


def F(x):
    return Fraction.from_float(float(x))


def exact_product(a_lo, a_hi, b_lo, b_hi):
    values = (a_lo*b_lo, a_lo*b_hi, a_hi*b_lo, a_hi*b_hi)
    return min(values), max(values)


def monomial_range(alpha, h):
    H = Fraction(1, 1)
    for ai, hi in zip(alpha, h):
        H *= F(hi) ** int(ai)
    if sum(alpha) == 0:
        return Fraction(1, 1), Fraction(1, 1)
    if all((int(ai) % 2) == 0 for ai in alpha):
        return Fraction(0, 1), H
    return -H, H


def exact_point_termwise(coeff, support, h):
    lo = Fraction(0, 1); hi = Fraction(0, 1)
    for c, alpha in zip(coeff, support):
        ml, mh = monomial_range(alpha, h)
        pl, ph = exact_product(F(c), F(c), ml, mh)
        lo += pl; hi += ph
    return lo, hi


def exact_interval_termwise(clo, chi, support, h):
    lo = Fraction(0, 1); hi = Fraction(0, 1)
    for al, ah, alpha in zip(clo, chi, support):
        ml, mh = monomial_range(alpha, h)
        pl, ph = exact_product(F(al), F(ah), ml, mh)
        lo += pl; hi += ph
    return lo, hi


def encloses(lo, hi, elo, ehi):
    return F(lo) <= elo and F(hi) >= ehi


def main():
    rng = np.random.default_rng(2026082904)
    py_rng = random.Random(2026082904)
    backend = RigorousBinary64Backend(
        reduction_method="vectorized", vectorized_reduction_min_terms=1
    )

    point_cases = 0
    interval_cases = 0
    subtraction_cases = 0

    # Coefficient-error decomposition depends on rigorous I - representative.
    for _ in range(100):
        lo = rng.normal(size=(5, 7))
        hi = lo + rng.uniform(0.0, 1.0e-7, size=(5, 7))
        rep = np.minimum(np.maximum(0.5 * lo + 0.5 * hi, lo), hi)
        elo, ehi = rigorous_interval_subtract_point_arrays(lo, hi, rep, backend)
        for idx in np.ndindex(lo.shape):
            exact_lo = F(lo[idx]) - F(rep[idx])
            exact_hi = F(hi[idx]) - F(rep[idx])
            if not encloses(elo[idx], ehi[idx], exact_lo, exact_hi):
                raise AssertionError(("subtract-point", idx, exact_lo, exact_hi, elo[idx], ehi[idx]))
            subtraction_cases += 1
    for _ in range(80):
        m = py_rng.randint(1, 4)
        h = rng.uniform(1.0e-4, 0.8, size=m)
        support = {(0,) * m}
        while len(support) < py_rng.randint(4, 22):
            alpha = tuple(py_rng.randint(0, 4) for _ in range(m))
            if sum(alpha) <= 5:
                support.add(alpha)
        support = sorted(support)
        family_shape = (py_rng.randint(1, 4), py_rng.randint(1, 4))

        point_coeff = {a: rng.normal(size=family_shape) for a in support}
        pp = pack_point_polynomial_family(point_coeff, m)
        plo, phi = range_packed_point_termwise(pp, h, backend)
        for idx in np.ndindex(family_shape):
            coeff = [point_coeff[a][idx] for a in pp.plan.support]
            elo, ehi = exact_point_termwise(coeff, pp.plan.support, h)
            if not encloses(plo[idx], phi[idx], elo, ehi):
                raise AssertionError(("point", m, idx, elo, ehi, plo[idx], phi[idx]))
            point_cases += 1

        interval_coeff = {}
        for a in support:
            center = rng.normal(size=family_shape)
            radius = rng.uniform(0.0, 5.0e-10, size=family_shape)
            arr = np.empty(family_shape, dtype=object)
            for idx in np.ndindex(family_shape):
                arr[idx] = Interval(center[idx] - radius[idx], center[idx] + radius[idx])
            interval_coeff[a] = arr
        ip = pack_interval_polynomial_family(interval_coeff, m)
        ilo, ihi = range_packed_interval_termwise(ip, h, backend)
        for idx in np.ndindex(family_shape):
            clo = [ip.lower[idx + (t,)] for t in range(ip.plan.term_count)]
            chi = [ip.upper[idx + (t,)] for t in range(ip.plan.term_count)]
            elo, ehi = exact_interval_termwise(clo, chi, ip.plan.support, h)
            if not encloses(ilo[idx], ihi[idx], elo, ehi):
                raise AssertionError(("interval", m, idx, elo, ehi, ilo[idx], ihi[idx]))
            interval_cases += 1

    print("Step 9d packed polynomial exact-rational audit: PASS")
    print(f"point_family_outputs_checked={point_cases}")
    print(f"interval_family_outputs_checked={interval_cases}")
    print(f"interval_minus_point_checks={subtraction_cases}")
    print(f"total_exact_enclosures_checked={point_cases + interval_cases + subtraction_cases}")
    print("reduction_stats=", backend.reduction_stats)


if __name__ == "__main__":
    main()
