#!/usr/bin/env python3
"""Exact-rational audit for Step 9d phase-2 derivative/monotonicity kernels."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fractions import Fraction
import itertools
import numpy as np

from autoia.trusted import RigorousBinary64Backend
from autoia.polynomial import (
    pack_point_polynomial_family,
    range_packed_point_derivatives,
    range_packed_point_monotonicity,
)


def F(x):
    return Fraction.from_float(float(x))


def interval_mul(a_lo, a_hi, b_lo, b_hi):
    values = (a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi)
    return min(values), max(values)


def monomial_symmetric(alpha, h):
    H = Fraction(1)
    for k, hj in zip(alpha, h):
        H *= F(hj) ** int(k)
    if sum(alpha) == 0:
        return Fraction(1), Fraction(1)
    if all(int(k) % 2 == 0 for k in alpha):
        return Fraction(0), H
    return -H, H


def derivative_range(coeff, support, h, variable):
    lo = Fraction(0)
    hi = Fraction(0)
    for c, alpha in zip(coeff, support):
        k = int(alpha[variable])
        if k == 0:
            continue
        beta = list(alpha)
        beta[variable] -= 1
        ml, mh = monomial_symmetric(tuple(beta), h)
        scale = F(c) * k
        pl, ph = interval_mul(scale, scale, ml, mh)
        lo += pl
        hi += ph
    return lo, hi


def variable_fixed_range(power, h, sign, lower_problem):
    power = int(power)
    if power == 0:
        return Fraction(1), Fraction(1)
    H = F(h) ** power
    if int(sign) > 0:
        x = -F(h) if lower_problem else F(h)
        y = x ** power
        return y, y
    if int(sign) < 0:
        x = F(h) if lower_problem else -F(h)
        y = x ** power
        return y, y
    if power % 2 == 0:
        return Fraction(0), H
    return -H, H


def fixed_termwise_range(coeff, support, h, signs, lower_problem):
    lo = Fraction(0)
    hi = Fraction(0)
    for c, alpha in zip(coeff, support):
        ml = Fraction(1)
        mh = Fraction(1)
        for power, hj, sign in zip(alpha, h, signs):
            vl, vh = variable_fixed_range(power, hj, sign, lower_problem)
            ml, mh = interval_mul(ml, mh, vl, vh)
        pl, ph = interval_mul(F(c), F(c), ml, mh)
        lo += pl
        hi += ph
    return lo, hi


def check_enclosure(lo, hi, exact_lo, exact_hi):
    return F(lo) <= exact_lo and F(hi) >= exact_hi


def main():
    rng = np.random.default_rng(2026083007)
    backend = RigorousBinary64Backend(
        reduction_method="vectorized", vectorized_reduction_min_terms=1
    )
    derivative_checks = 0
    fixed_range_checks = 0
    sign_checks = 0

    for m in (1, 2, 3, 4):
        candidates = [
            alpha for alpha in itertools.product(range(5), repeat=m)
            if sum(alpha) <= 4
        ]
        take = min(18, len(candidates))
        chosen = rng.choice(len(candidates), size=take, replace=False)
        support = sorted(candidates[int(j)] for j in chosen)
        if (0,) * m not in support:
            support[0] = (0,) * m
            support = sorted(set(support))
        h = rng.uniform(0.04, 0.55, size=m)
        coeff_arrays = {alpha: rng.normal(size=(5, 4)) for alpha in support}
        packed = pack_point_polynomial_family(coeff_arrays, m)

        dlo, dhi = range_packed_point_derivatives(packed, h, backend)
        rlo, rhi, meta = range_packed_point_monotonicity(packed, h, backend)

        for idx in np.ndindex((5, 4)):
            coeff = [coeff_arrays[a][idx] for a in packed.plan.support]
            signs = [int(v) for v in meta["signs"][idx]]
            for i in range(m):
                elo, ehi = derivative_range(coeff, packed.plan.support, h, i)
                if not check_enclosure(dlo[idx + (i,)], dhi[idx + (i,)], elo, ehi):
                    raise AssertionError(("derivative", m, idx, i))
                derivative_checks += 1
                if signs[i] > 0 and elo < 0:
                    raise AssertionError(("false-positive monotone increasing", m, idx, i))
                if signs[i] < 0 and ehi > 0:
                    raise AssertionError(("false-positive monotone decreasing", m, idx, i))
                sign_checks += 1

            llo, lhi = fixed_termwise_range(
                coeff, packed.plan.support, h, signs, True
            )
            ulo, uhi = fixed_termwise_range(
                coeff, packed.plan.support, h, signs, False
            )
            if not check_enclosure(
                meta["lower_fixed_range_lower"][idx],
                meta["lower_fixed_range_upper"][idx],
                llo, lhi,
            ):
                raise AssertionError(("lower-fixed", m, idx))
            fixed_range_checks += 1
            if not check_enclosure(
                meta["upper_fixed_range_lower"][idx],
                meta["upper_fixed_range_upper"][idx],
                ulo, uhi,
            ):
                raise AssertionError(("upper-fixed", m, idx))
            fixed_range_checks += 1
            if F(rlo[idx]) > llo or F(rhi[idx]) < uhi:
                raise AssertionError(("final-monotonicity", m, idx))
            fixed_range_checks += 1

    total = derivative_checks + sign_checks + fixed_range_checks
    print(f"derivative_exact_enclosures_checked={derivative_checks}")
    print(f"monotonicity_sign_implications_checked={sign_checks}")
    print(f"fixed_and_final_range_enclosures_checked={fixed_range_checks}")
    print(f"total_phase2_exact_checks={total}")
    print("status=PASS")


if __name__ == "__main__":
    main()
