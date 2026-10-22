"""Rigorous range bounds for AutoIA point polynomials.

This module contains the termwise, monotonicity, Bernstein, and
monotonicity-reduced Bernstein algorithms extracted unchanged from the
Version 0.11 reference implementation during Step 1 (0.12.0.dev4).

All proof-critical arithmetic is delegated to :mod:`autoia.trusted`.
Controller policy is deliberately outside this module; this layer only
computes sound polynomial ranges and structural metadata.
"""

from __future__ import annotations

from math import comb
import itertools

from ..trusted import Interval, point_interval, integer_interval
from .algebra import (
    monomial_range_symmetric,
    monomial_range_with_fixed,
    point_power_interval,
)

def _degree_vector_and_bernstein_cost(p, m):
    degrees = [0] * int(m)
    for alpha in p:
        for i, ai in enumerate(alpha):
            degrees[i] = max(degrees[i], int(ai))
    cost = 1
    for d in degrees:
        cost *= (d + 1)
    return tuple(degrees), int(cost)

def range_termwise_scalar(p, h, backend) -> Interval:
    total = Interval(0.0, 0.0)
    for alpha, coeff in p.items():
        M = monomial_range_symmetric(alpha, h, backend)
        term = backend.mul(point_interval(coeff), M)
        total = backend.add(total, term)
    return total


def range_polynomial_with_fixed(p, h, fixed, backend) -> Interval:
    total = Interval(0.0, 0.0)
    for alpha, coeff in p.items():
        M = monomial_range_with_fixed(alpha, h, fixed, backend)
        term = backend.mul(point_interval(coeff), M)
        total = backend.add(total, term)
    return total


def derivative_range_termwise(p, variable, h, backend) -> Interval:
    total = Interval(0.0, 0.0)
    for alpha, coeff in p.items():
        k = alpha[variable]
        if k == 0:
            continue
        beta = list(alpha)
        beta[variable] -= 1
        beta = tuple(beta)
        coeff_I = backend.mul(point_interval(coeff), integer_interval(k))
        M = monomial_range_symmetric(beta, h, backend)
        total = backend.add(total, backend.mul(coeff_I, M))
    return total


def range_monotonicity_scalar(p, h, backend):
    m = len(h)
    lower_fixed = {}
    upper_fixed = {}
    signs = []
    derivative_ranges = []

    for i in range(m):
        dr = derivative_range_termwise(p, i, h, backend)
        derivative_ranges.append(dr)
        if dr.lo >= 0.0:
            lower_fixed[i] = -float(h[i])
            upper_fixed[i] = +float(h[i])
            signs.append(+1)
        elif dr.hi <= 0.0:
            lower_fixed[i] = +float(h[i])
            upper_fixed[i] = -float(h[i])
            signs.append(-1)
        else:
            signs.append(0)

    lo_range = range_polynomial_with_fixed(p, h, lower_fixed, backend)
    hi_range = range_polynomial_with_fixed(p, h, upper_fixed, backend)
    return Interval(lo_range.lo, hi_range.hi), {
        "signs": signs,
        "derivative_ranges": derivative_ranges,
    }


def all_indices_leq(degrees):
    yield from itertools.product(*[range(d + 1) for d in degrees])


def power_coeff_unit_box_interval(p, h, backend):
    """
    Rigorous power coefficients after q_i = -h_i + 2 h_i t_i, t_i in [0,1].

    Returns beta -> Interval coefficient.
    """
    m = len(h)
    out = {}

    for alpha, coeff in p.items():
        beta_ranges = [range(k + 1) for k in alpha]
        for beta in itertools.product(*beta_ranges):
            factor = point_interval(coeff)
            for i, (ai, bi) in enumerate(zip(alpha, beta)):
                choose = comb(ai, bi)
                factor = backend.mul(factor, integer_interval(choose))

                # (-h_i)^(ai-bi)
                factor = backend.mul(
                    factor,
                    point_power_interval(-float(h[i]), ai - bi, backend),
                )

                # (2 h_i)^bi, computed as an interval before exponentiation.
                two_h = backend.mul(point_interval(2.0), point_interval(float(h[i])))
                p2h = Interval(1.0, 1.0)
                for _ in range(bi):
                    p2h = backend.mul(p2h, two_h)
                factor = backend.mul(factor, p2h)

            if beta in out:
                out[beta] = backend.add(out[beta], factor)
            else:
                out[beta] = factor

    return out


def range_bernstein_scalar(p, h, backend):
    m = len(h)
    degrees = [0] * m
    for alpha in p:
        for i, ai in enumerate(alpha):
            degrees[i] = max(degrees[i], ai)

    power = power_coeff_unit_box_interval(p, h, backend)
    bvals = []

    for gamma in all_indices_leq(degrees):
        bg = Interval(0.0, 0.0)
        for beta, coeff_I in power.items():
            if any(beta[i] > gamma[i] for i in range(m)):
                continue
            weight = Interval(1.0, 1.0)
            for i in range(m):
                if degrees[i] == 0:
                    continue
                num = comb(gamma[i], beta[i])
                den = comb(degrees[i], beta[i])
                ratio = backend.div(integer_interval(num), integer_interval(den))
                weight = backend.mul(weight, ratio)
            bg = backend.add(bg, backend.mul(weight, coeff_I))
        bvals.append(bg)

    if not bvals:
        return Interval(0.0, 0.0), {
            "degree_vector": tuple(degrees),
            "num_coefficients": 1,
        }

    return Interval(
        min(v.lo for v in bvals),
        max(v.hi for v in bvals),
    ), {
        "degree_vector": tuple(degrees),
        "num_coefficients": len(bvals),
    }



def power_coeff_reduced_unit_box_interval(p, h, fixed, unresolved, backend):
    """Rigorous power coefficients after fixing proven-monotone variables.

    ``fixed`` maps original variable index -> exact binary64 endpoint.  Only
    variables in ``unresolved`` are transformed to unit-box coordinates.  The
    returned coefficient dictionary uses multi-indices of length
    ``len(unresolved)`` and interval coefficients, because folding fixed
    variables into point coefficients is itself proof-critical arithmetic.
    """
    unresolved = tuple(int(i) for i in unresolved)
    out = {}

    for alpha, coeff in p.items():
        base = point_interval(coeff)
        for i, value in fixed.items():
            base = backend.mul(
                base,
                point_power_interval(float(value), int(alpha[i]), backend),
            )

        beta_ranges = [range(int(alpha[i]) + 1) for i in unresolved]
        if not beta_ranges:
            beta_iter = [tuple()]
        else:
            beta_iter = itertools.product(*beta_ranges)

        for beta in beta_iter:
            factor = base
            for ui, i in enumerate(unresolved):
                ai = int(alpha[i])
                bi = int(beta[ui])
                factor = backend.mul(factor, integer_interval(comb(ai, bi)))
                factor = backend.mul(
                    factor,
                    point_power_interval(-float(h[i]), ai - bi, backend),
                )
                two_h = backend.mul(point_interval(2.0), point_interval(float(h[i])))
                p2h = Interval(1.0, 1.0)
                for _ in range(bi):
                    p2h = backend.mul(p2h, two_h)
                factor = backend.mul(factor, p2h)

            beta = tuple(beta)
            if beta in out:
                out[beta] = backend.add(out[beta], factor)
            else:
                out[beta] = factor

    return out


def range_bernstein_with_fixed_scalar(p, h, fixed, unresolved, backend):
    """Bernstein range after exact endpoint fixing of selected variables."""
    unresolved = tuple(int(i) for i in unresolved)
    if not unresolved:
        r = range_polynomial_with_fixed(p, h, fixed, backend)
        return r, {
            "degree_vector": tuple(),
            "num_coefficients": 0,
            "unresolved_variables": tuple(),
        }

    degrees = [0] * len(unresolved)
    for alpha in p:
        for ui, i in enumerate(unresolved):
            degrees[ui] = max(degrees[ui], int(alpha[i]))

    power = power_coeff_reduced_unit_box_interval(
        p, h, fixed, unresolved, backend
    )
    bvals = []
    for gamma in all_indices_leq(degrees):
        bg = Interval(0.0, 0.0)
        for beta, coeff_I in power.items():
            if any(beta[i] > gamma[i] for i in range(len(unresolved))):
                continue
            weight = Interval(1.0, 1.0)
            for i in range(len(unresolved)):
                if degrees[i] == 0:
                    continue
                ratio = backend.div(
                    integer_interval(comb(gamma[i], beta[i])),
                    integer_interval(comb(degrees[i], beta[i])),
                )
                weight = backend.mul(weight, ratio)
            bg = backend.add(bg, backend.mul(weight, coeff_I))
        bvals.append(bg)

    if not bvals:
        return Interval(0.0, 0.0), {
            "degree_vector": tuple(degrees),
            "num_coefficients": 1,
            "unresolved_variables": unresolved,
        }

    return Interval(
        min(v.lo for v in bvals),
        max(v.hi for v in bvals),
    ), {
        "degree_vector": tuple(degrees),
        "num_coefficients": len(bvals),
        "unresolved_variables": unresolved,
    }


def estimate_monotonicity_reduced_bernstein_cost(p, m, signs):
    """Cheap upper estimate of Bernstein coefficients after monotonicity fixing."""
    degrees, full_cost = _degree_vector_and_bernstein_cost(p, m)
    unresolved = tuple(i for i, s in enumerate(signs) if int(s) == 0)
    if not unresolved:
        return 0, unresolved, degrees
    if len(unresolved) == int(m):
        return int(full_cost), unresolved, degrees
    reduced_one_side = 1
    for i in unresolved:
        reduced_one_side *= int(degrees[i]) + 1
    # lower and upper extremum problems use different fixed endpoints.
    return int(2 * reduced_one_side), unresolved, degrees


def range_monotonicity_reduced_bernstein_scalar(p, h, backend, monotonicity_meta=None):
    """Use proven monotonicity to remove dimensions before Bernstein ranging.

    For variables whose derivative sign is rigorously certified, the global
    lower and upper extrema occur at known opposite box endpoints.  Bernstein
    is therefore needed only on variables whose derivative sign remains
    unresolved.  This is sound and can be dramatically cheaper than a full
    tensor Bernstein transformation.
    """
    m = len(h)
    if monotonicity_meta is None:
        _, monotonicity_meta = range_monotonicity_scalar(p, h, backend)
    signs = tuple(int(s) for s in monotonicity_meta["signs"])

    lower_fixed = {}
    upper_fixed = {}
    unresolved = []
    for i, sign in enumerate(signs):
        if sign > 0:
            lower_fixed[i] = -float(h[i])
            upper_fixed[i] = +float(h[i])
        elif sign < 0:
            lower_fixed[i] = +float(h[i])
            upper_fixed[i] = -float(h[i])
        else:
            unresolved.append(i)

    unresolved = tuple(unresolved)
    full_degrees, full_cost = _degree_vector_and_bernstein_cost(p, m)

    if not unresolved:
        lo_r = range_polynomial_with_fixed(p, h, lower_fixed, backend)
        hi_r = range_polynomial_with_fixed(p, h, upper_fixed, backend)
        return Interval(lo_r.lo, hi_r.hi), {
            "method": "monotonicity-reduced-bernstein",
            "unresolved_variables": tuple(),
            "num_coefficients": 0,
            "full_bernstein_coefficients": int(full_cost),
            "degree_vector": full_degrees,
            "monotonicity_complete": True,
        }

    if len(unresolved) == m:
        r, meta = range_bernstein_scalar(p, h, backend)
        return r, {
            "method": "monotonicity-reduced-bernstein",
            "unresolved_variables": unresolved,
            "num_coefficients": int(meta["num_coefficients"]),
            "full_bernstein_coefficients": int(full_cost),
            "degree_vector": full_degrees,
            "monotonicity_complete": False,
        }

    rl, ml = range_bernstein_with_fixed_scalar(
        p, h, lower_fixed, unresolved, backend
    )
    ru, mu = range_bernstein_with_fixed_scalar(
        p, h, upper_fixed, unresolved, backend
    )
    return Interval(rl.lo, ru.hi), {
        "method": "monotonicity-reduced-bernstein",
        "unresolved_variables": unresolved,
        "num_coefficients": int(ml["num_coefficients"] + mu["num_coefficients"]),
        "full_bernstein_coefficients": int(full_cost),
        "lower_meta": ml,
        "upper_meta": mu,
        "degree_vector": full_degrees,
        "monotonicity_complete": False,
    }


def range_polynomial(p, h, backend, method="portfolio"):
    if method == "termwise":
        return range_termwise_scalar(p, h, backend), {"method": "termwise"}

    if method == "monotonicity":
        r, meta = range_monotonicity_scalar(p, h, backend)
        return r, {"method": "monotonicity", **meta}

    if method == "bernstein":
        r, meta = range_bernstein_scalar(p, h, backend)
        return r, {"method": "bernstein", **meta}

    if method == "portfolio":
        rt = range_termwise_scalar(p, h, backend)
        rm, mm = range_monotonicity_scalar(p, h, backend)
        rb, mb = range_bernstein_scalar(p, h, backend)
        lo = max(rt.lo, rm.lo, rb.lo)
        hi = min(rt.hi, rm.hi, rb.hi)
        if lo > hi:
            raise RuntimeError(
                "Rigorous range methods produced disjoint enclosures.  "
                "If all methods are sound this cannot happen; treat as an internal error."
            )
        return Interval(lo, hi), {
            "method": "portfolio",
            "termwise": rt,
            "monotonicity": rm,
            "bernstein": rb,
            "monotonicity_meta": mm,
            "bernstein_meta": mb,
        }

    raise ValueError(f"Unknown range method {method}")
