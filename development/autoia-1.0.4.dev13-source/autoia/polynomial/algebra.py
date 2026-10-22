"""Point-polynomial algebra for AutoIA.

This module is part of the proof-critical polynomial layer extracted during
Step 1 (0.12.0.dev4).  The algorithms are behavior-preserving copies of the
Version 0.11 reference implementation.  Stored binary64 polynomial
coefficients are interpreted as exact real point values; any coefficient
rounding error created by supported algebra is rigorously moved into an
interval remainder using :mod:`autoia.trusted`.

This module may depend on the trusted kernel, NumPy, and the Python standard
library.  It must not depend on representations, semantic primitives,
controllers, or the legacy reference core.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import numpy as np

from ..trusted import Interval, point_interval

def zero_alpha(m):
    return (0,) * m


def basis_alpha(m, i):
    a = [0] * m
    a[i] = 1
    return tuple(a)


def total_degree(alpha):
    return sum(alpha)


@lru_cache(maxsize=512)
def _multi_indices_exact_degree_cached(m, degree):
    m = int(m); degree = int(degree)
    if m < 1 or degree < 0:
        return tuple()
    if m == 1:
        return ((degree,),)
    out = []
    for first in range(degree + 1):
        for rest in _multi_indices_exact_degree_cached(m - 1, degree - first):
            out.append((first,) + rest)
    return tuple(out)


def multi_indices_exact_degree(m, degree):
    # Return a fresh iterator over immutable cached combinatorics.
    yield from _multi_indices_exact_degree_cached(int(m), int(degree))


def clean_poly(p, tol=0.0):
    out = {}
    for a, c in p.items():
        c = float(c)
        if not np.isfinite(c):
            raise FloatingPointError(
                "Nonfinite point polynomial coefficient.  "
                "Version 0.6 reference implementation stops rather than "
                "claiming a certificate through a nonfinite predictor."
            )
        if tol == 0.0:
            if c != 0.0:
                out[a] = c
        elif abs(c) > tol:
            out[a] = c
    return out


def monomial_magnitude_upper(alpha, h, backend) -> float:
    """Rigorous upper bound H >= prod_i h_i**alpha_i for h_i >= 0.

    Step 9 caches repeated results on the backend instance.  The key contains
    only stored binary64 values and the multi-index; the cached value was itself
    produced by the trusted scalar backend in the current audited execution.
    """
    alpha = tuple(int(v) for v in alpha)
    hkey = tuple(float(v) for v in h)
    cache = getattr(backend, "_monomial_magnitude_cache", None)
    key = (alpha, hkey)
    if cache is not None and key in cache:
        return float(cache[key])
    H = 1.0
    for hi, power in zip(hkey, alpha):
        if hi < 0.0:
            raise ValueError("Half-widths h must be nonnegative.")
        for _ in range(power):
            H = backend.mul(point_interval(H), point_interval(hi)).hi
    H = float(H)
    if cache is not None:
        cache[key] = H
    return H


def monomial_range_symmetric(alpha, h, backend) -> Interval:
    if total_degree(alpha) == 0:
        return Interval(1.0, 1.0)
    H = monomial_magnitude_upper(alpha, h, backend)
    if all(k % 2 == 0 for k in alpha):
        return Interval(0.0, H)
    return Interval(-H, H)


def point_power_interval(x: float, power: int, backend) -> Interval:
    out = Interval(1.0, 1.0)
    base = point_interval(x)
    for _ in range(power):
        out = backend.mul(out, base)
    return out


def variable_power_range(index, power, h, fixed, backend) -> Interval:
    if power == 0:
        return Interval(1.0, 1.0)
    if index in fixed:
        return point_power_interval(float(fixed[index]), power, backend)
    H = 1.0
    for _ in range(power):
        H = backend.mul(point_interval(H), point_interval(float(h[index]))).hi
    if power % 2 == 0:
        return Interval(0.0, H)
    return Interval(-H, H)


def monomial_range_with_fixed(alpha, h, fixed, backend) -> Interval:
    out = Interval(1.0, 1.0)
    for i, power in enumerate(alpha):
        out = backend.mul(out, variable_power_range(i, power, h, fixed, backend))
    return out


def bound_coefficient_error_poly(coeff_error, h, backend) -> Interval:
    """Range sum_alpha E_alpha q^alpha with interval coefficient errors E_alpha."""
    total = Interval(0.0, 0.0)
    for alpha, E in coeff_error.items():
        if E.lo == E.hi == 0.0:
            continue
        M = monomial_range_symmetric(alpha, h, backend)
        total = backend.add(total, backend.mul(E, M))
    return total


def rigorous_poly_add_point(p, q, h, backend):
    """
    Return (stored point polynomial, rigorous scalar range of coefficient-rounding
    error) for the exact sum of stored point polynomials p and q.
    """
    out = {}
    errors = {}
    for alpha in set(p) | set(q):
        if alpha in p and alpha in q:
            a = float(p[alpha])
            b = float(q[alpha])
            # Safe exact-real special cases.
            if b == 0.0:
                s = a
                exact = point_interval(a)
            elif a == 0.0:
                s = b
                exact = point_interval(b)
            elif b == -a:
                s = 0.0
                exact = Interval(0.0, 0.0)
            else:
                s = float(a + b)
                if not np.isfinite(s):
                    raise FloatingPointError("Nonfinite point coefficient in polynomial addition.")
                exact = backend.add(point_interval(a), point_interval(b))
            err = backend.sub(exact, point_interval(s))
            if s != 0.0:
                out[alpha] = s
            if not (err.lo == err.hi == 0.0):
                errors[alpha] = err
        elif alpha in p:
            out[alpha] = float(p[alpha])
        else:
            out[alpha] = float(q[alpha])
    return clean_poly(out), bound_coefficient_error_poly(errors, h, backend)



@lru_cache(maxsize=256)
def _polynomial_product_plan(p_keys, q_keys):
    """Cache the sparse monomial-convolution structure for repeated supports."""
    groups = defaultdict(list)
    for alpha in p_keys:
        for beta in q_keys:
            gamma = tuple(x + y for x, y in zip(alpha, beta))
            groups[gamma].append((alpha, beta))
    return tuple((gamma, tuple(pairs)) for gamma, pairs in groups.items())

def rigorous_poly_mul_truncated(p, q, max_order, h, backend):
    """Multiply two sparse stored point polynomials rigorously.

    Step 9 caches the monomial-convolution *structure* for repeated polynomial
    supports.  The arithmetic order within every coefficient is unchanged from
    the historical implementation, so the trusted coefficient enclosures remain
    bitwise compatible on the reference reduction path.
    """
    kept = {}
    coefficient_intervals = {}
    plan = _polynomial_product_plan(tuple(p.keys()), tuple(q.keys()))

    for gamma, pairs in plan:
        s_point = 0.0
        S = Interval(0.0, 0.0)
        for alpha, beta in pairs:
            ca = float(p[alpha])
            cb = float(q[beta])
            s_point = float(s_point + float(ca * cb))
            if not np.isfinite(s_point):
                raise FloatingPointError("Nonfinite point coefficient in polynomial multiplication.")
            prod = backend.mul(point_interval(ca), point_interval(cb))
            S = backend.add(S, prod)
        coefficient_intervals[gamma] = S
        if total_degree(gamma) <= max_order and s_point != 0.0:
            kept[gamma] = s_point

    extra = Interval(0.0, 0.0)
    for gamma, Cgamma in coefficient_intervals.items():
        M = monomial_range_symmetric(gamma, h, backend)
        if total_degree(gamma) <= max_order:
            stored = float(kept.get(gamma, 0.0))
            E = backend.sub(Cgamma, point_interval(stored))
            extra = backend.add(extra, backend.mul(E, M))
        else:
            extra = backend.add(extra, backend.mul(Cgamma, M))

    return clean_poly(kept), extra


def build_reciprocal_predictor_point(p, m, order, fallback_center=None):
    """
    Proposal-only point polynomial Q(q) approximating 1/P(q).

    If the stored constant coefficient p_0 is nonzero, use the formal multivariate
    reciprocal recurrence through total degree `order`.  These calculations are
    deliberately ordinary floating point: the returned Q is only a predictor.
    Its ACTUAL stored residual is certified later.

    If p_0 is zero or the recurrence produces a nonfinite coefficient, fall back
    to a constant predictor 1/fallback_center.
    """
    z = zero_alpha(m)
    p0 = float(p.get(z, 0.0))

    def constant_fallback():
        c = float(fallback_center) if fallback_center is not None else p0
        if c == 0.0 or not np.isfinite(c):
            raise ZeroDivisionError(
                "Could not construct a finite reciprocal predictor center."
            )
        q0 = float(1.0 / c)
        if not np.isfinite(q0):
            raise FloatingPointError("Nonfinite reciprocal predictor coefficient.")
        return {z: q0}

    if p0 == 0.0 or not np.isfinite(p0):
        return constant_fallback()

    try:
        q = {z: float(1.0 / p0)}
        if not np.isfinite(q[z]):
            return constant_fallback()

        for degree in range(1, int(order) + 1):
            for alpha in multi_indices_exact_degree(m, degree):
                ssum = 0.0
                for beta, pb in p.items():
                    if total_degree(beta) == 0:
                        continue
                    if any(beta[i] > alpha[i] for i in range(m)):
                        continue
                    gamma = tuple(alpha[i] - beta[i] for i in range(m))
                    qg = float(q.get(gamma, 0.0))
                    ssum = float(ssum + float(pb * qg))
                qa = float(-ssum / p0)
                if not np.isfinite(qa):
                    return constant_fallback()
                if qa != 0.0:
                    q[alpha] = qa
        return clean_poly(q)
    except (FloatingPointError, OverflowError, ZeroDivisionError):
        return constant_fallback()



