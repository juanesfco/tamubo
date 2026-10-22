"""Proof-critical real elementary-function enclosures.

The routines in this module intentionally do *not* assume that the platform
``sqrt``, ``exp``, ``log``, ``sin`` or ``cos`` implementation is correctly
rounded.  Non-trusted library calls may be used only to choose a proposal or an
argument-reduction integer.  Inclusion is established using the already-audited
binary64 interval operations plus analytic series remainder bounds.

Supported real functions
------------------------
``sqrt``
    Certified by a non-trusted square-root proposal followed by rigorous
    squaring inequalities.  The interval extension is monotone on ``[0, inf)``.
``exp``
    Argument scaling by powers of two, Taylor series on a small interval, and a
    geometric majorant for the Lagrange remainder, followed by repeated
    squaring.
``log`` / ``log10``
    Exact binary64 power-of-two decomposition followed by
    ``log(m) = 2*atanh((m-1)/(m+1))`` and a rigorously bounded atanh series.
``sin`` / ``cos``
    Rigorous Machin-formula enclosure of pi, fixed-integer quadrant reduction,
    and Taylor series with derivative-bound remainder.  If argument reduction
    is not sufficiently sharp, the sound fallback ``[-1, 1]`` is returned.
``erf``
    Alternating Maclaurin series for moderate arguments and an integration-by-
    parts erfc bracket for large arguments.
``standard_normal_pdf`` / ``standard_normal_cdf``
    Built rigorously from exp, erf, sqrt, and the internally certified pi.

All returned endpoints are binary64 values.  Overflow in proof-critical finite
operations fails closed through the trusted backend.
"""

from __future__ import annotations

import math
import struct
from typing import Tuple

import numpy as np

from .environment import nextafter_binary64
from .interval import (
    Interval,
    point_interval,
    integer_interval,
    abs_upper_interval,
    interval_neg,
)

from .array_ops import (
    rigorous_interval_add_arrays,
    rigorous_interval_subtract_arrays,
    rigorous_point_interval_multiply_arrays,
    rigorous_interval_multiply_arrays,
    rigorous_interval_divide_point_arrays,
    rigorous_interval_divide_arrays,
)


class ElementaryDomainError(ValueError):
    """The requested real elementary function is not defined on the full box."""


def _interval_intersection(a: Interval, b: Interval) -> Interval:
    lo = max(a.lo, b.lo)
    hi = min(a.hi, b.hi)
    if lo > hi:
        raise RuntimeError("Internal elementary-function certificates are disjoint.")
    return Interval(lo, hi)


def _symmetric(radius: float) -> Interval:
    radius = float(radius)
    if radius < 0.0 or np.isnan(radius):
        raise ValueError("A symmetric interval radius must be nonnegative.")
    return Interval(-radius, radius)


def _pow_interval_nonnegative(x: Interval, n: int, backend) -> Interval:
    """Rigorous integer power for an interval known to be nonnegative."""
    if n < 0:
        raise ValueError("Power must be nonnegative.")
    out = point_interval(1.0)
    base = x
    k = int(n)
    while k:
        if k & 1:
            out = backend.mul(out, base)
        k >>= 1
        if k:
            base = backend.mul(base, base)
    return out


def _positive_integer_point(n: int, backend) -> Interval:
    if int(n) <= 0:
        raise ValueError("Expected a positive integer.")
    return integer_interval(int(n), backend)


def _taylor_remainder_power_over_factorial(a: float, n: int, backend) -> float:
    """Upper bound ``a**n/n!`` for ``a >= 0`` using trusted operations."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    A = point_interval(float(a))
    p = _pow_interval_nonnegative(A, n, backend)
    for k in range(2, n + 1):
        p = backend.div(p, _positive_integer_point(k, backend))
    return float(p.hi)


def _atanh_series(z: Interval, backend, terms: int = 28) -> Interval:
    """Rigorous enclosure of ``atanh(z)`` for ``|z| < 1``.

    ``terms`` is the number of retained odd-power terms, starting with ``z``.
    The tail uses

    ``sum_{j=N+1} inf a^(2j+1)/(2j+1)
       <= a^(2N+3) / ((2N+3)*(1-a^2))``.
    """
    if terms < 1:
        raise ValueError("terms must be positive")
    a = abs_upper_interval(z)
    if not np.isfinite(a) or a >= 1.0:
        raise ValueError("atanh series requires |z| < 1")

    z2 = backend.mul(z, z)
    power = z
    total = Interval(0.0, 0.0)
    for j in range(terms):
        denom = 2 * j + 1
        term = backend.div(power, _positive_integer_point(denom, backend))
        total = backend.add(total, term)
        if j + 1 < terms:
            power = backend.mul(power, z2)

    first_omitted_power = 2 * terms + 1
    A = point_interval(a)
    tail_num = _pow_interval_nonnegative(A, first_omitted_power, backend)
    a2 = backend.mul(A, A)
    one_minus = backend.sub(point_interval(1.0), a2)
    denom = backend.mul(
        _positive_integer_point(first_omitted_power, backend), one_minus
    )
    tail = backend.div(tail_num, denom)
    return backend.add(total, _symmetric(tail.hi))


def ln2_interval(backend) -> Interval:
    """Rigorous enclosure of ``ln(2)`` from ``2*atanh(1/3)``."""
    z = backend.div(point_interval(1.0), point_interval(3.0))
    return backend.mul(point_interval(2.0), _atanh_series(z, backend, terms=30))


def _atan_series_positive_ratio(num: int, den: int, backend, terms: int) -> Interval:
    """Rigorous ``atan(num/den)`` for a small positive rational ratio."""
    z = backend.div(integer_interval(num, backend), integer_interval(den, backend))
    if z.lo < 0.0 or z.hi >= 1.0:
        raise ValueError("atan helper expects a ratio in [0,1).")

    z2 = backend.mul(z, z)
    power = z
    total = Interval(0.0, 0.0)
    sign = 1
    for j in range(terms):
        term = backend.div(power, _positive_integer_point(2 * j + 1, backend))
        total = backend.add(total, term if sign > 0 else interval_neg(term))
        sign = -sign
        if j + 1 < terms:
            power = backend.mul(power, z2)

    # Alternating-series remainder magnitude is no larger than the first
    # omitted term.  Use the largest possible |z| from the rigorous z interval.
    a = z.hi
    omitted_power = 2 * terms + 1
    A = point_interval(a)
    tail = _pow_interval_nonnegative(A, omitted_power, backend)
    tail = backend.div(tail, _positive_integer_point(omitted_power, backend))
    return backend.add(total, _symmetric(tail.hi))


def pi_interval(backend) -> Interval:
    """Rigorous pi enclosure using Machin's formula.

    ``pi = 16 atan(1/5) - 4 atan(1/239)``.
    """
    a = _atan_series_positive_ratio(1, 5, backend, terms=32)
    b = _atan_series_positive_ratio(1, 239, backend, terms=10)
    return backend.sub(
        backend.mul(integer_interval(16, backend), a),
        backend.mul(integer_interval(4, backend), b),
    )


def _binary64_floor_log2_positive(x: float) -> int:
    """Exact ``floor(log2(x))`` from the binary64 bit pattern for ``x > 0``."""
    x = float(x)
    if not (x > 0.0 and np.isfinite(x)):
        raise ValueError("x must be positive and finite")
    bits = struct.unpack(">Q", struct.pack(">d", x))[0]
    expbits = (bits >> 52) & 0x7FF
    frac = bits & ((1 << 52) - 1)
    if expbits:
        return int(expbits) - 1023
    # Positive subnormal: x = frac * 2^-1074.
    return int(frac.bit_length() - 1 - 1074)


def _power_of_two_binary64(k: int) -> float:
    """Construct the exact binary64 value ``2**k`` for -1074 <= k <= 1023."""
    k = int(k)
    if k < -1074 or k > 1023:
        raise OverflowError("Requested power of two is outside finite binary64.")
    if k >= -1022:
        bits = (k + 1023) << 52
    else:
        bits = 1 << (k + 1074)
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def sqrt_point_interval(x: float, backend) -> Interval:
    """Rigorous enclosure of ``sqrt(x)`` for one exact binary64 point.

    ``math.sqrt`` is used only as a proposal.  The returned lower endpoint is
    accepted only after its rigorously rounded square is proven <= x, and the
    upper endpoint only after its square is proven >= x.
    """
    x = float(x)
    if np.isnan(x) or x < 0.0:
        raise ElementaryDomainError("sqrt requires a nonnegative real argument.")
    if x == 0.0:
        return Interval(0.0, 0.0)
    if not np.isfinite(x):
        if x == np.inf:
            return Interval(np.inf, np.inf)
        raise ElementaryDomainError("sqrt received a non-real finite-domain value.")

    # At both extremes of the binary64 exponent range, direct squaring of a
    # candidate root can make the endpoint proof numerically awkward even when
    # the exact square root is perfectly finite.  Subnormal inputs can lose the
    # square to underflow; inputs near max-finite can require moving the upper
    # root proposal by one ulp, after which its square overflows.  In either
    # regime use the exact identity
    #
    #     sqrt(x) = 2**s * sqrt(x / 2**(2s))
    #
    # with a representable power-of-two scale.  Every division/multiplication
    # is still performed by the trusted backend, so the transformation is a
    # proof computation rather than a floating proposal shortcut.
    min_normal = float(np.finfo(np.float64).tiny)
    if x < min_normal:
        k = _binary64_floor_log2_positive(x)
        s = int(math.ceil((-1022 - k) / 2.0))
        scale_down = _power_of_two_binary64(-2 * s)
        scaled = backend.div(point_interval(x), point_interval(scale_down))
        root_scaled = sqrt_interval(scaled, backend)
        root_scale = point_interval(_power_of_two_binary64(-s))
        return backend.mul(root_scaled, root_scale)

    k = _binary64_floor_log2_positive(x)
    if k >= 1022:
        even_exponent = k if (k % 2 == 0) else (k - 1)
        scale = _power_of_two_binary64(even_exponent)
        scaled = backend.div(point_interval(x), point_interval(scale))
        root_scaled = sqrt_interval(scaled, backend)
        root_scale = point_interval(_power_of_two_binary64(even_exponent // 2))
        return backend.mul(root_scaled, root_scale)

    proposal = float(math.sqrt(x))
    if not (proposal > 0.0 and np.isfinite(proposal)):
        # Defensive proposal only; a normal platform should never take this path.
        proposal = 1.0 if x <= 1.0 else x

    lo = proposal
    hi = proposal
    X = point_interval(x)

    # Move outward until the proof inequalities hold.  On normal binary64 sqrt
    # this needs at most a couple of ulps; the loop does not assume that fact.
    for _ in range(10000):
        sq = backend.mul(point_interval(lo), point_interval(lo))
        if sq.hi <= X.lo:
            break
        lo = nextafter_binary64(lo, 0.0)
    else:  # pragma: no cover - highly defensive platform failure
        raise FloatingPointError("Could not certify a lower sqrt endpoint.")

    for _ in range(10000):
        sq = backend.mul(point_interval(hi), point_interval(hi))
        if sq.lo >= X.hi:
            break
        hi = nextafter_binary64(hi, np.inf)
    else:  # pragma: no cover - highly defensive platform failure
        raise FloatingPointError("Could not certify an upper sqrt endpoint.")

    return Interval(lo, hi)



def sqrt_point_interval_arrays(x, backend):
    """Rigorously enclose square roots of nonnegative binary64 point arrays.

    ``numpy.sqrt`` supplies proposals only.  Each lower proposal is accepted
    only after an outward-rounded square is proved no larger than the input;
    each upper proposal is accepted only after its outward-rounded square is
    proved no smaller.  Extreme exponent cases use the mature scalar proof.
    """
    x = np.asarray(x, dtype=np.float64)
    if np.any(np.isnan(x)) or np.any(x < 0.0):
        raise ElementaryDomainError("sqrt requires nonnegative real arguments.")
    lo = np.empty(x.shape, dtype=np.float64); hi = np.empty(x.shape, dtype=np.float64)
    zero = x == 0.0; posinf = x == np.inf
    lo[zero] = 0.0; hi[zero] = 0.0
    lo[posinf] = np.inf; hi[posinf] = np.inf

    finite_pos = np.isfinite(x) & (x > 0.0)
    if not np.any(finite_pos):
        return lo, hi
    min_normal = float(np.finfo(np.float64).tiny)
    maxish = np.zeros(x.shape, dtype=bool)
    for idx in zip(*np.nonzero(finite_pos)) if x.ndim else [()]:
        if x[idx] < min_normal or _binary64_floor_log2_positive(float(x[idx])) >= 1022:
            maxish[idx] = True
    normal = finite_pos & ~maxish
    if np.any(normal):
        proposal = np.sqrt(x[normal], dtype=np.float64)
        l = proposal.copy(); h = proposal.copy(); target = x[normal]
        for _ in range(16):
            sl, sh = rigorous_interval_multiply_arrays(l, l, l, l, backend)
            bad = sh > target
            if not np.any(bad): break
            l[bad] = np.nextafter(l[bad], np.float64(0.0))
        else:  # pragma: no cover
            raise FloatingPointError("Could not certify batched lower sqrt endpoints.")
        for _ in range(16):
            sl, sh = rigorous_interval_multiply_arrays(h, h, h, h, backend)
            bad = sl < target
            if not np.any(bad): break
            h[bad] = np.nextafter(h[bad], np.float64(np.inf))
        else:  # pragma: no cover
            raise FloatingPointError("Could not certify batched upper sqrt endpoints.")
        lo[normal], hi[normal] = l, h

    if np.any(maxish):
        for idx in zip(*np.nonzero(maxish)) if x.ndim else [()]:
            I = sqrt_point_interval(float(x[idx]), backend)
            lo[idx], hi[idx] = I.lo, I.hi
    return lo, hi


def sqrt_interval_arrays(lo, hi, backend):
    """Natural rigorous square root for interval endpoint arrays."""
    lo, hi = np.broadcast_arrays(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))
    if np.any(lo < 0.0) or np.any(lo > hi):
        raise ElementaryDomainError("sqrt interval arrays require 0 <= lo <= hi.")
    lout, _ = sqrt_point_interval_arrays(lo, backend)
    _, hout = sqrt_point_interval_arrays(hi, backend)
    return lout, hout

def sqrt_interval(x: Interval, backend) -> Interval:
    """Natural interval extension of real square root."""
    if x.lo < 0.0:
        raise ElementaryDomainError(
            f"sqrt is not real-valued over the full interval [{x.lo}, {x.hi}]."
        )
    lo = 0.0 if x.lo == 0.0 else sqrt_point_interval(x.lo, backend).lo
    hi = sqrt_point_interval(x.hi, backend).hi
    return Interval(lo, hi)


def _exp_small_interval(r: Interval, backend, degree: int = 18) -> Interval:
    """Rigorous exp enclosure for ``|r| <= 1/8`` using Taylor + majorant."""
    a = abs_upper_interval(r)
    if a > 0.12500000000000003:
        raise ValueError("Internal exp small-argument reduction failed.")

    total = point_interval(1.0)
    term = point_interval(1.0)
    for n in range(1, degree + 1):
        term = backend.mul(term, r)
        term = backend.div(term, _positive_integer_point(n, backend))
        total = backend.add(total, term)

    # |R_degree| <= exp(a)*a^(degree+1)/(degree+1)!
    # and exp(a) <= 1/(1-a) for 0 <= a < 1.
    p = _taylor_remainder_power_over_factorial(a, degree + 1, backend)
    majorant = backend.div(
        point_interval(1.0),
        backend.sub(point_interval(1.0), point_interval(a)),
    )
    tail = backend.mul(point_interval(p), majorant)
    result = backend.add(total, _symmetric(tail.hi))
    return Interval(max(0.0, result.lo), result.hi)


def exp_point_interval(x: float, backend) -> Interval:
    """Rigorous enclosure of ``exp(x)`` for an exact finite binary64 point."""
    x = float(x)
    if np.isnan(x):
        raise ElementaryDomainError("exp does not accept NaN.")
    if x == -np.inf:
        return Interval(0.0, 0.0)
    if x == np.inf:
        return Interval(np.inf, np.inf)
    if x == 0.0:
        return Interval(1.0, 1.0)

    r = point_interval(x)
    halvings = 0
    while abs_upper_interval(r) > 0.125:
        r = backend.mul(r, point_interval(0.5))
        halvings += 1
        if halvings > 4096:  # pragma: no cover - defensive
            raise FloatingPointError("exp argument reduction did not converge.")

    out = _exp_small_interval(r, backend)
    for _ in range(halvings):
        out = backend.mul(out, out)
        # exp is nonnegative; intersect away a possible negative minimum
        # subnormal introduced solely by outward rounding after underflow.
        out = Interval(max(0.0, out.lo), out.hi)
    return out


def exp_interval(x: Interval, backend) -> Interval:
    """Monotone natural interval extension of the real exponential."""
    lo = exp_point_interval(x.lo, backend).lo
    hi = exp_point_interval(x.hi, backend).hi
    return Interval(lo, hi)



def exp_point_interval_arrays(x, backend, degree: int = 18):
    """Rigorously enclose ``exp(x)`` for an array of finite binary64 points.

    This is the endpoint-array analogue of :func:`exp_point_interval`.  It uses
    the same repeated-halving reduction, Taylor polynomial, majorant remainder,
    and repeated squaring argument, but all independent elements are carried in
    numeric lower/upper endpoint arrays.
    """
    x = np.asarray(x, dtype=np.float64)
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        # Keep the initial vectorized proof kernel deliberately finite.  The
        # mature scalar primitive handles extended endpoints.
        lo = np.empty(x.shape, dtype=np.float64); hi = np.empty(x.shape, dtype=np.float64)
        for idx in np.ndindex(x.shape):
            I = exp_point_interval(float(x[idx]), backend)
            lo[idx], hi[idx] = I.lo, I.hi
        return lo, hi
    if degree < 1:
        raise ValueError("degree must be positive")

    r_lo = x.copy(); r_hi = x.copy()
    halvings = np.zeros(x.shape, dtype=np.int32)
    active = np.maximum(np.abs(r_lo), np.abs(r_hi)) > 0.125
    while np.any(active):
        nl, nh = rigorous_point_interval_multiply_arrays(0.5, r_lo[active], r_hi[active], backend)
        r_lo[active], r_hi[active] = nl, nh
        halvings[active] += 1
        if int(np.max(halvings)) > 4096:
            raise FloatingPointError("exp array argument reduction did not converge.")
        active = np.maximum(np.abs(r_lo), np.abs(r_hi)) > 0.12500000000000003

    total_lo = np.ones(x.shape, dtype=np.float64); total_hi = total_lo.copy()
    term_lo = total_lo.copy(); term_hi = total_hi.copy()
    for n in range(1, degree + 1):
        term_lo, term_hi = rigorous_interval_multiply_arrays(term_lo, term_hi, r_lo, r_hi, backend)
        term_lo, term_hi = rigorous_interval_divide_point_arrays(term_lo, term_hi, float(n), backend)
        total_lo, total_hi = rigorous_interval_add_arrays(total_lo, total_hi, term_lo, term_hi, backend)

    # Majorant remainder: exp(a)*a^(degree+1)/(degree+1)! with
    # exp(a) <= 1/(1-a), where a is the endpoint-derived exact upper bound on |r|.
    a = np.maximum(np.abs(r_lo), np.abs(r_hi))
    p_lo = a.copy(); p_hi = a.copy()
    power = degree + 1
    if power == 0:
        p_lo = np.ones_like(a); p_hi = p_lo.copy()
    else:
        # Simple repeated multiplication is inexpensive for the fixed small
        # degree and keeps this audit path easy to compare with the scalar proof.
        p_lo = np.ones_like(a); p_hi = p_lo.copy()
        A_lo = a; A_hi = a
        for _ in range(power):
            p_lo, p_hi = rigorous_interval_multiply_arrays(p_lo, p_hi, A_lo, A_hi, backend)
    for k in range(2, power + 1):
        p_lo, p_hi = rigorous_interval_divide_point_arrays(p_lo, p_hi, float(k), backend)

    one = np.ones_like(a)
    den_lo, den_hi = rigorous_interval_subtract_arrays(one, one, a, a, backend)
    maj_lo, maj_hi = rigorous_interval_divide_arrays(one, one, den_lo, den_hi, backend)
    tail_lo, tail_hi = rigorous_interval_multiply_arrays(p_lo, p_hi, maj_lo, maj_hi, backend)
    total_lo, total_hi = rigorous_interval_add_arrays(total_lo, total_hi, -tail_hi, tail_hi, backend)
    total_lo = np.maximum(0.0, total_lo)

    max_h = int(np.max(halvings)) if halvings.size else 0
    for level in range(max_h):
        mask = halvings > level
        if not np.any(mask):
            break
        sl, sh = rigorous_interval_multiply_arrays(total_lo[mask], total_hi[mask], total_lo[mask], total_hi[mask], backend)
        total_lo[mask], total_hi[mask] = np.maximum(0.0, sl), sh

    zero = x == 0.0
    total_lo[zero] = 1.0; total_hi[zero] = 1.0
    return total_lo, total_hi


def exp_interval_arrays(lo, hi, backend, degree: int = 18):
    """Monotone rigorous exponential for interval endpoint arrays."""
    lo, hi = np.broadcast_arrays(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))
    if np.any(lo > hi):
        raise ValueError("exp interval lower endpoints exceed upper endpoints.")
    elo, _ = exp_point_interval_arrays(lo, backend, degree=degree)
    _, ehi = exp_point_interval_arrays(hi, backend, degree=degree)
    return elo, ehi

def log_point_interval(x: float, backend) -> Interval:
    """Rigorous enclosure of natural logarithm at one positive binary64 point."""
    x = float(x)
    if not (x > 0.0):
        raise ElementaryDomainError("log requires a strictly positive real argument.")
    if x == np.inf:
        return Interval(np.inf, np.inf)
    if x == 1.0:
        return Interval(0.0, 0.0)

    k = _binary64_floor_log2_positive(x)
    scale = point_interval(_power_of_two_binary64(k))
    m = backend.div(point_interval(x), scale)

    z = backend.div(
        backend.sub(m, point_interval(1.0)),
        backend.add(m, point_interval(1.0)),
    )
    log_m = backend.mul(point_interval(2.0), _atanh_series(z, backend, terms=28))
    k_ln2 = backend.mul(integer_interval(k, backend), ln2_interval(backend))
    return backend.add(log_m, k_ln2)


def log_interval(x: Interval, backend) -> Interval:
    """Monotone natural interval extension of real logarithm."""
    if x.lo <= 0.0:
        raise ElementaryDomainError(
            f"log is not real-valued over the full interval [{x.lo}, {x.hi}]."
        )
    return Interval(
        log_point_interval(x.lo, backend).lo,
        log_point_interval(x.hi, backend).hi,
    )


def ln10_interval(backend) -> Interval:
    """Rigorous enclosure of the exact real constant ``ln(10)``."""
    return log_point_interval(10.0, backend)


def log10_point_interval(x: float, backend) -> Interval:
    """Rigorous enclosure of base-10 logarithm at one positive binary64 point.

    The identity ``log10(x) = ln(x)/ln(10)`` is evaluated using two
    independently rigorous interval enclosures and the audited interval
    division primitive.
    """
    return backend.div(log_point_interval(x, backend), ln10_interval(backend))


def log10_interval(x: Interval, backend) -> Interval:
    """Monotone interval extension of the real base-10 logarithm."""
    if x.lo <= 0.0:
        raise ElementaryDomainError(
            f"log10 is not real-valued over the full interval [{x.lo}, {x.hi}]."
        )
    return Interval(
        log10_point_interval(x.lo, backend).lo,
        log10_point_interval(x.hi, backend).hi,
    )


def _sin_cos_small_interval(r: Interval, backend) -> Tuple[Interval, Interval]:
    """Rigorous sin/cos enclosure for a reduced interval with |r| <= 1."""
    a = abs_upper_interval(r)
    if a > 1.0000000000000002:
        raise ValueError("Small-angle trig evaluator requires |r| <= 1.")

    r2 = backend.mul(r, r)

    # sin Taylor through degree 19.
    sin_total = Interval(0.0, 0.0)
    sin_term = r
    for j in range(10):
        sin_total = backend.add(sin_total, sin_term)
        if j < 9:
            denom = (2 * j + 2) * (2 * j + 3)
            sin_term = backend.mul(sin_term, r2)
            sin_term = backend.div(sin_term, _positive_integer_point(denom, backend))
            sin_term = interval_neg(sin_term)
    sin_tail = _taylor_remainder_power_over_factorial(a, 20, backend)
    sin_out = backend.add(sin_total, _symmetric(sin_tail))

    # cos Taylor through degree 20.
    cos_total = Interval(0.0, 0.0)
    cos_term = point_interval(1.0)
    for j in range(11):
        cos_total = backend.add(cos_total, cos_term)
        if j < 10:
            denom = (2 * j + 1) * (2 * j + 2)
            cos_term = backend.mul(cos_term, r2)
            cos_term = backend.div(cos_term, _positive_integer_point(denom, backend))
            cos_term = interval_neg(cos_term)
    cos_tail = _taylor_remainder_power_over_factorial(a, 21, backend)
    cos_out = backend.add(cos_total, _symmetric(cos_tail))

    sin_out = _interval_intersection(sin_out, Interval(-1.0, 1.0))
    cos_out = _interval_intersection(cos_out, Interval(-1.0, 1.0))
    return sin_out, cos_out


def sin_cos_interval(x: Interval, backend) -> Tuple[Interval, Interval]:
    """Rigorous real sin/cos interval enclosure.

    A fixed integer ``n`` is chosen as a *proposal* and the exact identities
    ``sin(x)=sin(r+n*pi/2)`` and ``cos(x)=cos(r+n*pi/2)`` are certified with a
    rigorous pi interval.  If the resulting reduced interval is too wide for
    the small-angle evaluator, ``[-1,1]`` is returned for both functions.
    """
    if np.isnan(x.lo) or np.isnan(x.hi):
        raise ElementaryDomainError("sin/cos do not accept NaN.")
    if not (np.isfinite(x.lo) and np.isfinite(x.hi)):
        return Interval(-1.0, 1.0), Interval(-1.0, 1.0)

    mid = 0.5 * x.lo + 0.5 * x.hi
    candidates = []
    try:
        n0 = int(round(mid / (0.5 * math.pi)))
        candidates = [n0, n0 - 1, n0 + 1]
    except Exception:
        return Interval(-1.0, 1.0), Interval(-1.0, 1.0)

    pi2 = backend.mul(point_interval(0.5), pi_interval(backend))
    best = None
    for n in candidates:
        try:
            shift = backend.mul(integer_interval(n, backend), pi2)
            r = backend.sub(x, shift)
        except (OverflowError, FloatingPointError):
            continue
        score = abs_upper_interval(r)
        if best is None or score < best[0]:
            best = (score, n, r)

    if best is None or best[0] > 1.0:
        return Interval(-1.0, 1.0), Interval(-1.0, 1.0)

    _, n, r = best
    sr, cr = _sin_cos_small_interval(r, backend)
    q = n % 4
    if q == 0:
        return sr, cr
    if q == 1:
        return cr, interval_neg(sr)
    if q == 2:
        return interval_neg(sr), interval_neg(cr)
    return interval_neg(cr), sr


def sin_interval(x: Interval, backend) -> Interval:
    return sin_cos_interval(x, backend)[0]


def cos_interval(x: Interval, backend) -> Interval:
    return sin_cos_interval(x, backend)[1]


def absolute_interval(x: Interval, backend=None) -> Interval:
    """Exact natural interval extension of the real absolute value."""
    if x.lo >= 0.0:
        return x
    if x.hi <= 0.0:
        return interval_neg(x)
    return Interval(0.0, max(-x.lo, x.hi))


def sinh_point_interval(x: float, backend) -> Interval:
    """Rigorous enclosure of ``sinh(x)`` for one binary64 point."""
    x = float(x)
    if np.isnan(x):
        raise ElementaryDomainError("sinh does not accept NaN.")
    if x == 0.0:
        return Interval(0.0, 0.0)
    if x == np.inf:
        return Interval(np.inf, np.inf)
    if x == -np.inf:
        return Interval(-np.inf, -np.inf)
    a = abs(x)
    ep = exp_point_interval(a, backend)
    em = exp_point_interval(-a, backend)
    out = backend.mul(
        point_interval(0.5),
        backend.sub(ep, em),
    )
    return out if x > 0.0 else interval_neg(out)


def sinh_interval(x: Interval, backend) -> Interval:
    """Monotone natural interval extension of real hyperbolic sine."""
    return Interval(
        sinh_point_interval(x.lo, backend).lo,
        sinh_point_interval(x.hi, backend).hi,
    )


def cosh_point_interval(x: float, backend) -> Interval:
    """Rigorous enclosure of ``cosh(x)`` for one binary64 point."""
    x = float(x)
    if np.isnan(x):
        raise ElementaryDomainError("cosh does not accept NaN.")
    if x == 0.0:
        return Interval(1.0, 1.0)
    if np.isinf(x):
        return Interval(np.inf, np.inf)
    a = abs(x)
    ep = exp_point_interval(a, backend)
    em = exp_point_interval(-a, backend)
    return backend.mul(
        point_interval(0.5),
        backend.add(ep, em),
    )


def cosh_interval(x: Interval, backend) -> Interval:
    """Natural interval extension of real hyperbolic cosine."""
    if x.lo <= 0.0 <= x.hi:
        lo = 1.0
    elif x.hi < 0.0:
        lo = cosh_point_interval(x.hi, backend).lo
    else:
        lo = cosh_point_interval(x.lo, backend).lo
    far = x.lo if abs(x.lo) >= abs(x.hi) else x.hi
    hi = cosh_point_interval(far, backend).hi
    return Interval(lo, hi)


def tanh_point_interval(x: float, backend) -> Interval:
    """Rigorous enclosure of ``tanh(x)`` using a non-overflowing formula."""
    x = float(x)
    if np.isnan(x):
        raise ElementaryDomainError("tanh does not accept NaN.")
    if x == 0.0:
        return Interval(0.0, 0.0)
    if x == np.inf:
        return Interval(1.0, 1.0)
    if x == -np.inf:
        return Interval(-1.0, -1.0)
    a = abs(x)
    # q = exp(-a), so tanh(a) = (1-q^2)/(1+q^2).  This avoids exp(+2a)
    # overflow and remains well behaved in the saturated tails.
    q = exp_point_interval(-a, backend)
    q2 = backend.mul(q, q)
    one = point_interval(1.0)
    out = backend.div(backend.sub(one, q2), backend.add(one, q2))
    # tanh(a) is mathematically in [0,1].  This intersection removes only
    # outward-rounding spill from the certified formula.
    out = _interval_intersection(out, Interval(0.0, 1.0))
    return out if x > 0.0 else interval_neg(out)


def tanh_interval(x: Interval, backend) -> Interval:
    """Monotone natural interval extension of real hyperbolic tangent."""
    return Interval(
        tanh_point_interval(x.lo, backend).lo,
        tanh_point_interval(x.hi, backend).hi,
    )


def _atan_series_interval_small(z: Interval, backend, terms: int = 36) -> Interval:
    """Rigorous atan series for ``|z| <= 1/2`` with an absolute tail bound."""
    if terms < 1:
        raise ValueError("terms must be positive")
    a = abs_upper_interval(z)
    if a > 0.5000000000000001:
        raise ValueError("internal atan series requires |z| <= 1/2")
    z2 = backend.mul(z, z)
    power = z
    total = Interval(0.0, 0.0)
    for j in range(terms):
        term = backend.div(power, _positive_integer_point(2 * j + 1, backend))
        total = backend.add(total, term if (j % 2 == 0) else interval_neg(term))
        if j + 1 < terms:
            power = backend.mul(power, z2)
    # Absolute-series tail: sum_{j=N} inf a^(2j+1)/(2j+1)
    # <= a^(2N+1)/((2N+1)*(1-a^2)).
    omitted = 2 * terms + 1
    A = point_interval(a)
    num = _pow_interval_nonnegative(A, omitted, backend)
    den = backend.mul(
        _positive_integer_point(omitted, backend),
        backend.sub(point_interval(1.0), backend.mul(A, A)),
    )
    tail = backend.div(num, den)
    return backend.add(total, _symmetric(tail.hi))


def arctan_point_interval(x: float, backend) -> Interval:
    """Rigorous enclosure of principal ``atan(x)`` for one binary64 point."""
    x = float(x)
    if np.isnan(x):
        raise ElementaryDomainError("arctan does not accept NaN.")
    if x == 0.0:
        return Interval(0.0, 0.0)
    if x == np.inf:
        return backend.div(pi_interval(backend), point_interval(2.0))
    if x == -np.inf:
        return interval_neg(backend.div(pi_interval(backend), point_interval(2.0)))
    if x < 0.0:
        return interval_neg(arctan_point_interval(-x, backend))

    X = point_interval(x)
    if x <= 0.5:
        return _atan_series_interval_small(X, backend)
    if x <= 2.0:
        z = backend.div(
            backend.sub(X, point_interval(1.0)),
            backend.add(X, point_interval(1.0)),
        )
        quarter_pi = backend.div(pi_interval(backend), point_interval(4.0))
        return backend.add(quarter_pi, _atan_series_interval_small(z, backend))
    z = backend.div(point_interval(1.0), X)
    half_pi = backend.div(pi_interval(backend), point_interval(2.0))
    return backend.sub(half_pi, _atan_series_interval_small(z, backend))


def arctan_interval(x: Interval, backend) -> Interval:
    """Monotone natural interval extension of principal real arctangent."""
    return Interval(
        arctan_point_interval(x.lo, backend).lo,
        arctan_point_interval(x.hi, backend).hi,
    )


def log1p_point_interval(x: float, backend) -> Interval:
    """Rigorous enclosure of ``log(1+x)`` for one binary64 point ``x > -1``."""
    x = float(x)
    if not x > -1.0:
        raise ElementaryDomainError("log1p requires x > -1.")
    if x == 0.0:
        return Interval(0.0, 0.0)
    shifted = backend.add(point_interval(1.0), point_interval(x))
    if shifted.lo <= 0.0:
        raise ElementaryDomainError("log1p shifted proof interval did not remain positive.")
    return log_interval(shifted, backend)


def log1p_interval(x: Interval, backend) -> Interval:
    """Monotone natural interval extension of real ``log1p``."""
    if x.lo <= -1.0:
        raise ElementaryDomainError(
            f"log1p is not real-valued over the full interval [{x.lo}, {x.hi}]."
        )
    return Interval(
        log1p_point_interval(x.lo, backend).lo,
        log1p_point_interval(x.hi, backend).hi,
    )



def _two_over_sqrt_pi_interval(backend) -> Interval:
    """Rigorous enclosure of ``2/sqrt(pi)``."""
    sqrt_pi = sqrt_interval(pi_interval(backend), backend)
    return backend.div(point_interval(2.0), sqrt_pi)


def _erf_maclaurin_positive(x: float, backend) -> Interval:
    """Rigorous ``erf(x)`` for one finite ``0 <= x <= 4`` point.

    We use the alternating entire series

        erf(x) = 2/sqrt(pi) * sum (-1)^n x^(2n+1)/(n! (2n+1)).

    Once ``n+1 >= x^2``, the exact term magnitudes decrease thereafter, so
    the alternating-series tail is bounded by the first omitted term.
    """
    x = float(x)
    if x < 0.0 or x > 4.0 or not np.isfinite(x):
        raise ValueError("erf Maclaurin helper requires finite 0 <= x <= 4")
    if x == 0.0:
        return Interval(0.0, 0.0)

    X = point_interval(x)
    x2 = backend.mul(X, X)
    term = X
    total = Interval(0.0, 0.0)
    n = 0
    # The threshold ensures monotone decrease of all subsequent exact terms.
    threshold = int(math.ceil(x * x))
    while True:
        total = backend.add(total, term)
        # Construct t_{n+1} = -t_n*x^2*(2n+1)/((n+1)(2n+3)).
        num = backend.mul(term, x2)
        num = backend.mul(num, _positive_integer_point(2 * n + 1, backend))
        den = backend.mul(
            _positive_integer_point(n + 1, backend),
            _positive_integer_point(2 * n + 3, backend),
        )
        next_term = interval_neg(backend.div(num, den))
        if n + 1 >= threshold:
            tail = abs_upper_interval(next_term)
            # Do not stop merely because the alternating theorem has become
            # applicable; continue until the first omitted term is small enough
            # to make the scalar primitive useful as a Taylor coefficient.
            if tail <= 2.0e-16 or n >= threshold + 256:
                series = backend.add(total, _symmetric(tail))
                out = backend.mul(_two_over_sqrt_pi_interval(backend), series)
                return _interval_intersection(out, Interval(0.0, 1.0))
        term = next_term
        n += 1
        if n > 10000:  # pragma: no cover - defensive
            raise FloatingPointError("erf Maclaurin evaluation did not converge")


def _erfc_asymptotic_positive(x: float, backend) -> Interval:
    """Rigorous ``erfc(x)`` for one finite ``x >= 4`` point.

    Repeated integration by parts gives alternating partial sums for
    ``I(x)=integral_x^inf exp(-t^2) dt`` with the remainder sign known and
    magnitude bounded by the first omitted asymptotic term.  Two consecutive
    partial sums therefore rigorously bracket ``I`` without trusting a
    platform erfc implementation.
    """
    x = float(x)
    if x < 4.0 or not np.isfinite(x):
        raise ValueError("erfc asymptotic helper requires finite x >= 4")
    X = point_interval(x)
    x2 = backend.mul(X, X)
    minus_x2 = interval_neg(x2)
    e = exp_interval(minus_x2, backend)
    # A_0 = exp(-x^2)/(2x).
    A = backend.div(e, backend.mul(point_interval(2.0), X))
    total = A
    sign = -1
    prev_mag = abs_upper_interval(A)
    best_bracket = None
    for k in range(0, 256):
        # A_{k+1}/A_k = (2k+1)/(2 x^2).
        ratio = backend.div(
            _positive_integer_point(2 * k + 1, backend),
            backend.mul(point_interval(2.0), x2),
        )
        A_next = backend.mul(A, ratio)
        total_next = backend.add(total, A_next if sign > 0 else interval_neg(A_next))
        lo = min(total.lo, total_next.lo)
        hi = max(total.hi, total_next.hi)
        bracket = Interval(max(0.0, lo), hi)
        mag = abs_upper_interval(A_next)
        best_bracket = bracket
        # Stop near the least asymptotic term, or once the bracket is already
        # narrower than a few ulps relative to its scale.
        if mag >= prev_mag or mag <= max(5e-324, 2e-16 * max(1.0, abs_upper_interval(total))):
            break
        total = total_next
        A = A_next
        sign = -sign
        prev_mag = mag
    if best_bracket is None:  # pragma: no cover - defensive
        best_bracket = Interval(0.0, A.hi)
    return backend.mul(_two_over_sqrt_pi_interval(backend), best_bracket)


def erf_point_interval(x: float, backend) -> Interval:
    """Rigorous enclosure of the real error function at one binary64 point."""
    x = float(x)
    if np.isnan(x):
        raise ElementaryDomainError("erf does not accept NaN")
    if x == np.inf:
        return Interval(1.0, 1.0)
    if x == -np.inf:
        return Interval(-1.0, -1.0)
    if x == 0.0:
        return Interval(0.0, 0.0)
    if x < 0.0:
        return interval_neg(erf_point_interval(-x, backend))
    if x < 4.0:
        return _erf_maclaurin_positive(x, backend)
    erfc = _erfc_asymptotic_positive(x, backend)
    out = backend.sub(point_interval(1.0), erfc)
    return _interval_intersection(out, Interval(0.0, 1.0))


def erf_interval(x: Interval, backend) -> Interval:
    """Monotone natural interval extension of the real error function."""
    return Interval(
        erf_point_interval(x.lo, backend).lo,
        erf_point_interval(x.hi, backend).hi,
    )


def standard_normal_pdf_interval(x: Interval, backend) -> Interval:
    """Rigorous standard-normal density range over ``x``."""
    x2 = real_power_interval(x, 2.0, backend)
    exponent = backend.mul(point_interval(-0.5), x2)
    numerator = exp_interval(exponent, backend)
    two_pi = backend.mul(point_interval(2.0), pi_interval(backend))
    denom = sqrt_interval(two_pi, backend)
    return backend.div(numerator, denom)


def standard_normal_pdf_point_interval(x: float, backend) -> Interval:
    return standard_normal_pdf_interval(point_interval(float(x)), backend)


def standard_normal_cdf_interval(x: Interval, backend) -> Interval:
    """Rigorous standard-normal CDF range over ``x``."""
    sqrt2 = sqrt_interval(point_interval(2.0), backend)
    z = backend.div(x, sqrt2)
    e = erf_interval(z, backend)
    return backend.mul(point_interval(0.5), backend.add(point_interval(1.0), e))


def standard_normal_cdf_point_interval(x: float, backend) -> Interval:
    return standard_normal_cdf_interval(point_interval(float(x)), backend)

def _coerce_finite_real_exponent(power: float) -> float:
    """Return a finite binary64 exponent used as an exact real constant."""
    p = float(power)
    if not np.isfinite(p):
        raise ElementaryDomainError("real power requires a finite real exponent.")
    return p


def _integer_power_point_interval(x: float, n: int, backend) -> Interval:
    """Rigorous integer power of one exact binary64 point for n >= 0."""
    n = int(n)
    if n < 0:
        raise ValueError("Point integer-power helper requires n >= 0.")
    out = point_interval(1.0)
    base = point_interval(float(x))
    k = n
    while k:
        if k & 1:
            out = backend.mul(out, base)
        k >>= 1
        if k:
            base = backend.mul(base, base)
    return out


def _integer_power_interval(x: Interval, n: int, backend) -> Interval:
    """Rigorous natural interval extension for an integer exponent."""
    n = int(n)
    if n == 0:
        return point_interval(1.0)
    if n < 0:
        positive = _integer_power_interval(x, -n, backend)
        if positive.lo <= 0.0 <= positive.hi:
            raise ElementaryDomainError(
                "negative integer power is singular because the base range contains zero."
            )
        return backend.div(point_interval(1.0), positive)

    lo_pow = _integer_power_point_interval(x.lo, n, backend)
    hi_pow = _integer_power_point_interval(x.hi, n, backend)
    if n % 2 == 1:
        return Interval(lo_pow.lo, hi_pow.hi)
    if x.lo >= 0.0:
        return Interval(lo_pow.lo, hi_pow.hi)
    if x.hi <= 0.0:
        return Interval(hi_pow.lo, lo_pow.hi)
    return Interval(0.0, max(lo_pow.hi, hi_pow.hi))


def positive_power_interval_exponent_interval(
    x: Interval, exponent: Interval, backend
) -> Interval:
    """Rigorous ``x**p`` for positive ``x`` and an interval of real exponents.

    This helper is used by the Taylor remainder proof for non-integer powers.
    It is intentionally defined through the trusted identity
    ``x**p = exp(p*log(x))`` rather than through the platform ``pow`` routine.
    """
    if x.lo <= 0.0:
        raise ElementaryDomainError(
            "real non-integer power requires a strictly positive base range."
        )
    L = log_interval(x, backend)
    return exp_interval(backend.mul(exponent, L), backend)


def real_power_point_interval(x: float, power: float, backend) -> Interval:
    """Rigorous enclosure of ``x**power`` for exact binary64 constants.

    Integer-valued exponents use integer power semantics, including negative
    bases.  Genuinely non-integer exponents use the principal real function
    ``exp(power*log(x))`` and therefore require ``x > 0``.  ``0**p`` is
    supported for positive non-integer ``p`` and equals zero.
    """
    x = float(x)
    p = _coerce_finite_real_exponent(power)
    if np.isnan(x):
        raise ElementaryDomainError("real power does not accept NaN bases.")

    if p.is_integer():
        return _integer_power_interval(point_interval(x), int(p), backend)

    if x < 0.0:
        raise ElementaryDomainError(
            "a non-integer real exponent is not real-valued for a negative base."
        )
    if x == 0.0:
        if p > 0.0:
            return point_interval(0.0)
        raise ElementaryDomainError(
            "a non-integer negative power is singular at zero."
        )
    if x == np.inf:
        return point_interval(np.inf if p > 0.0 else 0.0)
    if not np.isfinite(x):
        raise ElementaryDomainError("real power received a non-real base.")

    exponent = point_interval(p)
    L = log_point_interval(x, backend)
    return exp_interval(backend.mul(exponent, L), backend)


def real_power_interval(x: Interval, power: float, backend) -> Interval:
    """Natural real interval extension of ``x**power`` for constant power.

    The exponent is interpreted as the exact real number represented by the
    supplied finite binary64 value.  Non-integer exponents require a
    nonnegative base; a base touching zero is allowed only for positive powers.
    """
    p = _coerce_finite_real_exponent(power)
    if p.is_integer():
        return _integer_power_interval(x, int(p), backend)

    if x.lo < 0.0:
        raise ElementaryDomainError(
            f"non-integer real power {p!r} is not real-valued over the full "
            f"base interval [{x.lo}, {x.hi}]."
        )
    if x.lo == 0.0:
        if p <= 0.0:
            raise ElementaryDomainError(
                f"real power {p!r} is singular on a base interval containing zero."
            )
        hi = real_power_point_interval(x.hi, p, backend).hi
        return Interval(0.0, hi)

    # For strictly positive x, x**p is monotone increasing when p>0 and
    # decreasing when p<0.  Endpoint evaluation is therefore sufficient.
    if p > 0.0:
        return Interval(
            real_power_point_interval(x.lo, p, backend).lo,
            real_power_point_interval(x.hi, p, backend).hi,
        )
    return Interval(
        real_power_point_interval(x.hi, p, backend).lo,
        real_power_point_interval(x.lo, p, backend).hi,
    )


__all__ = [
    "ElementaryDomainError",
    "sqrt_point_interval",
    "sqrt_interval",
    "exp_point_interval",
    "exp_interval",
    "log_point_interval",
    "log_interval",
    "sin_cos_interval",
    "sin_interval",
    "cos_interval",
    "erf_point_interval",
    "erf_interval",
    "standard_normal_pdf_point_interval",
    "standard_normal_pdf_interval",
    "standard_normal_cdf_point_interval",
    "standard_normal_cdf_interval",
    "ln2_interval",
    "pi_interval",
    "real_power_point_interval",
    "real_power_interval",
    "positive_power_interval_exponent_interval",
]
