"""
autoia_trace_v11.py

AutoIA Version 0.11 structure-aware range-policy/controller prototype.

Purpose
-------
Turn the Version 0.5 exact-real inclusion formulas into a machine-rigorous
reference implementation for the currently supported execution subset.

The engineering model remains ordinary NumPy-style code.  Uncertain inputs are
represented internally by representation-independent ValidatedValue objects.
The mature polynomial backend from Version 0.6.1 is retained as one concrete
representation.  np.linalg.solve is
intercepted semantically.  The predictor may use ordinary floating-point linear
algebra; every calculation used to ESTABLISH the enclosure is routed through a
small outward-rounded trusted kernel.

Current supported mathematical scope
------------------------------------
* Input uncertainty is a box of finite binary64 interval endpoints.
* Generic scalar +, -, *, /, and nonnegative integer powers.
* np.linalg.solve for matrices/right-hand sides whose retained polynomial part
  is affine in the original centered uncertain parameters.  Small interval
  remainders are allowed and are incorporated rigorously into the certificate.
* Polynomial predictor of user-selected order.
* Rigorous termwise, monotonicity, Bernstein, and portfolio range bounds.
* Separate cheap internal proof ranging from strong final-output ranging.
* Cost-aware adaptive final range controller and proof-driven escalation.
* Generic postprocessing after a semantic solve.

Machine-rigor contract
----------------------
RigorousBinary64Backend assumes ordinary Python/NumPy binary64 basic arithmetic
(+,-,*,/) is IEEE-754 round-to-nearest and uses np.nextafter after EACH scalar
operation to obtain an outward enclosure.  Proof-critical dot products, matrix
products, polynomial coefficient accumulation, residual bounds, contraction
bounds, polynomial ranges, and final endpoint construction are compositions of
those trusted scalar interval operations.

The predictor-generating np.linalg.solve calls are intentionally NOT verified.
Their stored binary64 outputs are treated as exact point values, and the trusted
layer rigorously bounds the residual of the predictor actually stored.

Semantics
---------
As in Version 0.5, source literals such as 0.2 are interpreted as the binary64
values already present when AutoIA executes the Python model.  This script
certifies the real-valued mathematical semantics of those stored binary64
constants, not the decimal-source-literal semantics and not every rounding path
of the original engineering code.

Dependencies: numpy only
Run: python autoia_trace_v11.py
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from math import comb
import itertools
import numpy as np


# =============================================================================
# Basic interval type
# =============================================================================

@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def __post_init__(self):
        lo = float(self.lo)
        hi = float(self.hi)
        if np.isnan(lo) or np.isnan(hi):
            raise ValueError("NaN is not allowed in an Interval.")
        if lo > hi:
            raise ValueError(f"Invalid interval [{lo}, {hi}]")

    @property
    def mid(self):
        # Convenience only.  Proof code does not rely on this midpoint.
        return 0.5 * (self.lo + self.hi)

    @property
    def rad(self):
        # Convenience only.  Proof code does not rely on this radius.
        return 0.5 * (self.hi - self.lo)

    def __repr__(self):
        return f"Interval({self.lo:.17g}, {self.hi:.17g})"


def point_interval(x: float) -> Interval:
    """Interpret a stored binary64 number as an exact real point."""
    x = float(x)
    if np.isnan(x):
        raise ValueError("NaN point value is not supported.")
    return Interval(x, x)


def integer_interval(n: int) -> Interval:
    """Rigorous binary64 enclosure of an exact Python integer."""
    n = int(n)
    f = float(n)
    if not np.isfinite(f):
        raise OverflowError("Integer is too large for the binary64 reference backend.")
    nf = int(f)
    if nf == n:
        return Interval(f, f)
    if nf < n:
        return Interval(f, float(np.nextafter(f, np.inf)))
    return Interval(float(np.nextafter(f, -np.inf)), f)


def abs_upper_interval(x: Interval) -> float:
    """Exact comparison-based upper bound on |z| for z in x."""
    return float(max(abs(x.lo), abs(x.hi)))


def interval_neg(x: Interval) -> Interval:
    # Unary negation of a finite/extended binary64 endpoint is exact.
    return Interval(-x.hi, -x.lo)


# =============================================================================
# Arithmetic backends
# =============================================================================

class FloatBackend:
    """Development backend: same algorithms, but no machine-rigorous claim."""

    name = "float-development-v6"
    machine_rigorous = False

    @staticmethod
    def down(x):
        return float(x)

    @staticmethod
    def up(x):
        return float(x)

    def add(self, x: Interval, y: Interval) -> Interval:
        return Interval(x.lo + y.lo, x.hi + y.hi)

    def sub(self, x: Interval, y: Interval) -> Interval:
        return Interval(x.lo - y.hi, x.hi - y.lo)

    def mul(self, x: Interval, y: Interval) -> Interval:
        vals = [x.lo*y.lo, x.lo*y.hi, x.hi*y.lo, x.hi*y.hi]
        if any(np.isnan(v) for v in vals):
            raise FloatingPointError("NaN encountered in interval multiplication.")
        return Interval(min(vals), max(vals))

    def div(self, x: Interval, y: Interval) -> Interval:
        if y.lo <= 0.0 <= y.hi:
            raise ZeroDivisionError("Interval division by an interval containing zero.")
        vals = [x.lo/y.lo, x.lo/y.hi, x.hi/y.lo, x.hi/y.hi]
        if any(np.isnan(v) for v in vals):
            raise FloatingPointError("NaN encountered in interval division.")
        return Interval(min(vals), max(vals))

    def hull(self, x: Interval, y: Interval) -> Interval:
        return Interval(min(x.lo, y.lo), max(x.hi, y.hi))


class RigorousBinary64Backend(FloatBackend):
    """
    Reference outward-rounded binary64 interval backend.

    Each endpoint of each scalar +,-,*,/ is expanded by one binary64 neighbor.
    The basic operation itself is assumed IEEE-754 round-to-nearest binary64.
    Higher-level proof calculations use only compositions of these operations.
    """

    name = "binary64-nextafter-reference"
    machine_rigorous = True

    @staticmethod
    def down(x):
        x = float(x)
        if np.isnan(x):
            raise FloatingPointError("NaN cannot be rounded downward.")
        if x == -np.inf:
            return -np.inf
        return float(np.nextafter(x, -np.inf))

    @staticmethod
    def up(x):
        x = float(x)
        if np.isnan(x):
            raise FloatingPointError("NaN cannot be rounded upward.")
        if x == np.inf:
            return np.inf
        return float(np.nextafter(x, np.inf))

    def add(self, x: Interval, y: Interval) -> Interval:
        # Exact identity cases avoid gratuitous ulp inflation.
        if y.lo == y.hi == 0.0:
            return x
        if x.lo == x.hi == 0.0:
            return y
        lo_rn = x.lo + y.lo
        hi_rn = x.hi + y.hi
        if (np.isfinite(x.lo) and np.isfinite(y.lo) and not np.isfinite(lo_rn)) or \
           (np.isfinite(x.hi) and np.isfinite(y.hi) and not np.isfinite(hi_rn)):
            raise FloatingPointError("Overflow in trusted interval addition.")
        return Interval(self.down(lo_rn), self.up(hi_rn))

    def sub(self, x: Interval, y: Interval) -> Interval:
        if y.lo == y.hi == 0.0:
            return x
        lo_rn = x.lo - y.hi
        hi_rn = x.hi - y.lo
        if (np.isfinite(x.lo) and np.isfinite(y.hi) and not np.isfinite(lo_rn)) or \
           (np.isfinite(x.hi) and np.isfinite(y.lo) and not np.isfinite(hi_rn)):
            raise FloatingPointError("Overflow in trusted interval subtraction.")
        return Interval(self.down(lo_rn), self.up(hi_rn))

    def mul(self, x: Interval, y: Interval) -> Interval:
        if x.lo == x.hi == 0.0 or y.lo == y.hi == 0.0:
            return Interval(0.0, 0.0)
        if x.lo == x.hi == 1.0:
            return y
        if y.lo == y.hi == 1.0:
            return x
        if x.lo == x.hi == -1.0:
            return interval_neg(y)
        if y.lo == y.hi == -1.0:
            return interval_neg(x)

        pairs = (
            (x.lo, y.lo), (x.lo, y.hi),
            (x.hi, y.lo), (x.hi, y.hi),
        )
        lows = []
        highs = []
        for a, b in pairs:
            z = a * b
            if np.isnan(z):
                raise FloatingPointError(
                    "NaN encountered in interval multiplication; "
                    "extended 0*inf cases are not supported in v0.6."
                )
            if np.isfinite(a) and np.isfinite(b) and not np.isfinite(z):
                raise FloatingPointError("Overflow in trusted interval multiplication.")
            lows.append(self.down(z))
            highs.append(self.up(z))
        return Interval(min(lows), max(highs))

    def div(self, x: Interval, y: Interval) -> Interval:
        if y.lo <= 0.0 <= y.hi:
            raise ZeroDivisionError("Interval division by an interval containing zero.")
        if x.lo == x.hi == 0.0:
            return Interval(0.0, 0.0)
        if y.lo == y.hi == 1.0:
            return x
        if y.lo == y.hi == -1.0:
            return interval_neg(x)

        pairs = (
            (x.lo, y.lo), (x.lo, y.hi),
            (x.hi, y.lo), (x.hi, y.hi),
        )
        lows = []
        highs = []
        for a, b in pairs:
            z = a / b
            if np.isnan(z):
                raise FloatingPointError("NaN encountered in interval division.")
            if np.isfinite(a) and np.isfinite(b) and not np.isfinite(z):
                raise FloatingPointError("Overflow in trusted interval division.")
            lows.append(self.down(z))
            highs.append(self.up(z))
        return Interval(min(lows), max(highs))


# Backwards-friendly name.  Unlike Version 0.5, this backend is now used
# throughout every proof-critical path in this reference implementation.
NextafterBackend = RigorousBinary64Backend
DEFAULT_BACKEND = RigorousBinary64Backend()


# =============================================================================
# Small trusted reductions built only from scalar interval operations
# =============================================================================

def interval_sum(values, backend) -> Interval:
    total = Interval(0.0, 0.0)
    for value in values:
        total = backend.add(total, value)
    return total


def rigorous_dot_point(a, b, backend) -> Interval:
    """Enclose the exact-real dot product of stored point binary64 vectors."""
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Dot-product shapes do not match.")
    s = Interval(0.0, 0.0)
    for ai, bi in zip(a, b):
        prod = backend.mul(point_interval(ai), point_interval(bi))
        s = backend.add(s, prod)
    return s


def rigorous_matmul_point(A, B, backend):
    """Interval enclosure of exact-real A@B for stored point matrices."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError("Invalid matrix shapes for multiplication.")
    out = np.empty((A.shape[0], B.shape[1]), dtype=object)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            out[i, j] = rigorous_dot_point(A[i, :], B[:, j], backend)
    return out


def rigorous_matvec_point(A, x, backend):
    """Interval enclosure of exact-real A@x for stored point data."""
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)
    if A.ndim != 2 or x.ndim != 1 or A.shape[1] != x.shape[0]:
        raise ValueError("Invalid matrix/vector shapes.")
    out = np.empty(A.shape[0], dtype=object)
    for i in range(A.shape[0]):
        out[i] = rigorous_dot_point(A[i, :], x, backend)
    return out


def nonnegative_dot_upper(a, b, backend) -> float:
    """Upward bound for sum_i a_i b_i, assuming stored a_i,b_i >= 0."""
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Dot-product shapes do not match.")
    s = 0.0
    for ai, bi in zip(a, b):
        if ai < 0.0 or bi < 0.0:
            raise ValueError("nonnegative_dot_upper received a negative input.")
        prod_hi = backend.mul(point_interval(ai), point_interval(bi)).hi
        s = backend.add(point_interval(s), point_interval(prod_hi)).hi
    return float(s)


def nonnegative_matvec_upper(A, x, backend):
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)
    if np.any(A < 0.0) or np.any(x < 0.0):
        raise ValueError("nonnegative_matvec_upper requires nonnegative inputs.")
    if A.ndim != 2 or x.ndim != 1 or A.shape[1] != len(x):
        raise ValueError("Invalid matrix/vector shapes.")
    out = np.empty(A.shape[0])
    for i in range(A.shape[0]):
        out[i] = nonnegative_dot_upper(A[i], x, backend)
    return out


def nonnegative_matmul_upper(A, B, backend):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if np.any(A < 0.0) or np.any(B < 0.0):
        raise ValueError("nonnegative_matmul_upper requires nonnegative inputs.")
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError("Invalid matrix shapes.")
    out = np.empty((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            out[i, j] = nonnegative_dot_upper(A[i, :], B[:, j], backend)
    return out


def row_sum_norm_upper(B, backend) -> float:
    """Rigorous upper bound on ||B||_infinity for a nonnegative point upper matrix."""
    B = np.asarray(B, dtype=float)
    if np.any(B < 0.0):
        raise ValueError("row_sum_norm_upper requires B >= 0.")
    rho = 0.0
    for i in range(B.shape[0]):
        s = 0.0
        for value in B[i]:
            s = backend.add(point_interval(s), point_interval(value)).hi
        rho = max(rho, s)
    return float(rho)


# =============================================================================
# Polynomial utilities
# =============================================================================

# A scalar point polynomial is a dictionary
#     alpha -> stored binary64 coefficient
# where alpha=(alpha_1,...,alpha_m) is a multi-index and
#     q^alpha = product_i q_i^alpha_i.
#
# Every stored coefficient is interpreted as an exact real point.  Any error from
# having computed that point coefficient in floating point is rigorously moved to
# the scalar interval remainder of the ParameterizedEnclosure.


def zero_alpha(m):
    return (0,) * m


def basis_alpha(m, i):
    a = [0] * m
    a[i] = 1
    return tuple(a)


def total_degree(alpha):
    return sum(alpha)


def multi_indices_exact_degree(m, degree):
    if m == 1:
        yield (degree,)
        return
    for first in range(degree + 1):
        for rest in multi_indices_exact_degree(m - 1, degree - first):
            yield (first,) + rest


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
    """Rigorous upper bound H >= prod_i h_i**alpha_i for h_i >= 0."""
    H = 1.0
    for hi, power in zip(h, alpha):
        hi = float(hi)
        if hi < 0.0:
            raise ValueError("Half-widths h must be nonnegative.")
        for _ in range(power):
            H = backend.mul(point_interval(H), point_interval(hi)).hi
    return float(H)


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


def rigorous_poly_mul_truncated(p, q, max_order, h, backend):
    """
    Multiply two stored point polynomials.

    Returns
    -------
    kept_point : dict
        Stored point coefficients for terms of total degree <= max_order.
    extra_remainder : Interval
        Rigorous range containing BOTH
        (a) rounding error in retained point coefficients and
        (b) the full discarded high-order polynomial.
    """
    contributions = defaultdict(list)
    for alpha, ca in p.items():
        for beta, cb in q.items():
            gamma = tuple(x + y for x, y in zip(alpha, beta))
            contributions[gamma].append((float(ca), float(cb)))

    kept = {}
    coefficient_intervals = {}

    for gamma, terms in contributions.items():
        # Fast stored point proposal coefficient.
        s_point = 0.0
        for ca, cb in terms:
            s_point = float(s_point + float(ca * cb))
        if not np.isfinite(s_point):
            raise FloatingPointError("Nonfinite point coefficient in polynomial multiplication.")

        # Trusted enclosure of the exact coefficient formed from stored point inputs.
        S = Interval(0.0, 0.0)
        for ca, cb in terms:
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


# =============================================================================
# Rigorous polynomial range bounders
# =============================================================================

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


# =============================================================================
# Version 0.11 range-policy/controller layer
# =============================================================================

@dataclass(frozen=True)
class RangePolicy:
    """Generic structure- and cost-aware range policy.

    Parameters
    ----------
    internal_method : {'termwise', 'monotonicity', 'bernstein', 'portfolio'}
        Cheap method used repeatedly inside generic algebra and semantic
        certificates. The default is deliberately termwise.
    final_method : {'adaptive', 'termwise', 'monotonicity', 'bernstein', 'portfolio'}
        Method used only when final user-visible output intervals are requested.
    bernstein_max_coefficients : int
        Cost cap for the Version 0.10 adaptive final method. The cost estimate is
        for the monotonicity-reduced Bernstein calculation, not the full tensor.
    proof_bernstein_max_coefficients : int
        Cost cap for proof-driven escalation when a cheap proof is inconclusive
        (currently reciprocal nonzero tests).

    Soundness never depends on these cost caps: skipped optional tightening
    leaves a previously computed rigorous enclosure unchanged.
    """
    internal_method: str = "termwise"
    final_method: str = "adaptive"
    bernstein_max_coefficients: int = 256
    proof_bernstein_max_coefficients: int = 256

    def __post_init__(self):
        if self.internal_method not in {"termwise", "monotonicity", "bernstein", "portfolio"}:
            raise ValueError("Invalid internal_method.")
        if self.final_method not in {
            "adaptive", "termwise", "monotonicity", "bernstein", "portfolio"
        }:
            raise ValueError("Invalid final_method.")
        if self.bernstein_max_coefficients < 1:
            raise ValueError("bernstein_max_coefficients must be positive.")
        if self.proof_bernstein_max_coefficients < 1:
            raise ValueError("proof_bernstein_max_coefficients must be positive.")


def _degree_vector_and_bernstein_cost(p, m):
    degrees = [0] * int(m)
    for alpha in p:
        for i, ai in enumerate(alpha):
            degrees[i] = max(degrees[i], int(ai))
    cost = 1
    for d in degrees:
        cost *= (d + 1)
    return tuple(degrees), int(cost)


def _interval_intersection(a, b):
    lo = max(a.lo, b.lo)
    hi = min(a.hi, b.hi)
    if lo > hi:
        raise RuntimeError(
            "Two independently sound interval enclosures are disjoint; "
            "treat this as an internal soundness error."
        )
    return Interval(lo, hi)


_ACTIVE_RANGE_POLICY_STACK = []
_ACTIVE_RANGE_EVENT_STACK = []


class RangePolicyContext:
    """Small dynamic context so proof operations can request policy escalation."""
    def __init__(self, policy):
        self.policy = policy
        self.events = []

    def __enter__(self):
        _ACTIVE_RANGE_POLICY_STACK.append(self.policy)
        _ACTIVE_RANGE_EVENT_STACK.append(self.events)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _ACTIVE_RANGE_EVENT_STACK.pop()
        _ACTIVE_RANGE_POLICY_STACK.pop()
        return False


def record_range_policy_event(event):
    if _ACTIVE_RANGE_EVENT_STACK:
        _ACTIVE_RANGE_EVENT_STACK[-1].append(dict(event))


def active_range_policy():
    if _ACTIVE_RANGE_POLICY_STACK:
        return _ACTIVE_RANGE_POLICY_STACK[-1]
    return RangePolicy()


def adaptive_final_range(validated_value, policy):
    """Return a rigorous final range using structure-aware cost escalation.

    Version 0.10 replaces the Version 0.10 width-disagreement trigger with a
    stronger structural diagnostic: whether monotonicity leaves any input
    directions unresolved.  If all directions are certified monotone, the
    monotonicity corner range is already exact for the stored polynomial (up to
    outward floating-point rounding), so Bernstein cannot improve it.

    When directions remain unresolved, Bernstein is applied only in those
    dimensions.  The controller therefore compares a reduced Bernstein cost to
    the budget cap instead of paying for the full tensor representation.
    """
    rep = validated_value.rep

    if isinstance(rep, IntervalEnclosure):
        r, meta = rep.range()
        return r, {
            "method": "interval",
            "adaptive": False,
            "bernstein_used": False,
        }

    if not isinstance(rep, PolynomialEnclosure):
        return validated_value.range(policy.final_method)

    if policy.final_method != "adaptive":
        r, meta = validated_value.range(policy.final_method)
        return r, {
            **meta,
            "adaptive": False,
            "bernstein_used": policy.final_method in {"bernstein", "portfolio"},
        }

    rt, mt = validated_value.range("termwise")
    rm, mm = validated_value.range("monotonicity")
    base = _interval_intersection(rt, rm)

    wt = max(0.0, rt.hi - rt.lo)
    wm = max(0.0, rm.hi - rm.lo)
    wb = max(0.0, base.hi - base.lo)
    scale = max(wb, np.finfo(float).tiny)
    relative_disagreement = abs(wt - wm) / scale

    signs = tuple(int(s) for s in mm.get("signs", ()))
    reduced_cost, unresolved, degrees = estimate_monotonicity_reduced_bernstein_cost(
        rep.coeff, rep.m, signs
    )
    _, full_cost = _degree_vector_and_bernstein_cost(rep.coeff, rep.m)
    active = sum(int(d) > 0 for d in degrees)
    unresolved_active = sum(1 for i in unresolved if int(degrees[i]) > 0)
    unresolved_fraction = (
        unresolved_active / active if active > 0 else 0.0
    )

    bernstein_used = False
    bernstein_reason = "skipped"
    rb = None
    mb = None

    if unresolved_active == 0:
        # Structural theorem: all active coordinates are rigorously monotone, so
        # their extrema occur at the certified corners used by rm.
        bernstein_reason = "monotonicity-complete"
    elif reduced_cost <= policy.bernstein_max_coefficients:
        bernstein_used = True
        bernstein_reason = "unresolved-directions-affordable"
    else:
        bernstein_reason = "reduced-cost-cap"

    final = base
    if bernstein_used:
        rb, mb = range_monotonicity_reduced_bernstein_scalar(
            rep.coeff, rep.h, rep.backend, monotonicity_meta=mm
        )
        # Add the same scalar remainder carried by the PolynomialEnclosure.
        rb = rep.backend.add(rb, rep.remainder)
        final = _interval_intersection(final, rb)

    return final, {
        "method": "adaptive-v10",
        "adaptive": True,
        "termwise": rt,
        "monotonicity": rm,
        "cheap_intersection": base,
        "relative_disagreement": relative_disagreement,
        "degree_vector": degrees,
        "estimated_bernstein_coefficients": int(full_cost),
        "estimated_reduced_bernstein_coefficients": int(reduced_cost),
        "unresolved_variables": unresolved,
        "unresolved_fraction": float(unresolved_fraction),
        "bernstein_used": bernstein_used,
        "bernstein_reason": bernstein_reason,
        "bernstein": rb,
        "bernstein_meta": mb,
        "termwise_meta": mt,
        "monotonicity_meta": mm,
    }

# =============================================================================
# ParameterizedEnclosure common intermediate representation
# =============================================================================

class ParameterizedEnclosure:
    """
    Scalar enclosure

        z(q) in P(q) + R,

    where
      P(q) = stored point-coefficient polynomial in the ORIGINAL centered inputs,
      R    = rigorous interval remainder,
      |q_i| <= h_i.

    The stored point coefficients are interpreted as exact binary64 real numbers.
    R contains model/truncation uncertainty AND every rigorously bounded coefficient
    rounding error created by supported generic polynomial arithmetic.
    """

    __array_priority__ = 1000

    def __init__(self, coeff, remainder, h, order,
                 backend=None, range_method="portfolio"):
        self.coeff = clean_poly(coeff)
        self.remainder = remainder
        self.h = np.asarray(h, dtype=float)
        self.m = len(self.h)
        self.order = int(order)
        self.backend = backend if backend is not None else DEFAULT_BACKEND
        self.range_method = range_method

        if np.any(self.h < 0.0) or np.any(~np.isfinite(self.h)):
            raise ValueError("Version 0.6 requires finite nonnegative parameter half-widths.")

    @staticmethod
    def constant(value, h, order, backend=None, range_method="portfolio"):
        m = len(h)
        return ParameterizedEnclosure(
            {zero_alpha(m): float(value)}, Interval(0.0, 0.0), h, order,
            backend=backend, range_method=range_method,
        )

    @staticmethod
    def parameter(center, index, h, order, backend=None, range_method="portfolio"):
        m = len(h)
        return ParameterizedEnclosure(
            {zero_alpha(m): float(center), basis_alpha(m, index): 1.0},
            Interval(0.0, 0.0), h, order,
            backend=backend, range_method=range_method,
        )

    def copy(self):
        return ParameterizedEnclosure(
            dict(self.coeff), self.remainder, self.h.copy(), self.order,
            self.backend, self.range_method,
        )

    def _coerce(self, other):
        if isinstance(other, ParameterizedEnclosure):
            if self.m != other.m or not np.all(self.h == other.h):
                raise ValueError("ParameterizedEnclosure domains do not match.")
            if self.backend.name != other.backend.name:
                raise ValueError("ParameterizedEnclosure backends do not match.")
            return other
        return ParameterizedEnclosure.constant(
            other, self.h, self.order, self.backend, self.range_method,
        )

    def polynomial_range(self, method=None):
        return range_polynomial(
            self.coeff,
            self.h,
            self.backend,
            method=self.range_method if method is None else method,
        )

    def range(self, method=None):
        pr, meta = self.polynomial_range(method)
        return self.backend.add(pr, self.remainder), meta

    def is_affine_structured(self):
        # A nonzero remainder is allowed in Version 0.6 and is incorporated into
        # the semantic-solve certificate as an additive matrix/vector remainder.
        return all(total_degree(alpha) <= 1 for alpha in self.coeff)

    def affine_data(self):
        if not self.is_affine_structured():
            raise NotImplementedError(
                "Current solve primitive requires an affine retained polynomial part. "
                "A scalar interval remainder is allowed and will be certified."
            )
        c0 = self.coeff.get(zero_alpha(self.m), 0.0)
        grad = np.zeros(self.m)
        for i in range(self.m):
            grad[i] = self.coeff.get(basis_alpha(self.m, i), 0.0)
        return c0, grad, self.remainder

    def __add__(self, other):
        other = self._coerce(other)
        coeff, coeff_error = rigorous_poly_add_point(
            self.coeff, other.coeff, self.h, self.backend
        )
        r = self.backend.add(self.remainder, other.remainder)
        r = self.backend.add(r, coeff_error)
        return ParameterizedEnclosure(
            coeff, r, self.h, min(self.order, other.order),
            self.backend, self.range_method,
        )

    __radd__ = __add__

    def __neg__(self):
        # Sign change of a stored binary64 point coefficient is exact.
        return ParameterizedEnclosure(
            {alpha: -c for alpha, c in self.coeff.items()},
            interval_neg(self.remainder),
            self.h, self.order, self.backend, self.range_method,
        )

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        order = min(self.order, other.order)

        kept, poly_extra = rigorous_poly_mul_truncated(
            self.coeff, other.coeff, order, self.h, self.backend
        )

        px_range, _ = range_polynomial(
            self.coeff, self.h, self.backend, self.range_method
        )
        py_range, _ = range_polynomial(
            other.coeff, self.h, self.backend, self.range_method
        )

        # Exact enclosure identity:
        # (Px+Rx)(Py+Ry) = stored(Px Py) +
        #                  [coefficient/truncation error]
        #                  + Px Ry + Py Rx + Rx Ry.
        r = poly_extra
        r = self.backend.add(r, self.backend.mul(px_range, other.remainder))
        r = self.backend.add(r, self.backend.mul(py_range, self.remainder))
        r = self.backend.add(r, self.backend.mul(self.remainder, other.remainder))

        return ParameterizedEnclosure(
            kept, r, self.h, order, self.backend, self.range_method,
        )

    __rmul__ = __mul__

    def reciprocal(self):
        """
        Rigorous reciprocal using a proposal polynomial plus a residual certificate.

        Let y(q) be this enclosure and let Q(q) be ANY stored point-polynomial
        predictor for 1/y(q).  The exact identity

            1/y - Q = (1 - y Q) / y

        means that a rigorous range of the actual stored residual 1-yQ, divided
        by a rigorous denominator range that excludes zero, certifies the
        reciprocal predictor.  Predictor construction itself is not trusted.
        """
        policy = active_range_policy()
        proof_ranges = []

        # Start with the configured cheap internal range.  If it cannot prove
        # the denominator excludes zero, escalate only as far as necessary.
        Y, _ = self.range(self.range_method)
        proof_ranges.append((self.range_method, Y))

        if Y.lo <= 0.0 <= Y.hi and self.range_method != "monotonicity":
            Ym, _ = self.range("monotonicity")
            Y = _interval_intersection(Y, Ym)
            proof_ranges.append(("monotonicity", Ym))

        degrees, bern_cost = _degree_vector_and_bernstein_cost(self.coeff, self.m)
        if (
            Y.lo <= 0.0 <= Y.hi
            and bern_cost <= policy.proof_bernstein_max_coefficients
        ):
            Yb, _ = self.range("bernstein")
            Y = _interval_intersection(Y, Yb)
            proof_ranges.append(("bernstein", Yb))

        record_range_policy_event({
            "kind": "reciprocal-nonzero-proof",
            "methods_tried": tuple(name for name, _ in proof_ranges),
            "final_range": Y,
            "estimated_bernstein_coefficients": bern_cost,
            "success": not (Y.lo <= 0.0 <= Y.hi),
        })

        if Y.lo <= 0.0 <= Y.hi:
            raise ZeroDivisionError(
                "Cannot certify reciprocal: all affordable proof ranges still "
                f"contain zero (final range [{Y.lo:.17g}, {Y.hi:.17g}], "
                f"estimated Bernstein coefficients={bern_cost}). "
                "Subdivide or use a stronger representation."
            )

        center = _safe_interval_representative(Y)
        Qcoeff = build_reciprocal_predictor_point(
            self.coeff, self.m, self.order, fallback_center=center
        )
        Q = ParameterizedEnclosure(
            Qcoeff, Interval(0.0, 0.0), self.h, self.order,
            self.backend, self.range_method,
        )

        one = ParameterizedEnclosure.constant(
            1.0, self.h, self.order, self.backend, self.range_method
        )
        residual = one - self * Q
        H, _ = residual.range(self.range_method)

        # Exact error e = (1 - y Q)/y.  Interval division is sound even though
        # it discards dependence between numerator and denominator.
        E_residual = self.backend.div(H, Y)

        # Independent sound error enclosure from 1/Y - range(Q); intersecting
        # two sound enclosures cannot reduce validity and can improve sharpness.
        invY = self.backend.div(Interval(1.0, 1.0), Y)
        Qrange, _ = Q.range(self.range_method)
        E_range = self.backend.sub(invY, Qrange)
        Elo = max(E_residual.lo, E_range.lo)
        Ehi = min(E_residual.hi, E_range.hi)
        if Elo > Ehi:
            raise RuntimeError(
                "Independent reciprocal certificates are disjoint; treat as "
                "an internal soundness error."
            )
        E = Interval(Elo, Ehi)

        return ParameterizedEnclosure(
            Qcoeff, E, self.h, self.order, self.backend, self.range_method
        )

    def __truediv__(self, other):
        other = self._coerce(other)
        return self * other.reciprocal()

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def __pow__(self, power):
        if not isinstance(power, int) or power < 0:
            raise NotImplementedError(
                "Only nonnegative integer powers are supported in Version 0.6."
            )
        out = ParameterizedEnclosure.constant(
            1.0, self.h, self.order, self.backend, self.range_method
        )
        for _ in range(power):
            out = out * self
        return out

    def __repr__(self):
        r, _ = self.range()
        return (
            f"ParameterizedEnclosure(range=[{r.lo:.12g}, {r.hi:.12g}], "
            f"terms={len(self.coeff)}, remainder=[{self.remainder.lo:.3e},"
            f"{self.remainder.hi:.3e}])"
        )


# =============================================================================
# Semantic np.linalg.solve primitive
# =============================================================================


def contains_pe(arr):
    obj = np.asarray(arr, dtype=object)
    return any(isinstance(v, ParameterizedEnclosure) for v in obj.flat)


def extract_affine_array(arr, prototype: ParameterizedEnclosure, name):
    """
    Extract stored point affine coefficients plus scalar interval remainders.

    Returns
    -------
    values : ndarray
        Constant point coefficients.
    deriv : ndarray, shape (m, ...) 
        Point coefficients multiplying q_i.
    rem : object ndarray of Interval
        Additive scalar interval remainder for each entry.
    """
    obj = np.asarray(arr, dtype=object)
    values = np.empty(obj.shape)
    deriv = np.empty((prototype.m,) + obj.shape)
    rem = np.empty(obj.shape, dtype=object)

    for idx in np.ndindex(obj.shape):
        v = obj[idx]
        if not isinstance(v, ParameterizedEnclosure):
            v = prototype._coerce(v)
        try:
            c0, g, r = v.affine_data()
        except NotImplementedError as exc:
            raise NotImplementedError(f"{name}{idx}: {exc}") from exc
        values[idx] = c0
        deriv[(slice(None),) + idx] = g
        rem[idx] = r

    return values, deriv, rem


def build_solution_polynomials(A0, A_i, b0, b_i, order):
    """
    Proposal layer only: vector point coefficients alpha -> x_alpha.

    These solves need not be verified.  The trusted layer certifies the residual of
    the actual stored coefficients returned here.
    """
    m = len(A_i)
    coeff = {}
    z = zero_alpha(m)
    coeff[z] = np.linalg.solve(A0, b0)

    if order >= 1:
        for i in range(m):
            alpha = basis_alpha(m, i)
            coeff[alpha] = np.linalg.solve(A0, b_i[i] - A_i[i] @ coeff[z])

    for degree in range(2, order + 1):
        for alpha in multi_indices_exact_degree(m, degree):
            rhs = np.zeros_like(b0)
            for i in range(m):
                if alpha[i] > 0:
                    beta = list(alpha)
                    beta[i] -= 1
                    beta = tuple(beta)
                    rhs -= A_i[i] @ coeff[beta]
            coeff[alpha] = np.linalg.solve(A0, rhs)

    for alpha, vec in coeff.items():
        if np.any(~np.isfinite(vec)):
            raise FloatingPointError(
                f"Nonfinite predictor coefficient at multi-index {alpha}."
            )
    return coeff


def interval_vector_zeros(n):
    out = np.empty(n, dtype=object)
    for i in range(n):
        out[i] = Interval(0.0, 0.0)
    return out


def add_interval_vector_inplace(target, source, backend, sign=+1):
    for j in range(len(target)):
        if sign == +1:
            target[j] = backend.add(target[j], source[j])
        elif sign == -1:
            target[j] = backend.sub(target[j], source[j])
        else:
            raise ValueError("sign must be +1 or -1")


def point_vector_as_intervals(v):
    v = np.asarray(v, dtype=float)
    out = np.empty(len(v), dtype=object)
    for j, value in enumerate(v):
        out[j] = point_interval(value)
    return out


def residual_coefficient_intervals(A0, A_i, b0, b_i, X, backend):
    """
    Rigorous interval coefficients of

        r_hat(q) = b_hat(q) - A_hat(q) P(q)

    where all A,b,P coefficients are the stored point binary64 values.
    """
    m = len(A_i)
    n = len(b0)
    R = {}

    def get(alpha):
        if alpha not in R:
            R[alpha] = interval_vector_zeros(n)
        return R[alpha]

    # b_hat(q)
    add_interval_vector_inplace(
        get(zero_alpha(m)), point_vector_as_intervals(b0), backend, +1
    )
    for i in range(m):
        add_interval_vector_inplace(
            get(basis_alpha(m, i)), point_vector_as_intervals(b_i[i]), backend, +1
        )

    # -A0 P
    for alpha, xalpha in X.items():
        Ax = rigorous_matvec_point(A0, xalpha, backend)
        add_interval_vector_inplace(get(alpha), Ax, backend, -1)

    # -sum_i q_i A_i P
    for alpha, xalpha in X.items():
        for i in range(m):
            gamma = list(alpha)
            gamma[i] += 1
            gamma = tuple(gamma)
            Aix = rigorous_matvec_point(A_i[i], xalpha, backend)
            add_interval_vector_inplace(get(gamma), Aix, backend, -1)

    return R


def abs_bound_interval_vector_poly(R, h, backend):
    """Componentwise g with |r(q)| <= g for an interval-coefficient polynomial."""
    n = len(next(iter(R.values())))
    ranges = [Interval(0.0, 0.0) for _ in range(n)]

    for alpha, vecI in R.items():
        M = monomial_range_symmetric(alpha, h, backend)
        for j in range(n):
            ranges[j] = backend.add(ranges[j], backend.mul(vecI[j], M))

    g = np.empty(n)
    for j in range(n):
        g[j] = abs_upper_interval(ranges[j])
    return g, ranges


def rigorous_matvec_point_interval(A, xI, backend):
    """Enclose A @ x when A is a stored point matrix and x has interval entries."""
    A = np.asarray(A, dtype=float)
    xI = np.asarray(xI, dtype=object)
    if A.ndim != 2 or xI.ndim != 1 or A.shape[1] != xI.shape[0]:
        raise ValueError("Invalid matrix/interval-vector shapes.")
    out = np.empty(A.shape[0], dtype=object)
    for i in range(A.shape[0]):
        total = Interval(0.0, 0.0)
        for k in range(A.shape[1]):
            total = backend.add(
                total,
                backend.mul(point_interval(A[i, k]), xI[k]),
            )
        out[i] = total
    return out


def rigorous_matmul_point_interval(A, BI, backend):
    """Enclose A @ B when A is point-valued and B has interval entries."""
    A = np.asarray(A, dtype=float)
    BI = np.asarray(BI, dtype=object)
    if A.ndim != 2 or BI.ndim != 2 or A.shape[1] != BI.shape[0]:
        raise ValueError("Invalid matrix/interval-matrix shapes.")
    out = np.empty((A.shape[0], BI.shape[1]), dtype=object)
    for i in range(A.shape[0]):
        for j in range(BI.shape[1]):
            total = Interval(0.0, 0.0)
            for k in range(A.shape[1]):
                total = backend.add(
                    total,
                    backend.mul(point_interval(A[i, k]), BI[k, j]),
                )
            out[i, j] = total
    return out


def representative_in_interval(I):
    """Choose a finite point coefficient guaranteed to lie in interval I."""
    if I.lo == I.hi:
        return float(I.lo)
    # The representative is only a proposal; rigor comes from enclosing I-rep.
    mid = float(0.5 * I.lo + 0.5 * I.hi)
    if not np.isfinite(mid):
        raise FloatingPointError("Nonfinite interval-coefficient representative.")
    return float(min(max(mid, I.lo), I.hi))


def range_interval_coefficient_poly(polyI, h, backend, method="portfolio"):
    """
    Range an interval-coefficient polynomial rigorously.

    For each coefficient interval I_alpha choose a point representative c_alpha in
    I_alpha, range the point polynomial with the full range portfolio, and add a
    rigorous termwise range of the coefficient uncertainty

        sum_alpha (I_alpha - c_alpha) q^alpha.

    We also compute the direct termwise interval-coefficient range and intersect
    the two sound enclosures.  Therefore this routine cannot be worse than its
    direct termwise bound.
    """
    point_poly = {}
    error_range = Interval(0.0, 0.0)
    direct_range = Interval(0.0, 0.0)

    for alpha, Ialpha in polyI.items():
        rep = representative_in_interval(Ialpha)
        if rep != 0.0:
            point_poly[alpha] = rep

        M = monomial_range_symmetric(alpha, h, backend)

        # Direct interval-coefficient termwise range.
        direct_range = backend.add(
            direct_range, backend.mul(Ialpha, M)
        )

        # Coefficient uncertainty around the stored point representative.
        Ealpha = backend.sub(Ialpha, point_interval(rep))
        error_range = backend.add(
            error_range, backend.mul(Ealpha, M)
        )

    point_range, point_meta = range_polynomial(
        point_poly, h, backend, method
    )
    decomposed = backend.add(point_range, error_range)

    lo = max(direct_range.lo, decomposed.lo)
    hi = min(direct_range.hi, decomposed.hi)
    if lo > hi:
        raise RuntimeError(
            "Disjoint sound interval-coefficient polynomial enclosures; "
            "treat as an internal error."
        )
    return Interval(lo, hi), {
        "direct_termwise": direct_range,
        "point_plus_coefficient_error": decomposed,
        "point_range_meta": point_meta,
    }


def left_multiply_interval_vector_polynomial(C, R, backend):
    """Return interval coefficients of C R(q) for interval-vector polynomial R."""
    out = {}
    for alpha, vecI in R.items():
        out[alpha] = rigorous_matvec_point_interval(C, vecI, backend)
    return out


def range_interval_vector_polynomial(R, h, backend, method="portfolio"):
    """Componentwise ranges of an interval-vector polynomial."""
    n = len(next(iter(R.values())))
    ranges = []
    metas = []
    for j in range(n):
        poly_j = {alpha: vecI[j] for alpha, vecI in R.items()}
        rj, meta = range_interval_coefficient_poly(
            poly_j, h, backend, method
        )
        ranges.append(rj)
        metas.append(meta)
    return ranges, metas


def interval_matvec(AI, xI, backend):
    """Interval matrix times interval vector, with all operations outward rounded."""
    AI = np.asarray(AI, dtype=object)
    xI = np.asarray(xI, dtype=object)
    if AI.ndim != 2 or xI.ndim != 1 or AI.shape[1] != xI.shape[0]:
        raise ValueError("Invalid interval matrix/vector shapes.")
    out = np.empty(AI.shape[0], dtype=object)
    for i in range(AI.shape[0]):
        total = Interval(0.0, 0.0)
        for k in range(AI.shape[1]):
            total = backend.add(total, backend.mul(AI[i, k], xI[k]))
        out[i] = total
    return out


def interval_remainder_abs_array(rem):
    rem = np.asarray(rem, dtype=object)
    out = np.empty(rem.shape)
    for idx in np.ndindex(rem.shape):
        out[idx] = abs_upper_interval(rem[idx])
    return out


def predictor_component_polynomials(X, n):
    out = []
    for j in range(n):
        out.append({alpha: float(vec[j]) for alpha, vec in X.items() if vec[j] != 0.0})
    return out


def predictor_abs_bounds(X, h, backend, range_method, n):
    polys = predictor_component_polynomials(X, n)
    s = np.empty(n)
    ranges = []
    metas = []
    for j, poly in enumerate(polys):
        r, meta = range_polynomial(poly, h, backend, range_method)
        ranges.append(r)
        metas.append(meta)
        s[j] = abs_upper_interval(r)
    return s, ranges, metas


def refine_componentwise_correction(c, B, d_scalar, backend, max_iter=32):
    """
    Rigorously sharpen the scalar correction radius into a componentwise vector.

    Start from the already-proved bound

        |e| <= d_scalar * 1.

    If d is any valid componentwise upper bound, then

        |e| <= c + B |e| <= c + B d.

    Therefore an outward-rounded evaluation of c + B d is another valid upper
    bound.  Intersecting (componentwise minimum) with the previous valid bound is
    also sound.  Repeating approaches the fixed-point bound (I-B)^(-1)c without
    requiring a proof-critical linear solve.
    """
    c = np.asarray(c, dtype=float)
    B = np.asarray(B, dtype=float)
    if c.ndim != 1 or B.shape != (len(c), len(c)):
        raise ValueError("Invalid c/B shapes for componentwise correction.")
    if np.any(c < 0.0) or np.any(B < 0.0) or d_scalar < 0.0:
        raise ValueError("Correction refinement requires nonnegative data.")

    d = np.full(len(c), float(d_scalar))
    history = [d.copy()]

    for _ in range(int(max_iter)):
        Bd = nonnegative_matvec_upper(B, d, backend)
        candidate = np.empty_like(d)
        for j in range(len(d)):
            candidate[j] = backend.add(
                point_interval(c[j]), point_interval(Bd[j])
            ).hi

        new_d = np.minimum(d, candidate)
        history.append(new_d.copy())

        # Equality of stored binary64 values means outward iteration has plateaued.
        if np.array_equal(new_d, d):
            d = new_d
            break
        d = new_d

    return d, history


def matrix_product_abs_upper(C, A, backend):
    """Entrywise upper bound on |C A| for stored point matrices."""
    CA = rigorous_matmul_point(C, A, backend)
    out = np.empty(CA.shape)
    for idx in np.ndindex(CA.shape):
        out[idx] = abs_upper_interval(CA[idx])
    return out


def identity_minus_product_abs_upper(C, A0, backend):
    CA = rigorous_matmul_point(C, A0, backend)
    n = A0.shape[0]
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            delta = 1.0 if i == j else 0.0
            Dij = backend.sub(point_interval(delta), CA[i, j])
            out[i, j] = abs_upper_interval(Dij)
    return out


def add_nonnegative_matrices_upper(A, B, backend):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape or np.any(A < 0.0) or np.any(B < 0.0):
        raise ValueError("Expected same-shape nonnegative matrices.")
    out = np.empty(A.shape)
    for idx in np.ndindex(A.shape):
        out[idx] = backend.add(point_interval(A[idx]), point_interval(B[idx])).hi
    return out


def scale_nonnegative_matrix_upper(s, A, backend):
    s = float(s)
    A = np.asarray(A, dtype=float)
    if s < 0.0 or np.any(A < 0.0):
        raise ValueError("Expected nonnegative scale and matrix.")
    out = np.empty(A.shape)
    for idx in np.ndindex(A.shape):
        out[idx] = backend.mul(point_interval(s), point_interval(A[idx])).hi
    return out


class SolveContext:
    def __init__(self, prototype):
        self.prototype = prototype
        self.original_solve = np.linalg.solve
        self.certificates = []

    def traced_solve(self, A, b):
        if not (contains_pe(A) or contains_pe(b)):
            return self.original_solve(A, b)

        P = self.prototype
        A0, A_i, A_rem = extract_affine_array(A, P, "A")
        b0, b_i, b_rem = extract_affine_array(b, P, "b")

        if A0.ndim != 2 or A0.shape[0] != A0.shape[1]:
            raise ValueError("A must be square.")
        if b0.ndim != 1:
            raise ValueError("Version 0.6.1 expects a vector right-hand side.")
        if A0.shape[0] != len(b0):
            raise ValueError("A and b dimensions do not match.")

        n = len(b0)

        # ---------------------------------------------------------------------
        # Proposal layer: unrestricted ordinary floating point.
        # ---------------------------------------------------------------------
        X = build_solution_polynomials(A0, A_i, b0, b_i, P.order)
        C = self.original_solve(A0, np.eye(n))
        if np.any(~np.isfinite(C)):
            raise FloatingPointError("Nonfinite approximate inverse/preconditioner C.")

        # ---------------------------------------------------------------------
        # Trusted layer.
        # ---------------------------------------------------------------------

        # 1) Bound the point-affine residual of the ACTUAL stored predictor.
        Rhat = residual_coefficient_intervals(
            A0, A_i, b0, b_i, X, P.backend
        )
        g, residual_component_ranges = abs_bound_interval_vector_poly(
            Rhat, P.h, P.backend
        )

        # 2) Bound |P(q)| componentwise.
        s, predictor_ranges, predictor_range_meta = predictor_abs_bounds(
            X, P.h, P.backend, P.range_method, n
        )

        # 3) Add affine-input remainder effects:
        #       |r| <= g + e_b + E_A s.
        E_A = interval_remainder_abs_array(A_rem)
        e_b = interval_remainder_abs_array(b_rem)
        EA_s = nonnegative_matvec_upper(E_A, s, P.backend)

        rmax = np.empty(n)
        for j in range(n):
            temp = P.backend.add(point_interval(g[j]), point_interval(e_b[j])).hi
            rmax[j] = P.backend.add(point_interval(temp), point_interval(EA_s[j])).hi

        # 4) Two independent rigorous bounds for c >= |C r|.
        #
        # Legacy v0.6 path:
        #       |C r| <= |C| rmax.
        Cabs = np.abs(C)
        c_legacy = nonnegative_matvec_upper(Cabs, rmax, P.backend)

        # Sharper v0.6.1 path:
        #   Apply C to the interval-coefficient residual polynomial BEFORE
        #   destroying its signed/componentwise structure with absolute values.
        #
        #       C r = C r_hat + C Delta_b - (C Delta_A) P.
        #
        # Each term below is ranged as an interval, then combined before the
        # final absolute value is taken.
        CRhat = left_multiply_interval_vector_polynomial(C, Rhat, P.backend)
        CRhat_ranges, CRhat_range_meta = range_interval_vector_polynomial(
            CRhat, P.h, P.backend, P.range_method
        )

        Cb_rem = rigorous_matvec_point_interval(C, b_rem, P.backend)
        CA_rem = rigorous_matmul_point_interval(C, A_rem, P.backend)
        predictor_range_intervals = np.asarray(predictor_ranges, dtype=object)
        CArem_P = interval_matvec(
            CA_rem, predictor_range_intervals, P.backend
        )

        forcing_ranges = []
        c_direct = np.empty(n)
        for j in range(n):
            total = P.backend.add(CRhat_ranges[j], Cb_rem[j])
            total = P.backend.sub(total, CArem_P[j])
            forcing_ranges.append(total)
            c_direct[j] = abs_upper_interval(total)

        # Both are sound upper bounds on the same quantity.  Their componentwise
        # minimum is therefore also sound and gives a no-regression portfolio.
        c = np.minimum(c_legacy, c_direct)

        # 5) B >= |I - C A(q)|:
        #       |I-CA0| + sum_i h_i |C Ai| + |C| E_A.
        B = identity_minus_product_abs_upper(C, A0, P.backend)

        for i in range(P.m):
            CAi_abs = matrix_product_abs_upper(C, A_i[i], P.backend)
            term = scale_nonnegative_matrix_upper(P.h[i], CAi_abs, P.backend)
            B = add_nonnegative_matrices_upper(B, term, P.backend)

        CE = nonnegative_matmul_upper(Cabs, E_A, P.backend)
        B = add_nonnegative_matrices_upper(B, CE, P.backend)

        # 6) Rigorous infinity-norm upper bound and scalar correction.
        rho_upper = row_sum_norm_upper(B, P.backend)
        if not rho_upper < 1.0:
            raise RuntimeError(
                f"Semantic solve contraction failed: rigorous rho_upper={rho_upper:.17g} >= 1. "
                "A future generic controller should subdivide or switch certifiers."
            )

        eta_upper = float(np.max(c))
        delta_lower = P.backend.sub(
            point_interval(1.0), point_interval(rho_upper)
        ).lo
        if not delta_lower > 0.0:
            raise RuntimeError(
                "Contraction margin lost under outward rounding: "
                f"down(1-rho_upper)={delta_lower:.17g}."
            )

        d_upper = P.backend.div(
            point_interval(eta_upper), point_interval(delta_lower)
        ).hi

        # 7) Sharpen the common scalar radius into a componentwise rigorous
        #    correction vector without a verified linear solve.
        d_componentwise, correction_history = refine_componentwise_correction(
            c, B, d_upper, P.backend, max_iter=32
        )

        # 8) Return common representation with component-specific corrections.
        out = np.empty(n, dtype=object)
        for j in range(n):
            scalar_coeff = {
                alpha: float(vec[j])
                for alpha, vec in X.items()
                if vec[j] != 0.0
            }
            out[j] = ParameterizedEnclosure(
                scalar_coeff,
                Interval(-float(d_componentwise[j]), float(d_componentwise[j])),
                P.h, P.order, P.backend, P.range_method,
            )

        # Lightweight proof diagnostics attached to this context.  They are not
        # required by user model code, but are useful during development/tests.
        certificate = {
            "g": g,
            "s": s,
            "E_A": E_A,
            "e_b": e_b,
            "rmax": rmax,
            "c": c,
            "c_legacy": c_legacy,
            "c_direct": c_direct,
            "CRhat_ranges": CRhat_ranges,
            "CRhat_range_meta": CRhat_range_meta,
            "Cb_rem": Cb_rem,
            "CA_rem": CA_rem,
            "CArem_P": CArem_P,
            "forcing_ranges": forcing_ranges,
            "B": B,
            "rho_upper": rho_upper,
            "eta_upper": eta_upper,
            "delta_lower": delta_lower,
            "d_upper": d_upper,
            "d_componentwise": d_componentwise,
            "correction_history": correction_history,
            "residual_component_ranges": residual_component_ranges,
            "predictor_ranges": predictor_ranges,
            "predictor_range_meta": predictor_range_meta,
        }
        self.certificates.append(certificate)
        return out

    def __enter__(self):
        np.linalg.solve = self.traced_solve
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        np.linalg.solve = self.original_solve
        return False


# =============================================================================
# Public API
# =============================================================================

@dataclass
class AutoIAResult:
    lower: np.ndarray
    upper: np.ndarray
    objects: list
    backend_name: str
    machine_rigorous: bool
    certificates: list

    @property
    def width(self):
        return self.upper - self.lower

    def __repr__(self):
        lines = [
            f"AutoIAResult(backend='{self.backend_name}', "
            f"machine_rigorous={self.machine_rigorous})"
        ]
        for i, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            lines.append(
                f"  output[{i}] in [{lo:.12g}, {hi:.12g}] "
                f"width={hi-lo:.12g}"
            )
        return "\n".join(lines)


def rigorous_center_and_halfwidth(I: Interval, backend):
    """
    Choose any finite binary64 center, then rigorously enlarge to a symmetric box.

    The center is a coordinate choice and need not itself be a verified midpoint.
    The proof only needs [I.lo,I.hi] to lie inside [center-h, center+h].
    """
    if not np.isfinite(I.lo) or not np.isfinite(I.hi):
        raise ValueError("Version 0.6 reference implementation requires finite inputs.")

    # Overflow-safe-ish proposal midpoint.  If the usual formula overflows, use
    # lo + 0.5*(hi-lo) when possible; if that also fails, choose lo.
    center = float(0.5 * I.lo + 0.5 * I.hi)
    if not np.isfinite(center):
        center = float(I.lo)

    qbox = backend.sub(I, point_interval(center))
    h = max(abs(qbox.lo), abs(qbox.hi))
    if not np.isfinite(h):
        raise ValueError("Could not construct a finite centered parameter box.")
    return center, float(h)


def autoia(model, parameter_intervals, order=4, range_method="portfolio", backend=None):
    backend = backend if backend is not None else RigorousBinary64Backend()
    intervals = list(parameter_intervals)
    if not intervals:
        raise ValueError("At least one uncertain parameter is required.")

    centers = np.empty(len(intervals))
    h = np.empty(len(intervals))
    for i, I in enumerate(intervals):
        centers[i], h[i] = rigorous_center_and_halfwidth(I, backend)

    p = np.empty(len(intervals), dtype=object)
    for i, center in enumerate(centers):
        p[i] = ParameterizedEnclosure.parameter(
            center, i, h, order, backend, range_method
        )

    with SolveContext(p[0]) as solve_context:
        y = model(p)
        certificates = list(solve_context.certificates)

    arr = np.asarray(y, dtype=object)
    objs = []
    lo = []
    hi = []
    for v in arr.flat:
        if not isinstance(v, ParameterizedEnclosure):
            v = p[0]._coerce(v)
        r, _ = v.range(range_method)
        objs.append(v)
        lo.append(r.lo)
        hi.append(r.hi)

    return AutoIAResult(
        np.asarray(lo), np.asarray(hi), objs,
        backend.name, backend.machine_rigorous,
        certificates,
    )




# =============================================================================
# Version 0.10 representation-independent layer
# =============================================================================

# The Version 0.6.1 ParameterizedEnclosure remains the mature high-order
# polynomial backend.  In Version 0.10 it is deliberately treated as ONE
# representation rather than as the definition of an AutoIA value.
PolynomialEnclosure = ParameterizedEnclosure


class IntervalEnclosure:
    """
    Scalar interval-only representation.

        z in X

    This representation is deliberately dependency-blind.  It carries the same
    domain/backend metadata as PolynomialEnclosure so that representation changes
    are explicit and generic operations can dispatch without changing user code.
    """

    def __init__(self, interval, h, order, backend=None, range_method="portfolio"):
        if not isinstance(interval, Interval):
            raise TypeError("IntervalEnclosure requires an Interval.")
        self.interval = interval
        self.h = np.asarray(h, dtype=float)
        self.m = len(self.h)
        self.order = int(order)
        self.backend = backend if backend is not None else DEFAULT_BACKEND
        self.range_method = range_method

    @staticmethod
    def constant(value, h, order, backend=None, range_method="portfolio"):
        return IntervalEnclosure(
            point_interval(float(value)), h, order, backend, range_method
        )

    def copy(self):
        return IntervalEnclosure(
            self.interval, self.h.copy(), self.order,
            self.backend, self.range_method,
        )

    def range(self, method=None):
        return self.interval, {"method": "interval"}

    def __repr__(self):
        return (
            f"IntervalEnclosure([{self.interval.lo:.12g}, "
            f"{self.interval.hi:.12g}])"
        )


class PiecewiseEnclosure:
    """
    A finite union of already-certified scalar branch enclosures.

    Version 0.10 uses this first as a TOP-LEVEL subdivision representation.  It is
    intentionally not yet allowed to flow through arbitrary branch arithmetic,
    because branch alignment/path conditions need their own explicit semantics.
    """

    def __init__(self, branches, backend):
        self.branches = tuple(branches)
        if not self.branches:
            raise ValueError("PiecewiseEnclosure requires at least one branch.")
        self.backend = backend

    def range(self, method=None):
        intervals = []
        for branch in self.branches:
            r, _ = branch.range(method)
            intervals.append(r)
        lo = min(r.lo for r in intervals)
        hi = max(r.hi for r in intervals)
        return Interval(lo, hi), {
            "method": "piecewise-hull",
            "num_branches": len(self.branches),
            "branch_ranges": intervals,
        }

    def __repr__(self):
        r, _ = self.range()
        return (
            f"PiecewiseEnclosure(branches={len(self.branches)}, "
            f"hull=[{r.lo:.12g}, {r.hi:.12g}])"
        )


def _same_domain(a, b):
    return (
        len(a.h) == len(b.h)
        and np.all(a.h == b.h)
        and a.backend.name == b.backend.name
    )


def _safe_interval_representative(I):
    if I.lo == I.hi:
        return float(I.lo)
    c = float(0.5 * I.lo + 0.5 * I.hi)
    if not np.isfinite(c):
        c = float(I.lo)
    return float(min(max(c, I.lo), I.hi))


class ValidatedValue:
    """
    Representation-independent scalar AutoIA value.

    Universal contract
    ------------------
    The concrete representation must enclose the exact scalar program quantity
    for every admissible original input.  User model code sees ValidatedValue;
    it does not need to know whether the active representation is polynomial or
    interval.
    """

    __array_priority__ = 2000

    def __init__(self, representation):
        if not isinstance(representation, (PolynomialEnclosure, IntervalEnclosure)):
            raise TypeError(
                "ValidatedValue currently accepts PolynomialEnclosure or "
                "IntervalEnclosure in Version 0.10."
            )
        self.rep = representation

    @property
    def representation_name(self):
        if isinstance(self.rep, PolynomialEnclosure):
            return "polynomial"
        if isinstance(self.rep, IntervalEnclosure):
            return "interval"
        return type(self.rep).__name__

    @property
    def backend(self):
        return self.rep.backend

    @property
    def h(self):
        return self.rep.h

    @property
    def order(self):
        return self.rep.order

    @property
    def range_method(self):
        return self.rep.range_method

    def copy(self):
        return ValidatedValue(self.rep.copy())

    def range(self, method=None):
        return self.rep.range(method)

    def to_interval(self):
        if isinstance(self.rep, IntervalEnclosure):
            return self.rep.copy()
        r, _ = self.rep.range(self.range_method)
        return IntervalEnclosure(
            r, self.h, self.order, self.backend, self.range_method
        )

    def to_polynomial(self):
        if isinstance(self.rep, PolynomialEnclosure):
            return self.rep.copy()

        # Sound but dependency-destroying conversion:
        #   X = c + (X-c).
        I = self.rep.interval
        c = _safe_interval_representative(I)
        rem = self.backend.sub(I, point_interval(c))
        return PolynomialEnclosure(
            {zero_alpha(self.rep.m): c}, rem,
            self.h, self.order, self.backend, self.range_method,
        )

    def _constant_like(self, value):
        if isinstance(self.rep, PolynomialEnclosure):
            return ValidatedValue(PolynomialEnclosure.constant(
                value, self.h, self.order, self.backend, self.range_method
            ))
        return ValidatedValue(IntervalEnclosure.constant(
            value, self.h, self.order, self.backend, self.range_method
        ))

    def _coerce(self, other):
        if isinstance(other, ValidatedValue):
            if not _same_domain(self.rep, other.rep):
                raise ValueError("ValidatedValue domains/backends do not match.")
            return other
        return self._constant_like(other)

    def _binary(self, other, op):
        other = self._coerce(other)

        # Same-representation fast paths.
        if isinstance(self.rep, PolynomialEnclosure) and isinstance(other.rep, PolynomialEnclosure):
            if op == "add":
                return ValidatedValue(self.rep + other.rep)
            if op == "mul":
                return ValidatedValue(self.rep * other.rep)
            if op == "div":
                return ValidatedValue(self.rep / other.rep)

        if isinstance(self.rep, IntervalEnclosure) and isinstance(other.rep, IntervalEnclosure):
            if op == "add":
                I = self.backend.add(self.rep.interval, other.rep.interval)
            elif op == "mul":
                I = self.backend.mul(self.rep.interval, other.rep.interval)
            elif op == "div":
                I = self.backend.div(self.rep.interval, other.rep.interval)
            else:
                raise ValueError(op)
            return ValidatedValue(IntervalEnclosure(
                I, self.h, min(self.order, other.order),
                self.backend, self.range_method,
            ))

        # Mixed representation policy for the prototype: preserve the richer
        # polynomial structure, converting the interval value to a constant
        # polynomial plus rigorous remainder.  This is sound but cannot restore
        # provenance that was already lost by intervalization.
        a = self.to_polynomial()
        b = other.to_polynomial()
        if op == "add":
            return ValidatedValue(a + b)
        if op == "mul":
            return ValidatedValue(a * b)
        if op == "div":
            return ValidatedValue(a / b)
        raise ValueError(op)

    def __add__(self, other):
        return self._binary(other, "add")

    __radd__ = __add__

    def __neg__(self):
        if isinstance(self.rep, PolynomialEnclosure):
            return ValidatedValue(-self.rep)
        return ValidatedValue(IntervalEnclosure(
            interval_neg(self.rep.interval), self.h, self.order,
            self.backend, self.range_method,
        ))

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        return self._binary(other, "mul")

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary(other, "div")

    def __rtruediv__(self, other):
        return self._coerce(other)._binary(self, "div")

    def __pow__(self, power):
        if not isinstance(power, int) or power < 0:
            raise NotImplementedError(
                "Version 0.10 supports nonnegative integer powers."
            )
        out = self._constant_like(1.0)
        for _ in range(power):
            out = out * self
        return out

    def __repr__(self):
        r, _ = self.range()
        return (
            f"ValidatedValue(rep='{self.representation_name}', "
            f"range=[{r.lo:.12g}, {r.hi:.12g}])"
        )


def contains_validated(arr):
    obj = np.asarray(arr, dtype=object)
    return any(isinstance(v, ValidatedValue) for v in obj.flat)


def _first_validated(arrays):
    for arr in arrays:
        obj = np.asarray(arr, dtype=object)
        for v in obj.flat:
            if isinstance(v, ValidatedValue):
                return v
    return None


def _unwrap_for_polynomial(arr, prototype):
    obj = np.asarray(arr, dtype=object)
    out = np.empty(obj.shape, dtype=object)
    for idx in np.ndindex(obj.shape):
        v = obj[idx]
        if isinstance(v, ValidatedValue):
            out[idx] = v.to_polynomial()
        else:
            out[idx] = v
    return out


def _unwrap_for_interval(arr, prototype):
    obj = np.asarray(arr, dtype=object)
    out = np.empty(obj.shape, dtype=object)
    for idx in np.ndindex(obj.shape):
        v = obj[idx]
        if isinstance(v, ValidatedValue):
            out[idx] = v.to_interval().interval
        else:
            out[idx] = point_interval(float(v))
    return out


def _all_active_representations(arrays):
    kinds = set()
    for arr in arrays:
        obj = np.asarray(arr, dtype=object)
        for v in obj.flat:
            if isinstance(v, ValidatedValue):
                kinds.add(v.representation_name)
    return kinds


def _interval_matrix_midpoint(AI):
    AI = np.asarray(AI, dtype=object)
    A0 = np.empty(AI.shape)
    for idx in np.ndindex(AI.shape):
        A0[idx] = _safe_interval_representative(AI[idx])
    return A0


def _interval_vector_midpoint(bI):
    bI = np.asarray(bI, dtype=object)
    b0 = np.empty(bI.shape)
    for idx in np.ndindex(bI.shape):
        b0[idx] = _safe_interval_representative(bI[idx])
    return b0


def interval_semantic_solve(AI, bI, prototype, original_solve):
    """
    Rigorous semantic linear solve for dependency-blind interval A and b.

    This certifies every matrix/right-hand side in the independently boxed family.
    It therefore may fail even when the original parameterized family is perfectly
    nonsingular; that failure is an informative consequence of representation loss.
    """
    AI = np.asarray(AI, dtype=object)
    bI = np.asarray(bI, dtype=object)
    backend = prototype.backend

    if AI.ndim != 2 or AI.shape[0] != AI.shape[1]:
        raise ValueError("A must be square.")
    if bI.ndim != 1 or AI.shape[0] != len(bI):
        raise ValueError("A and b dimensions do not match.")

    n = len(bI)
    A0 = _interval_matrix_midpoint(AI)
    b0 = _interval_vector_midpoint(bI)

    # Proposal calculations only.
    x0 = original_solve(A0, b0)
    C = original_solve(A0, np.eye(n))
    if np.any(~np.isfinite(x0)) or np.any(~np.isfinite(C)):
        raise FloatingPointError("Nonfinite interval-solve proposal/preconditioner.")

    # Trusted residual r = b - A x0.
    x0I = np.asarray([point_interval(v) for v in x0], dtype=object)
    Ax0 = interval_matvec(AI, x0I, backend)
    rI = np.empty(n, dtype=object)
    for j in range(n):
        rI[j] = backend.sub(bI[j], Ax0[j])

    CrI = rigorous_matvec_point_interval(C, rI, backend)
    c = np.asarray([abs_upper_interval(v) for v in CrI], dtype=float)

    # Trusted D = I - C A and B >= |D|.
    CAI = rigorous_matmul_point_interval(C, AI, backend)
    B = np.empty((n, n))
    DI = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            delta = 1.0 if i == j else 0.0
            DI[i, j] = backend.sub(point_interval(delta), CAI[i, j])
            B[i, j] = abs_upper_interval(DI[i, j])

    rho_upper = row_sum_norm_upper(B, backend)
    if not rho_upper < 1.0:
        raise RuntimeError(
            "Interval semantic solve contraction failed: "
            f"rigorous rho_upper={rho_upper:.17g} >= 1.  "
            "The interval representation has discarded parameter dependence; "
            "switch to the polynomial representation or subdivide."
        )

    eta_upper = float(np.max(c))
    delta_lower = backend.sub(
        point_interval(1.0), point_interval(rho_upper)
    ).lo
    if not delta_lower > 0.0:
        raise RuntimeError("Contraction margin lost under outward rounding.")

    d_scalar = backend.div(
        point_interval(eta_upper), point_interval(delta_lower)
    ).hi
    d_componentwise, history = refine_componentwise_correction(
        c, B, d_scalar, backend, max_iter=32
    )

    out = np.empty(n, dtype=object)
    for j in range(n):
        corr = Interval(-float(d_componentwise[j]), float(d_componentwise[j]))
        xI = backend.add(point_interval(float(x0[j])), corr)
        out[j] = IntervalEnclosure(
            xI, prototype.h, prototype.order,
            backend, prototype.range_method,
        )

    certificate = {
        "representation": "interval",
        "A_interval": AI,
        "b_interval": bI,
        "A0_proposal": A0,
        "b0_proposal": b0,
        "x0_proposal": x0,
        "residual_interval": rI,
        "Cr_interval": CrI,
        "c": c,
        "D_interval": DI,
        "B": B,
        "rho_upper": rho_upper,
        "d_scalar": d_scalar,
        "d_componentwise": d_componentwise,
        "correction_history": history,
    }
    return out, certificate


class GenericSolveContext:
    """Representation-aware semantic dispatcher for np.linalg.solve."""

    def __init__(self, prototype):
        self.prototype = prototype
        self.original_solve = np.linalg.solve
        self.certificates = []

    def traced_solve(self, A, b):
        if not (contains_validated(A) or contains_validated(b)):
            return self.original_solve(A, b)

        P = self.prototype
        kinds = _all_active_representations((A, b))

        # Polynomial wins for mixed inputs because interval -> polynomial+remainder
        # is sound.  Lost provenance is not magically recovered, but the semantic
        # polynomial certifier can still consume the converted enclosure.
        if "polynomial" in kinds:
            A_poly = _unwrap_for_polynomial(A, P)
            b_poly = _unwrap_for_polynomial(b, P)
            poly_prototype = P.to_polynomial()

            ctx = SolveContext(poly_prototype)
            # SolveContext was written before the generic dispatcher and normally
            # captures np.linalg.solve at construction time.  Force it to use the
            # unpatched numerical solve for proposal calculations.
            ctx.original_solve = self.original_solve
            out_rep = ctx.traced_solve(A_poly, b_poly)
            for cert in ctx.certificates:
                cert = dict(cert)
                cert["representation"] = "polynomial"
                self.certificates.append(cert)
            return np.asarray([ValidatedValue(v) for v in out_rep], dtype=object)

        # Otherwise the active uncertain representation is interval-only.
        AI = _unwrap_for_interval(A, P)
        bI = _unwrap_for_interval(b, P)
        interval_prototype = P.to_interval()
        out_rep, cert = interval_semantic_solve(
            AI, bI, interval_prototype, self.original_solve
        )
        self.certificates.append(cert)
        return np.asarray([ValidatedValue(v) for v in out_rep], dtype=object)

    def __enter__(self):
        np.linalg.solve = self.traced_solve
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        np.linalg.solve = self.original_solve
        return False


@dataclass
class AutoIAResultV11:
    lower: np.ndarray
    upper: np.ndarray
    objects: list
    backend_name: str
    machine_rigorous: bool
    certificates: list
    requested_representation: str
    range_policy: RangePolicy
    final_range_diagnostics: list
    range_policy_events: list

    @property
    def width(self):
        return self.upper - self.lower

    @property
    def representation_names(self):
        return [v.representation_name for v in self.objects]

    def __repr__(self):
        lines = [
            f"AutoIAResultV11(requested_representation='{self.requested_representation}', "
            f"backend='{self.backend_name}', machine_rigorous={self.machine_rigorous}, "
            f"internal_range='{self.range_policy.internal_method}', "
            f"final_range='{self.range_policy.final_method}')"
        ]
        for i, (lo, hi, obj, meta) in enumerate(zip(
            self.lower, self.upper, self.objects, self.final_range_diagnostics
        )):
            extra = ""
            if meta.get("adaptive"):
                extra = (
                    f" bernstein_used={meta.get('bernstein_used')}"
                    f" bern_cost={meta.get('estimated_bernstein_coefficients')}"
                )
            lines.append(
                f"  output[{i}] rep={obj.representation_name} "
                f"in [{lo:.12g}, {hi:.12g}] width={hi-lo:.12g}{extra}"
            )
        return "\n".join(lines)


def _resolve_range_policy(range_method, internal_range_method,
                          final_range_method, range_policy):
    if range_policy is not None:
        if any(v is not None for v in (range_method, internal_range_method, final_range_method)):
            raise ValueError(
                "Pass either range_policy or the range_method/internal/final options, not both."
            )
        return range_policy

    # Backward compatibility: an explicitly supplied Version 0.8 range_method
    # means use the same method everywhere, exactly as before.
    if range_method is not None:
        return RangePolicy(
            internal_method=range_method,
            final_method=range_method,
        )

    return RangePolicy(
        internal_method=(internal_range_method or "termwise"),
        final_method=(final_range_method or "adaptive"),
    )


def autoia(model, parameter_intervals, order=4, range_method=None,
           backend=None, representation="polynomial",
           internal_range_method=None, final_range_method=None,
           range_policy=None):
    """Representation-independent Version 0.11 entry point.

    The Version 0.10 default deliberately separates two jobs:

      * internal proof/algebra ranging: cheap termwise bounds,
      * final user-visible ranging: adaptive termwise/monotonicity/Bernstein.

    Passing the legacy ``range_method=...`` argument explicitly preserves the
    Version 0.8 single-method behavior for comparison/regression tests.
    """
    backend = backend if backend is not None else RigorousBinary64Backend()
    policy = _resolve_range_policy(
        range_method, internal_range_method, final_range_method, range_policy
    )

    intervals = list(parameter_intervals)
    if not intervals:
        raise ValueError("At least one uncertain parameter is required.")
    if representation not in {"polynomial", "interval"}:
        raise ValueError("representation must be 'polynomial' or 'interval'.")

    centers = np.empty(len(intervals))
    h = np.empty(len(intervals))
    for i, I in enumerate(intervals):
        centers[i], h[i] = rigorous_center_and_halfwidth(I, backend)

    p = np.empty(len(intervals), dtype=object)
    for i, (I, center) in enumerate(zip(intervals, centers)):
        if representation == "polynomial":
            rep = PolynomialEnclosure.parameter(
                center, i, h, order, backend, policy.internal_method
            )
        else:
            rep = IntervalEnclosure(
                I, h, order, backend, policy.internal_method
            )
        p[i] = ValidatedValue(rep)

    policy_context = RangePolicyContext(policy)
    with policy_context:
        with GenericSolveContext(p[0]) as solve_context:
            y = model(p)
            certificates = list(solve_context.certificates)
    range_policy_events = list(policy_context.events)

    arr = np.asarray(y, dtype=object)
    objs = []
    lo = []
    hi = []
    final_meta = []
    for v in arr.flat:
        if not isinstance(v, ValidatedValue):
            v = p[0]._constant_like(v)
        r, meta = adaptive_final_range(v, policy)
        objs.append(v)
        lo.append(r.lo)
        hi.append(r.hi)
        final_meta.append(meta)

    return AutoIAResultV11(
        np.asarray(lo), np.asarray(hi), objs,
        backend.name, backend.machine_rigorous,
        certificates, representation, policy, final_meta, range_policy_events,
    )


def _partition_interval(I, count):
    count = int(count)
    if count < 1:
        raise ValueError("Subdivision counts must be positive integers.")
    if count == 1:
        return [I]
    edges = []
    for k in range(count + 1):
        if k == 0:
            x = float(I.lo)
        elif k == count:
            x = float(I.hi)
        else:
            # Any binary64 cut is acceptable: adjacent closed boxes share it.
            t = k / count
            x = float((1.0 - t) * I.lo + t * I.hi)
            x = min(max(x, I.lo), I.hi)
        edges.append(x)
    # Enforce nondecreasing shared boundaries.
    for k in range(1, len(edges)):
        if edges[k] < edges[k-1]:
            edges[k] = edges[k-1]
    return [Interval(edges[k], edges[k+1]) for k in range(count)]


def subdivide_parameter_box(parameter_intervals, split_counts):
    intervals = list(parameter_intervals)
    if np.isscalar(split_counts):
        split_counts = [int(split_counts)] * len(intervals)
    split_counts = list(split_counts)
    if len(split_counts) != len(intervals):
        raise ValueError("split_counts must have one entry per parameter.")
    parts = [_partition_interval(I, n) for I, n in zip(intervals, split_counts)]
    return [list(box) for box in itertools.product(*parts)]


@dataclass
class AutoIAPiecewiseResult:
    lower: np.ndarray
    upper: np.ndarray
    objects: list
    branch_results: list
    backend_name: str
    machine_rigorous: bool
    requested_representation: str

    @property
    def width(self):
        return self.upper - self.lower

    def __repr__(self):
        lines = [
            f"AutoIAPiecewiseResult(branches={len(self.branch_results)}, "
            f"requested_representation='{self.requested_representation}', "
            f"machine_rigorous={self.machine_rigorous})"
        ]
        for i, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            lines.append(
                f"  output[{i}] piecewise hull in [{lo:.12g}, {hi:.12g}] "
                f"width={hi-lo:.12g}"
            )
        return "\n".join(lines)


def autoia_subdivide(model, parameter_intervals, split_counts,
                     order=4, range_method=None, backend=None,
                     representation="polynomial", internal_range_method=None,
                     final_range_method=None, range_policy=None):
    """
    Top-level generic subdivision prototype.

    Each subbox is independently certified with unchanged model source code.  The
    returned PiecewiseEnclosure retains the finite union; lower/upper are only its
    convenient interval hull.
    """
    backend = backend if backend is not None else RigorousBinary64Backend()
    boxes = subdivide_parameter_box(parameter_intervals, split_counts)
    branch_results = []
    for box in boxes:
        branch_results.append(autoia(
            model, box, order=order, range_method=range_method,
            backend=backend, representation=representation,
            internal_range_method=internal_range_method,
            final_range_method=final_range_method,
            range_policy=range_policy,
        ))

    nout = len(branch_results[0].objects)
    objects = []
    lo = np.empty(nout)
    hi = np.empty(nout)
    for j in range(nout):
        branches = [r.objects[j] for r in branch_results]
        piece = PiecewiseEnclosure(branches, backend)
        objects.append(piece)
        # Use each branch's already-computed final policy range rather than
        # silently reverting to its cheap internal range method.
        lo[j] = min(r.lower[j] for r in branch_results)
        hi[j] = max(r.upper[j] for r in branch_results)

    return AutoIAPiecewiseResult(
        lo, hi, objects, branch_results,
        backend.name, backend.machine_rigorous,
        representation,
    )



# =============================================================================
# Version 0.11 targeted adaptive subdivision controller
# =============================================================================

@dataclass(frozen=True)
class SubdivisionPolicy:
    """Generic one-step-lookahead subdivision policy.

    Subdivision is considered only for branches whose adaptive final range has
    unresolved monotonicity directions and whose monotonicity-reduced Bernstein
    calculation exceeds the current RangePolicy cost cap.

    Parameters
    ----------
    max_branches : int
        Maximum number of retained certified subboxes.
    max_depth : int
        Maximum binary subdivision depth along any branch.
    max_probe_directions : int
        Maximum unresolved input directions probed at one controller decision.
        Candidate directions are pre-ranked by a cheap derivative-ambiguity score.
    min_structural_gain : float
        Minimum normalized reduction in the branch's worst reduced-Bernstein cost
        required to accept a split when there is no immediate width improvement.
    min_relative_width_gain : float
        Minimum relative reduction in the branch output-hull width that can by
        itself justify a split.

    All quantities here control strategy only.  Every retained branch is an
    independently certified AutoIA result, so soundness does not depend on the
    controller choosing a good split.
    """
    max_branches: int = 16
    max_depth: int = 6
    max_probe_directions: int = 2
    min_structural_gain: float = 0.05
    min_relative_width_gain: float = 1.0e-3

    def __post_init__(self):
        if self.max_branches < 1:
            raise ValueError("max_branches must be positive.")
        if self.max_depth < 0:
            raise ValueError("max_depth must be nonnegative.")
        if self.max_probe_directions < 1:
            raise ValueError("max_probe_directions must be positive.")
        if self.min_structural_gain < 0.0:
            raise ValueError("min_structural_gain must be nonnegative.")
        if self.min_relative_width_gain < 0.0:
            raise ValueError("min_relative_width_gain must be nonnegative.")


@dataclass
class _AdaptiveBranch:
    box: list
    result: object
    depth: int
    branch_id: int


def _branch_controller_metrics(result):
    """Aggregate final-range diagnostics into branch-level controller metrics."""
    unresolved_union = set()
    worst_reduced_cost = 0
    full_cost = 0
    capped_outputs = []
    output_unresolved = []

    for j, meta in enumerate(result.final_range_diagnostics):
        unresolved = tuple(int(i) for i in meta.get("unresolved_variables", ()))
        unresolved_union.update(unresolved)
        output_unresolved.append(unresolved)
        rcost = int(meta.get("estimated_reduced_bernstein_coefficients", 0) or 0)
        fcost = int(meta.get("estimated_bernstein_coefficients", 0) or 0)
        worst_reduced_cost = max(worst_reduced_cost, rcost)
        full_cost = max(full_cost, fcost)
        if meta.get("bernstein_reason") == "reduced-cost-cap":
            capped_outputs.append(j)

    widths = np.asarray(result.width, dtype=float)
    total_width = float(np.sum(np.maximum(widths, 0.0)))
    max_width = float(np.max(np.maximum(widths, 0.0))) if len(widths) else 0.0

    return {
        "unresolved_variables": tuple(sorted(unresolved_union)),
        "num_unresolved": len(unresolved_union),
        "worst_reduced_bernstein_cost": int(worst_reduced_cost),
        "worst_full_bernstein_cost": int(full_cost),
        "capped_outputs": tuple(capped_outputs),
        "needs_subdivision": bool(capped_outputs and unresolved_union),
        "total_width": total_width,
        "max_width": max_width,
        "output_unresolved": tuple(output_unresolved),
    }


def _direction_ambiguity_scores(result, box):
    """Cheap pre-ranking of unresolved directions from derivative uncertainty.

    For an unresolved derivative interval D_i=[l_i,u_i], the factor

        width(D_i) * halfwidth(q_i)

    has output units and estimates how much unresolved directional variation can
    be associated with that coordinate.  Scores are normalized by each output's
    current width before being summed across outputs.
    """
    m = len(box)
    scores = np.zeros(m, dtype=float)
    for j, meta in enumerate(result.final_range_diagnostics):
        mm = meta.get("monotonicity_meta", {}) or {}
        signs = tuple(int(s) for s in mm.get("signs", ()))
        dranges = tuple(mm.get("derivative_ranges", ()))
        scale = max(float(result.width[j]), np.finfo(float).tiny)
        obj = result.objects[j]
        rep = getattr(obj, "rep", None)
        h = getattr(rep, "h", None)
        if h is None:
            continue
        for i in range(min(m, len(signs), len(dranges))):
            if signs[i] != 0:
                continue
            D = dranges[i]
            span = max(0.0, float(D.hi - D.lo))
            scores[i] += span * float(h[i]) / scale
    return scores


def _split_one_box(box, direction):
    left_right = _partition_interval(box[direction], 2)
    if len(left_right) != 2:
        raise RuntimeError("Binary split did not produce two subintervals.")
    out = []
    for part in left_right:
        child = list(box)
        child[direction] = part
        out.append(child)
    return out


def _combined_child_hull_metrics(child_results):
    if not child_results:
        raise ValueError("At least one child result is required.")
    lower = np.min(np.vstack([r.lower for r in child_results]), axis=0)
    upper = np.max(np.vstack([r.upper for r in child_results]), axis=0)
    widths = np.maximum(upper - lower, 0.0)
    metrics = [_branch_controller_metrics(r) for r in child_results]
    unresolved_union = set()
    for M in metrics:
        unresolved_union.update(M["unresolved_variables"])
    return {
        "lower": lower,
        "upper": upper,
        "width": widths,
        "total_width": float(np.sum(widths)),
        "max_width": float(np.max(widths)) if len(widths) else 0.0,
        "num_unresolved": len(unresolved_union),
        "unresolved_variables": tuple(sorted(unresolved_union)),
        "worst_reduced_bernstein_cost": max(
            M["worst_reduced_bernstein_cost"] for M in metrics
        ),
        "needs_subdivision": any(M["needs_subdivision"] for M in metrics),
    }


def _evaluate_subdivision_candidate(model, branch, direction, *, order, backend,
                                    representation, range_policy):
    child_boxes = _split_one_box(branch.box, direction)
    child_results = [
        autoia(
            model, child_box, order=order, backend=backend,
            representation=representation, range_policy=range_policy,
        )
        for child_box in child_boxes
    ]
    combined = _combined_child_hull_metrics(child_results)
    parent = _branch_controller_metrics(branch.result)

    parent_cost = max(1, parent["worst_reduced_bernstein_cost"])
    child_cost = combined["worst_reduced_bernstein_cost"]
    structural_gain = max(0.0, (parent_cost - child_cost) / parent_cost)

    parent_width = max(parent["total_width"], np.finfo(float).tiny)
    width_gain = max(0.0, (parent["total_width"] - combined["total_width"]) / parent_width)

    unresolved_gain = max(0, parent["num_unresolved"] - combined["num_unresolved"])

    return {
        "direction": int(direction),
        "child_boxes": child_boxes,
        "child_results": child_results,
        "combined": combined,
        "structural_gain": float(structural_gain),
        "relative_width_gain": float(width_gain),
        "unresolved_gain": int(unresolved_gain),
        "parent_cost": int(parent["worst_reduced_bernstein_cost"]),
        "child_cost": int(child_cost),
        "parent_unresolved": int(parent["num_unresolved"]),
        "child_unresolved": int(combined["num_unresolved"]),
    }


@dataclass
class AutoIAAdaptiveSubdivisionResult:
    lower: np.ndarray
    upper: np.ndarray
    objects: list
    branch_results: list
    branch_boxes: list
    branch_depths: list
    backend_name: str
    machine_rigorous: bool
    requested_representation: str
    range_policy: RangePolicy
    subdivision_policy: SubdivisionPolicy
    subdivision_events: list

    @property
    def width(self):
        return self.upper - self.lower

    def __repr__(self):
        lines = [
            f"AutoIAAdaptiveSubdivisionResult(branches={len(self.branch_results)}, "
            f"requested_representation='{self.requested_representation}', "
            f"machine_rigorous={self.machine_rigorous})"
        ]
        for i, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            lines.append(
                f"  output[{i}] piecewise hull in [{lo:.12g}, {hi:.12g}] "
                f"width={hi-lo:.12g}"
            )
        lines.append(f"  subdivision_events={len(self.subdivision_events)}")
        return "\n".join(lines)


def autoia_adaptive_subdivide(model, parameter_intervals, order=4, *, backend=None,
                               representation="polynomial", range_policy=None,
                               subdivision_policy=None):
    """Targeted top-level adaptive subdivision for Version 0.11.

    The controller acts only on final range diagnostics.  A branch is eligible
    when monotonicity leaves unresolved directions AND the corresponding reduced
    Bernstein cost exceeds ``range_policy.bernstein_max_coefficients``.

    Candidate splits are restricted to those unresolved directions.  For this
    reference implementation, each candidate is evaluated by actually certifying
    the two half-boxes.  The selected split is therefore based on observed
    reduction in unresolved structure, reduced-Bernstein cost, and output-hull
    width rather than on an unvalidated surrogate score.

    The return value retains every certified branch.  Its lower/upper arrays are
    merely the interval hull of that finite union.
    """
    backend = backend if backend is not None else RigorousBinary64Backend()
    range_policy = range_policy if range_policy is not None else RangePolicy()
    subdivision_policy = (
        subdivision_policy if subdivision_policy is not None else SubdivisionPolicy()
    )
    intervals = list(parameter_intervals)

    root_result = autoia(
        model, intervals, order=order, backend=backend,
        representation=representation, range_policy=range_policy,
    )
    branches = [_AdaptiveBranch(intervals, root_result, 0, 0)]
    next_branch_id = 1
    events = []

    while len(branches) < subdivision_policy.max_branches:
        eligible = []
        for idx, branch in enumerate(branches):
            M = _branch_controller_metrics(branch.result)
            if M["needs_subdivision"] and branch.depth < subdivision_policy.max_depth:
                eligible.append((idx, branch, M))
        if not eligible:
            break

        # Work first on the branch with the largest unresolved Bernstein burden;
        # break ties by total output width.
        eligible.sort(
            key=lambda item: (
                item[2]["worst_reduced_bernstein_cost"],
                item[2]["total_width"],
            ),
            reverse=True,
        )
        branch_index, branch, parent_metrics = eligible[0]

        unresolved = list(parent_metrics["unresolved_variables"])
        ambiguity = _direction_ambiguity_scores(branch.result, branch.box)
        unresolved.sort(key=lambda i: ambiguity[i], reverse=True)
        candidates = unresolved[:subdivision_policy.max_probe_directions]
        if not candidates:
            break

        probe_records = []
        for direction in candidates:
            try:
                probe = _evaluate_subdivision_candidate(
                    model, branch, direction, order=order, backend=backend,
                    representation=representation, range_policy=range_policy,
                )
                probe["ambiguity_score"] = float(ambiguity[direction])
                probe_records.append(probe)
            except Exception as exc:
                # Candidate failure does not invalidate other sound branches.
                probe_records.append({
                    "direction": int(direction),
                    "failed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "ambiguity_score": float(ambiguity[direction]),
                })

        successful = [p for p in probe_records if not p.get("failed", False)]
        if not successful:
            events.append({
                "branch_id": branch.branch_id,
                "depth": branch.depth,
                "action": "stop-no-successful-candidate",
                "parent_metrics": parent_metrics,
                "probes": probe_records,
            })
            break

        # Lexicographic controller objective:
        #   1. resolve as many monotonicity directions as possible,
        #   2. reduce the worst reduced-Bernstein cost,
        #   3. tighten the output hull,
        #   4. use derivative ambiguity as a final tie breaker.
        best = max(
            successful,
            key=lambda p: (
                p["unresolved_gain"],
                p["structural_gain"],
                p["relative_width_gain"],
                p["ambiguity_score"],
            ),
        )

        accept = (
            best["unresolved_gain"] > 0
            or best["structural_gain"] >= subdivision_policy.min_structural_gain
            or best["relative_width_gain"] >= subdivision_policy.min_relative_width_gain
        )

        event = {
            "branch_id": branch.branch_id,
            "depth": branch.depth,
            "action": "split" if accept else "stop-insufficient-gain",
            "selected_direction": int(best["direction"]),
            "parent_metrics": parent_metrics,
            "selected_metrics": {
                k: best[k] for k in (
                    "structural_gain", "relative_width_gain", "unresolved_gain",
                    "parent_cost", "child_cost", "parent_unresolved",
                    "child_unresolved", "ambiguity_score"
                )
            },
            "probes": [
                {k: v for k, v in p.items()
                 if k not in {"child_boxes", "child_results", "combined"}}
                for p in probe_records
            ],
        }
        events.append(event)

        if not accept:
            break
        if len(branches) + 1 > subdivision_policy.max_branches:
            break

        # Reuse the already certified selected probe children; no rerun.
        children = []
        for child_box, child_result in zip(best["child_boxes"], best["child_results"]):
            children.append(_AdaptiveBranch(
                child_box, child_result, branch.depth + 1, next_branch_id
            ))
            next_branch_id += 1
        branches[branch_index:branch_index + 1] = children

    branch_results = [b.result for b in branches]
    branch_boxes = [b.box for b in branches]
    branch_depths = [b.depth for b in branches]
    if not branch_results:
        raise RuntimeError("Adaptive subdivision produced no branches.")

    nout = len(branch_results[0].objects)
    lower = np.empty(nout)
    upper = np.empty(nout)
    objects = []
    for j in range(nout):
        lower[j] = min(r.lower[j] for r in branch_results)
        upper[j] = max(r.upper[j] for r in branch_results)
        objects.append(PiecewiseEnclosure([r.objects[j] for r in branch_results], backend))

    return AutoIAAdaptiveSubdivisionResult(
        lower, upper, objects, branch_results, branch_boxes, branch_depths,
        backend.name, backend.machine_rigorous, representation,
        range_policy, subdivision_policy, events,
    )

# =============================================================================
# Demonstrations / regression tests
# =============================================================================


def spring_model(p):
    k1, k2, F = p
    A = np.array([
        [k1 + k2, -k2],
        [-k2,          k2],
    ])
    b = np.array([F, 0.0])
    return np.linalg.solve(A, b)


def spring_with_postprocessing(p):
    k1, k2, F = p
    A = np.array([
        [k1 + k2, -k2],
        [-k2,          k2],
    ])
    b = np.array([F, 0.0])
    u = np.linalg.solve(A, b)
    return np.array([u[0], F*u[0], u[0]**2], dtype=object)


def generic_coupled_model(p):
    p1, p2 = p
    A = np.array([
        [2.0 + p1, 0.2 * p2],
        [0.2 * p2, 1.5 - p1],
    ])
    b = np.array([1.0 + 0.1 * p2, 0.5 + p1])
    u = np.linalg.solve(A, b)
    return np.array([u[0], u[0]*u[1], u[1]**2], dtype=object)



def manual_solve_2x2(A, b):
    """
    Hand-coded exact 2x2 algebraic solve.  Deliberately does NOT call np.linalg.solve.

    This is useful for testing implementation-style independence: AutoIA sees only
    elementary +,-,*,/ operations and receives no semantic hint that the block is
    a linear solve.
    """
    a = A[0, 0]
    bb = A[0, 1]
    c = A[1, 0]
    d = A[1, 1]
    det = a*d - bb*c
    x0 = (b[0]*d - bb*b[1]) / det
    x1 = (a*b[1] - b[0]*c) / det
    return np.array([x0, x1], dtype=object)


def spring_model_manual(p):
    k1, k2, F = p
    A = np.array([
        [k1 + k2, -k2],
        [-k2,          k2],
    ])
    b = np.array([F, 0.0])
    return manual_solve_2x2(A, b)


def generic_coupled_model_manual(p):
    p1, p2 = p
    A = np.array([
        [2.0 + p1, 0.2 * p2],
        [0.2 * p2, 1.5 - p1],
    ])
    b = np.array([1.0 + 0.1 * p2, 0.5 + p1])
    u = manual_solve_2x2(A, b)
    return np.array([u[0], u[0]*u[1], u[1]**2], dtype=object)


def generic_ratio_model(p):
    """Standalone nonlinear rational expression for division regression testing."""
    x, y = p
    numerator = 1.0 + 0.3*x - 0.2*y + 0.1*x*y
    denominator = 2.0 + 0.15*x + 0.1*y - 0.05*x*y + 0.02*x*x
    return numerator / denominator


def plain_generic_ratio(x, y):
    numerator = 1.0 + 0.3*x - 0.2*y + 0.1*x*y
    denominator = 2.0 + 0.15*x + 0.1*y - 0.05*x*y + 0.02*x*x
    return numerator / denominator


def plain_generic_coupled(p1, p2):
    A = np.array([
        [2.0 + p1, 0.2 * p2],
        [0.2 * p2, 1.5 - p1],
    ], dtype=float)
    b = np.array([1.0 + 0.1 * p2, 0.5 + p1], dtype=float)
    u = np.linalg.solve(A, b)
    return np.array([u[0], u[0]*u[1], u[1]**2])


def random_sample_containment_generic(result, nsamples=2000, seed=17):
    rng = np.random.default_rng(seed)
    for _ in range(nsamples):
        p1 = rng.uniform(-0.2, 0.2)
        p2 = rng.uniform(-0.5, 0.5)
        y = plain_generic_coupled(p1, p2)
        if np.any(y < result.lower) or np.any(y > result.upper):
            raise AssertionError(
                f"Containment failure at p=({p1},{p2}), y={y}."
            )


if __name__ == "__main__":
    print("\nAUTOIA VERSION 0.10 -- STRUCTURE-AWARE RANGE POLICY")
    print("==============================================\n")

    backend = RigorousBinary64Backend()
    coupled_box = [Interval(-0.2, 0.2), Interval(-0.5, 0.5)]
    spring_box = [
        Interval(90.0, 110.0),
        Interval(900.0, 1100.0),
        Interval(9.5, 10.5),
    ]

    print("1) Coupled system through semantic np.linalg.solve")
    CS = autoia(
        generic_coupled_model, coupled_box,
        order=5, backend=backend,
        representation="polynomial",
    )
    print(CS)
    random_sample_containment_generic(CS)
    print("   diagnostic: 2000/2000 random exact evaluations contained\n")

    print("2) SAME coupled mathematics through hand-coded 2x2 algebra")
    CM = autoia(
        generic_coupled_model_manual, coupled_box,
        order=5, backend=backend,
        representation="polynomial",
    )
    print(CM)
    random_sample_containment_generic(CM)
    print("   diagnostic: 2000/2000 random exact evaluations contained")
    print("   width ratios manual/semantic =", CM.width / CS.width)
    print("   semantic certificates seen =", len(CS.certificates))
    print("   manual semantic certificates seen =", len(CM.certificates))
    print()

    print("3) Spring through semantic np.linalg.solve")
    SS = autoia(
        spring_model, spring_box,
        order=6, backend=backend,
        representation="polynomial",
    )
    print(SS)
    print()

    print("4) SAME spring mathematics through hand-coded 2x2 algebra")
    SM = autoia(
        spring_model_manual, spring_box,
        order=6, backend=backend,
        representation="polynomial",
    )
    print(SM)
    exact_lo = 9.5 / 110.0
    exact_hi = 10.5 / 90.0
    for j in range(2):
        if not (SM.lower[j] <= exact_lo <= exact_hi <= SM.upper[j]):
            raise AssertionError("Manual spring enclosure missed analytical range.")
    print("   analytical displacement range =", (exact_lo, exact_hi))
    print("   width ratios manual/semantic =", SM.width / SS.width)
    print("   semantic certificates seen =", len(SS.certificates))
    print("   manual semantic certificates seen =", len(SM.certificates))
    print()

    print("5) Interval representation also gets generic division")
    CI = autoia(
        generic_coupled_model_manual, coupled_box,
        order=5, backend=backend,
        representation="interval",
    )
    print(CI)
    random_sample_containment_generic(CI)
    print("   diagnostic: 2000/2000 random exact evaluations contained\n")

    print("6) Standalone generic rational-expression test")
    ratio_box = [Interval(-0.4, 0.4), Interval(-0.5, 0.5)]
    RR = autoia(
        generic_ratio_model, ratio_box,
        order=6, backend=backend,
        representation="polynomial",
    )
    print(RR)
    rng = np.random.default_rng(123)
    for _ in range(5000):
        x = rng.uniform(-0.4, 0.4)
        y = rng.uniform(-0.5, 0.5)
        z = plain_generic_ratio(x, y)
        if not (RR.lower[0] <= z <= RR.upper[0]):
            raise AssertionError("Generic division containment failure.")
    print("   diagnostic: 5000/5000 random exact evaluations contained")

    def denominator_crosses_zero(p):
        return 1.0 / p[0]

    try:
        autoia(
            denominator_crosses_zero, [Interval(-1.0, 1.0)],
            order=4, backend=backend, representation="polynomial",
        )
    except ZeroDivisionError as exc:
        print("   zero-crossing denominator correctly refused:")
        print("   ", exc)
    else:
        raise AssertionError("Division should refuse a denominator range containing zero.")
    print()

    print("All Version 0.10 generic-division/style diagnostics passed.")
