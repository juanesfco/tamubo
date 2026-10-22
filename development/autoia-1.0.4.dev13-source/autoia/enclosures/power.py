"""Validated constant-real powers for AutoIA enclosure representations.

Power support deliberately reuses the already-validated algebraic and
Step-3 elementary-function machinery instead of introducing a second special
Taylor implementation.

Supported contract
------------------
``validated_base ** p`` where ``p`` is one finite constant Python/NumPy
integer or floating scalar.

* Integer ``p``: all real bases are allowed when mathematically defined.
  Negative integers require the certified base range to exclude zero.
* Non-integer ``p``: the real branch is used and the certified base range must
  be strictly positive for the polynomial representation.  The interval
  representation may include zero only when ``p > 0``.
* Validated/uncertain exponents and complex branches are not supported.
"""

from __future__ import annotations

from fractions import Fraction
import math
import numpy as np

from ..trusted import ElementaryDomainError, real_power_interval
from .runtime import interval_intersection


def normalize_constant_exponent(power):
    """Return ``("integer", n)`` or ``("real", p)`` for supported exponents."""
    if isinstance(power, (bool, int, np.integer)):
        return "integer", int(power)
    if isinstance(power, (float, np.floating)):
        p = float(power)
        if not np.isfinite(p):
            raise NotImplementedError("AutoIA power requires a finite constant exponent.")
        if p.is_integer():
            return "integer", int(p)
        return "real", p
    raise NotImplementedError(
        "AutoIA currently supports only constant Python/NumPy integer or "
        "floating exponents; uncertain, complex, Decimal, and Fraction "
        "exponents are not yet supported."
    )


def _exact_nearest_integer_split(power: float) -> tuple[int, float]:
    """Return exact ``power = n + r`` with ``|r| <= 1/2`` when possible.

    ``power`` is interpreted as the exact dyadic rational represented by its
    binary64 bits.  We choose the nearest integer, breaking half-way ties
    toward zero, and verify with :class:`fractions.Fraction` that the residual
    ``r`` is represented exactly as binary64.  If an unusual platform value
    ever defeats that exact residual representation, the conservative fallback
    ``n=0, r=power`` preserves semantics.
    """
    p = float(power)
    f = Fraction.from_float(p)
    lo = math.floor(p)
    hi = math.ceil(p)
    candidates = sorted({lo, hi}, key=lambda n: (abs(f - n), abs(n)))
    n = int(candidates[0])
    r_exact = f - n
    r = float(r_exact)
    if Fraction.from_float(r) != r_exact:
        return 0, p
    return n, r


def integer_power_value(value, n: int):
    """Exponentiation by squaring for a ``ValidatedValue``-like object."""
    n = int(n)
    if n == 0:
        return value._constant_like(1.0)
    if n < 0:
        positive = integer_power_value(value, -n)
        return value._constant_like(1.0) / positive

    result = value._constant_like(1.0)
    base = value
    k = n
    while k:
        if k & 1:
            result = result * base
        k >>= 1
        if k:
            base = base * base
    return result


def integer_power_rep(rep, n: int):
    """Exponentiation by squaring directly on a representation object."""
    n = int(n)
    if n == 0:
        return rep.constant_like(1.0)
    if n < 0:
        positive = integer_power_rep(rep, -n)
        return rep.constant_like(1.0).binary_same(positive, "div")

    result = rep.constant_like(1.0)
    base = rep
    k = n
    while k:
        if k & 1:
            result = result.binary_same(base, "mul")
        k >>= 1
        if k:
            base = base.binary_same(base, "mul")
    return result


def apply_interval_real_power(rep, power: float, *, domain=None):
    X = rep.interval if domain is None else interval_intersection(rep.interval, domain)
    image = real_power_interval(X, float(power), rep.backend)
    return rep.__class__(
        image,
        rep.h,
        rep.order,
        rep.backend,
        rep.range_method,
    )


def apply_polynomial_real_power(rep, power: float, *, domain=None):
    """Validated real power built from existing algebra, log, exp, and sqrt.

    For positive ``x`` and exact constant ``p``, choose an exact decomposition

        p = n + r,  |r| <= 1/2,

    and evaluate

        x**p = x**n * x**r.

    The integer part uses ordinary validated algebra.  The fractional part uses
    the already-validated Step-3 elementary functions: ``sqrt`` for ``r=1/2``,
    reciprocal square root for ``r=-1/2``, and otherwise

        x**r = exp(r * log(x)).

    This avoids duplicating elementary-function proof machinery and is usually
    much sharper than a direct high-order Lagrange bound for arbitrary powers.
    """
    p = float(power)
    X_structural, _ = rep.range(rep.range_method)
    X = X_structural if domain is None else interval_intersection(X_structural, domain)

    if X.lo < 0.0 or (X.lo == 0.0 and p < 0.0):
        relation = "nonnegative" if p > 0.0 else "strictly positive"
        raise ElementaryDomainError(
            f"non-integer real power {p!r} requires a {relation} certified "
            f"base range, but AutoIA obtained [{X.lo}, {X.hi}]."
        )

    # Positive non-integer powers are mathematically defined at zero, but the
    # log-based polynomial construction is singular there.  Use a sound
    # dependency-destroying interval lift at that boundary rather than clipping
    # or invoking log(0).
    if X.lo == 0.0 and p > 0.0:
        from .elementary import _interval_lift_to_polynomial
        return _interval_lift_to_polynomial(
            rep, real_power_interval(X, p, rep.backend)
        )

    if X.lo <= 0.0:
        raise ElementaryDomainError(
            f"non-integer real power {p!r} requires a strictly positive "
            f"certified base range, but AutoIA obtained [{X.lo}, {X.hi}]."
        )

    n, r = _exact_nearest_integer_split(p)

    # If positivity was established only after intersecting an independent
    # physical-box range, the older reciprocal proof may still see zero in the
    # structural q-superset.  In that case use the already validated direct
    # identity exp(p*log(x)) rather than discarding a valid domain proof.
    if n < 0 and X_structural.lo <= 0.0 <= X_structural.hi:
        from .elementary import apply_polynomial_elementary
        log_rep = apply_polynomial_elementary(rep, "log", domain=X)
        return apply_polynomial_elementary(log_rep * p, "exp")

    integer_part = integer_power_rep(rep, n)
    if r == 0.0:
        return integer_part

    from .elementary import apply_polynomial_elementary

    if r == 0.5:
        fractional = apply_polynomial_elementary(rep, "sqrt", domain=X)
    elif r == -0.5:
        sqrt_rep = apply_polynomial_elementary(rep, "sqrt", domain=X)
        fractional = sqrt_rep.reciprocal()
    else:
        log_rep = apply_polynomial_elementary(rep, "log", domain=X)
        scaled = log_rep * r
        fractional = apply_polynomial_elementary(scaled, "exp")

    return integer_part * fractional


def power_representation(rep, power, *, domain=None):
    kind, p = normalize_constant_exponent(power)
    if getattr(rep, "representation_name", None) == "interval":
        return apply_interval_real_power(rep, float(p), domain=domain)
    if kind == "integer":
        return integer_power_rep(rep, p)
    if getattr(rep, "representation_name", None) in {"polynomial", "generator-polynomial"}:
        return apply_polynomial_real_power(rep, p, domain=domain)
    fn = getattr(rep, "power_constant", None)
    if callable(fn):
        if domain is None:
            return fn(p)
        return fn(p, domain=domain)
    raise NotImplementedError(
        f"Representation {getattr(rep, 'representation_name', type(rep).__name__)!r} "
        "does not support non-integer real powers."
    )


__all__ = [
    "normalize_constant_exponent",
    "integer_power_value",
    "integer_power_rep",
    "apply_interval_real_power",
    "apply_polynomial_real_power",
    "power_representation",
]
