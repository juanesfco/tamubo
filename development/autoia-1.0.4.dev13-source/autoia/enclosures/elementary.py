"""Lift trusted scalar elementary functions to AutoIA enclosure representations."""

from __future__ import annotations

from ..trusted import (
    Interval,
    point_interval,
    abs_upper_interval,
    ElementaryDomainError,
    sqrt_interval,
    exp_interval,
    log_interval,
    log10_interval,
    sin_interval,
    cos_interval,
    absolute_interval,
    sinh_interval,
    cosh_interval,
    tanh_interval,
    arctan_interval,
    log1p_interval,
    erf_interval,
    standard_normal_pdf_interval,
    standard_normal_cdf_interval,
)
from ..polynomial import zero_alpha
from ..elementary import build_taylor_data
from .runtime import safe_interval_representative, interval_intersection


_TRUSTED_INTERVAL_FUNCTIONS = {
    "sqrt": sqrt_interval,
    "exp": exp_interval,
    "log": log_interval,
    "log10": log10_interval,
    "sin": sin_interval,
    "cos": cos_interval,
    "abs": absolute_interval,
    "absolute": absolute_interval,
    "sinh": sinh_interval,
    "cosh": cosh_interval,
    "tanh": tanh_interval,
    "arctan": arctan_interval,
    "log1p": log1p_interval,
    "erf": erf_interval,
    "normal_pdf": standard_normal_pdf_interval,
    "normal_cdf": standard_normal_cdf_interval,
}


def trusted_elementary_range(name: str, domain: Interval, backend) -> Interval:
    try:
        fn = _TRUSTED_INTERVAL_FUNCTIONS[str(name)]
    except KeyError as exc:
        raise NotImplementedError(f"Unsupported elementary function {name!r}.") from exc
    return fn(domain, backend)


def _constant_interval_coefficient_rep(prototype, coefficient: Interval):
    """Represent a rigorous scalar coefficient interval as point+remainder."""
    c = safe_interval_representative(coefficient)
    error = prototype.backend.sub(coefficient, point_interval(c))
    return prototype.__class__(
        {zero_alpha(prototype.m): c},
        error,
        prototype.h,
        prototype.order,
        prototype.backend,
        prototype.range_method,
    )


def _interval_lift_to_polynomial(prototype, image: Interval):
    """Sound dependency-destroying fallback ``image = c + (image-c)``."""
    c = safe_interval_representative(image)
    rem = prototype.backend.sub(image, point_interval(c))
    return prototype.__class__(
        {zero_alpha(prototype.m): c},
        rem,
        prototype.h,
        prototype.order,
        prototype.backend,
        prototype.range_method,
    )


def _interval_image_as_remainder(prototype, image: Interval):
    """Dependency-destroying but endpoint-preserving image lift ``0 + image``."""
    return prototype.__class__(
        {zero_alpha(prototype.m): 0.0},
        image,
        prototype.h,
        prototype.order,
        prototype.backend,
        prototype.range_method,
    )


def apply_interval_elementary(rep, name: str, *, domain=None):
    X = rep.interval if domain is None else interval_intersection(rep.interval, domain)
    image = trusted_elementary_range(name, X, rep.backend)
    return rep.__class__(
        image,
        rep.h,
        rep.order,
        rep.backend,
        rep.range_method,
    )


def apply_polynomial_elementary(rep, name: str, *, domain=None):
    """Validated Taylor composition for ``f(rep)``.

    The exact scalar value represented by ``rep`` lies in ``X``.  Choose one
    stored binary64 center ``c in X`` and write ``delta = x-c``.  Rigorous
    intervals contain every Taylor coefficient ``f^(n)(c)/n!``.  Each interval
    coefficient is itself represented as a point coefficient plus a rigorous
    scalar remainder, so ordinary polynomial enclosure arithmetic propagates
    coefficient uncertainty, the input remainder, and polynomial truncation.
    A separate Lagrange remainder is added at the end.
    """
    name = str(name)
    X_structural, _ = rep.range(rep.range_method)
    X = X_structural if domain is None else interval_intersection(X_structural, domain)

    # ``domain`` may be an independently certified range over the actual
    # physical input box.  Intersecting it with the structural range is a proof
    # strengthening operation, never clipping: both operands independently
    # contain every feasible exact-real argument.

    # Domain validation and a sound nonsmooth-boundary fallback for sqrt.
    if name == "sqrt":
        if X.lo < 0.0:
            raise ElementaryDomainError(
                f"sqrt is not real-valued over the full certified range [{X.lo}, {X.hi}]."
            )
        if X.lo == 0.0:
            return _interval_lift_to_polynomial(
                rep, trusted_elementary_range(name, X, rep.backend)
            )
    elif name in {"log", "log10"} and X.lo <= 0.0:
        raise ElementaryDomainError(
            f"{name} is not real-valued over the full certified range [{X.lo}, {X.hi}]."
        )
    elif name == "log1p" and X.lo <= -1.0:
        raise ElementaryDomainError(
            f"log1p is not real-valued over the full certified range [{X.lo}, {X.hi}]."
        )
    elif name in {"abs", "absolute"}:
        if X.lo >= 0.0:
            return rep.copy()
        if X.hi <= 0.0:
            return rep.negated()
        return _interval_image_as_remainder(
            rep, trusted_elementary_range(name, X, rep.backend)
        )
    elif name in {"tanh", "arctan"}:
        # Both are globally monotone and bounded.  On broad boxes, a generic
        # Taylor remainder can be much wider than the exact monotone interval
        # image, so Version 1 deliberately chooses the latter rather than
        # returning a structurally richer but practically useless certificate.
        return _interval_image_as_remainder(
            rep, trusted_elementary_range(name, X, rep.backend)
        )
    elif name not in _TRUSTED_INTERVAL_FUNCTIONS:
        raise NotImplementedError(f"Unsupported elementary function {name!r}.")

    c = safe_interval_representative(X)
    if name in {"sqrt", "log", "log10"} and c <= 0.0:
        # sqrt can only reach here through X.lo==0, handled above; retain a
        # defensive sound fallback rather than constructing singular derivatives.
        if name == "sqrt":
            return _interval_lift_to_polynomial(
                rep, trusted_elementary_range(name, X, rep.backend)
            )
        raise ElementaryDomainError(f"{name} Taylor center is not positive.")

    D = rep.backend.sub(X, point_interval(c))
    d = abs_upper_interval(D)
    data = build_taylor_data(name, c, rep.order, X, d, rep.backend)

    delta = rep - c
    power = rep.__class__.constant(
        1.0, rep.h, rep.order, rep.backend, rep.range_method
    )
    out = rep.__class__.constant(
        0.0, rep.h, rep.order, rep.backend, rep.range_method
    )

    for n, coefficient in enumerate(data.coefficients):
        if n > 0:
            power = power * delta
        coeff_rep = _constant_interval_coefficient_rep(rep, coefficient)
        out = out + coeff_rep * power

    R = Interval(-data.remainder_radius, data.remainder_radius)
    final_remainder = rep.backend.add(out.remainder, R)
    with_remainder = getattr(out, "with_remainder", None)
    if callable(with_remainder):
        return with_remainder(final_remainder)
    return rep.__class__(
        dict(out.coeff),
        final_remainder,
        rep.h,
        rep.order,
        rep.backend,
        rep.range_method,
    )


__all__ = [
    "trusted_elementary_range",
    "apply_interval_elementary",
    "apply_polynomial_elementary",
]
