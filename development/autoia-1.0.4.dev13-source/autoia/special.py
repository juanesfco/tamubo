"""Validated special/statistical scalar functions for AutoIA.

The standard-normal functions are first-class validated unary operations.  The
parameterized ``normal_pdf`` and ``normal_cdf`` helpers standardize by constant
binary64 ``mean`` and strictly-positive ``std`` values, then reuse those unary
primitives.  This keeps the mathematical constants ``pi`` and ``sqrt(2)``
inside the trusted proof machinery instead of treating rounded decimal
constants as exact distribution constants.
"""

from __future__ import annotations

import math
import numpy as np

from .enclosures import ValidatedValue


def _contains_validated(obj) -> bool:
    arr = np.asarray(obj, dtype=object)
    return any(isinstance(v, ValidatedValue) for v in arr.flat)


def _apply_validated_unary(x, name: str, numeric_scalar):
    if isinstance(x, ValidatedValue):
        return x.unary(name)
    arr = np.asarray(x, dtype=object)
    if _contains_validated(arr):
        out = np.empty(arr.shape, dtype=object)
        for idx in np.ndindex(arr.shape):
            v = arr[idx]
            out[idx] = v.unary(name) if isinstance(v, ValidatedValue) else numeric_scalar(float(v))
        return out.item() if out.ndim == 0 else out
    if np.ndim(x) == 0:
        return numeric_scalar(float(x))
    fn = np.vectorize(lambda v: numeric_scalar(float(v)), otypes=[float])
    return fn(np.asarray(x))


def erf(x):
    """Real error function, validated when ``x`` contains AutoIA values."""
    return _apply_validated_unary(x, "erf", math.erf)


def standard_normal_pdf(x):
    """Standard-normal probability density ``phi(x)``."""
    def scalar(v: float) -> float:
        return math.exp(-0.5 * v * v) / math.sqrt(2.0 * math.pi)
    return _apply_validated_unary(x, "normal_pdf", scalar)


def standard_normal_cdf(x):
    """Standard-normal cumulative distribution ``Phi(x)``."""
    def scalar(v: float) -> float:
        return 0.5 * (1.0 + math.erf(v / math.sqrt(2.0)))
    return _apply_validated_unary(x, "normal_cdf", scalar)


def _coerce_distribution_parameters(mean, std) -> tuple[float, float]:
    if isinstance(mean, ValidatedValue) or isinstance(std, ValidatedValue):
        raise NotImplementedError(
            "AutoIA normal_pdf/normal_cdf currently require constant mean and "
            "standard deviation; uncertain distribution parameters are not yet supported."
        )
    mu = float(mean)
    sigma = float(std)
    if not np.isfinite(mu):
        raise ValueError("Normal mean must be a finite real constant.")
    if not (np.isfinite(sigma) and sigma > 0.0):
        raise ValueError("Normal standard deviation must be a finite positive real constant.")
    return mu, sigma


def normal_pdf(x, mean=0.0, std=1.0):
    """Normal density with constant ``mean`` and positive constant ``std``."""
    mu, sigma = _coerce_distribution_parameters(mean, std)
    z = (x - mu) / sigma
    return standard_normal_pdf(z) / sigma


def normal_cdf(x, mean=0.0, std=1.0):
    """Normal CDF with constant ``mean`` and positive constant ``std``."""
    mu, sigma = _coerce_distribution_parameters(mean, std)
    z = (x - mu) / sigma
    return standard_normal_cdf(z)


# Short aliases common in engineering/statistics code.
norm_pdf = normal_pdf
norm_cdf = normal_cdf


__all__ = [
    "erf",
    "standard_normal_pdf",
    "standard_normal_cdf",
    "normal_pdf",
    "normal_cdf",
    "norm_pdf",
    "norm_cdf",
]
