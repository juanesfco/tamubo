from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import norm

from tamubo.utils import BackendName, get_array_module as _array_module

__all__ = ["expected_improvement", "log_expected_improvement", "log_h"]

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_LOG_SQRT_PI_OVER_2 = 0.5 * math.log(math.pi / 2.0)
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
# Beyond |z| = 100 the erfcx form cancels (its bracket is ~1/z^2, relative error
# ~eps z^2); the asymptotic series truncated after z^-8 is accurate to ~1e-16 there.
_Z_ASYMPTOTIC = 100.0

def _norm_cdf(z: Any, xp) -> Any:
    if xp is np:
        return norm.cdf(z)
    from cupyx.scipy.special import ndtr
    return ndtr(z)

def _norm_pdf(z: Any, xp) -> Any:
    if xp is np:
        return norm.pdf(z)
    inv_sqrt2pi = 1.0 / xp.sqrt(2.0 * xp.pi)
    return inv_sqrt2pi * xp.exp(-0.5 * z * z)

def _erfcx(z: Any, xp) -> Any:
    if xp is np:
        from scipy.special import erfcx
    else:
        from cupyx.scipy.special import erfcx
    return erfcx(z)

def _log1mexp(x: Any, xp) -> Any:
    """log(1 - exp(x)) for x < 0, accurate on both sides of -log(2)."""
    near_zero = x > -math.log(2.0)
    x_near = xp.where(near_zero, x, -1.0)
    x_far = xp.where(near_zero, -1.0, x)
    return xp.where(near_zero, xp.log(-xp.expm1(x_near)), xp.log1p(-xp.exp(x_far)))

def log_h(z: Any, *, backend: BackendName = "auto"):
    """
    log(h(z)) with h(z) = phi(z) + z * Phi(z), so that EI = sigma * h((y_min - mu) / sigma).

    Direct evaluation cancels catastrophically for z << 0 (both terms ~phi(z)/z) and
    underflows for z < ~-38. Following LogEI (Ament et al., "Unexpected Improvements
    to Expected Improvement for Bayesian Optimization", NeurIPS 2023), for z <= -1:

        h(z) = phi(z) * (1 - |z| * sqrt(pi/2) * erfcx(|z| / sqrt(2))),

    evaluated in log space, and for z < -100 the asymptotic series
    h(z) = phi(z) / z^2 * (1 - 3/z^2 + 15/z^4 - 105/z^6 + 945/z^8 - ...), where the
    erfcx form loses accuracy. h is increasing, so log_h is too.
    """
    xp = _array_module(backend)
    z = xp.asarray(z, dtype=xp.float64)

    mid = z > -1.0
    far = z < -_Z_ASYMPTOTIC

    # z > -1: no cancellation.
    z_mid = xp.where(mid, z, 0.0)
    out_mid = xp.log(_norm_pdf(z_mid, xp) + z_mid * _norm_cdf(z_mid, xp))

    # -100 <= z <= -1: log phi(z) + log1mexp(log(|z| erfcx(|z|/sqrt2)) + log sqrt(pi/2)).
    z_tail = xp.where(mid | far, -1.0, z)
    a = -z_tail
    out_tail = (
        -0.5 * z_tail * z_tail
        - _LOG_SQRT_2PI
        + _log1mexp(xp.log(_erfcx(a * _INV_SQRT2, xp) * a) + _LOG_SQRT_PI_OVER_2, xp)
    )

    # z < -100: asymptotic series.
    z_far = xp.where(far, z, -_Z_ASYMPTOTIC)
    r = 1.0 / (z_far * z_far)
    series = r * (-3.0 + r * (15.0 + r * (-105.0 + r * 945.0)))
    out_far = -0.5 * z_far * z_far - _LOG_SQRT_2PI - 2.0 * xp.log(-z_far) + xp.log1p(series)

    return xp.where(mid, out_mid, xp.where(far, out_far, out_tail))

def expected_improvement(
    mu: Any,
    sigma: Any,
    y_min: Any,
    *,
    backend: BackendName = "auto",
):
    """
    Expected Improvement (EI) for minimization.

    Parameters
    ----------
    mu : array-like
        Predictive mean values.
    sigma : array-like
        Predictive standard deviations (non-zero).
    y_min : float
        Best observed objective value (broadcastable to mu/sigma).
    backend : {"auto", "numpy", "cupy"}, default="auto"
        Backend used for array ops.
    """

    xp = _array_module(backend)
    mu = xp.asarray(mu, dtype=xp.float64)
    sigma = xp.asarray(sigma, dtype=xp.float64)

    z = (y_min - mu) / sigma
    ei = (y_min - mu) * _norm_cdf(z, xp) + sigma * _norm_pdf(z, xp)
    return ei

def log_expected_improvement(
    mu: Any,
    sigma: Any,
    y_min: Any,
    *,
    backend: BackendName = "auto",
):
    """
    Numerically stable log of the Expected Improvement for minimization:
    log EI = log(sigma) + log_h((y_min - mu) / sigma). Finite wherever sigma > 0,
    including far from the data where EI itself underflows to 0.

    Parameters
    ----------
    mu : array-like
        Predictive mean values.
    sigma : array-like
        Predictive standard deviations (positive).
    y_min : float
        Best observed objective value (broadcastable to mu/sigma).
    backend : {"auto", "numpy", "cupy"}, default="auto"
        Backend used for array ops.
    """
    xp = _array_module(backend)
    mu = xp.asarray(mu, dtype=xp.float64)
    sigma = xp.asarray(sigma, dtype=xp.float64)

    return xp.log(sigma) + log_h((y_min - mu) / sigma, backend=backend)
