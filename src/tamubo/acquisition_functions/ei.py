from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
from scipy.stats import norm

from tamubo.utils import BackendName, resolve_backend

__all__ = ["expected_improvement"]

def _array_module(backend: BackendName = "auto"):
    """Return the resolved array module (`numpy` or `cupynumeric`)."""
    backend_info = resolve_backend(backend)
    if backend_info.selected == "numpy":
        return np
    # Import cupynumeric only when it is the selected backend.
    return import_module("cupynumeric")

def _erf_approx(x: Any, xp) -> Any:
    """Abramowitz & Stegun 7.1.26 approximation of erf(x)."""
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign = xp.sign(x)
    ax = xp.abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * xp.exp(-ax * ax)
    return sign * y

def _norm_cdf(z: Any, xp) -> Any:
    if xp is np:
        return norm.cdf(z)
    sqrt2 = xp.sqrt(2.0)
    return 0.5 * (1.0 + _erf_approx(z / sqrt2, xp))

def _norm_pdf(z: Any, xp) -> Any:
    if xp is np:
        return norm.pdf(z)
    inv_sqrt2pi = 1.0 / xp.sqrt(2.0 * xp.pi)
    return inv_sqrt2pi * xp.exp(-0.5 * z * z)

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
        Predictive standard deviations.
    y_min : float
        Best observed objective value (broadcastable to mu/sigma).
    backend : {"auto", "numpy", "cupynumeric"}, default="auto"
        Backend used for array ops.
    """

    xp = _array_module(backend)
    mu = xp.asarray(mu, dtype=xp.float64)
    sigma = xp.asarray(sigma, dtype=xp.float64)

    safe_sigma = xp.where(sigma == 0.0, 1.0, sigma)
    z = (y_min - mu) / safe_sigma
    ei = (y_min - mu) * _norm_cdf(z, xp) + safe_sigma * _norm_pdf(z, xp)
    return xp.where(sigma == 0.0, 0.0, ei)
