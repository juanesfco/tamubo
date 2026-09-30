from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm

from tamubo.utils import BackendName, get_array_module as _array_module

__all__ = ["expected_improvement"]

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
