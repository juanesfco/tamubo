from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .ei import expected_improvement

@dataclass
class EIBounds:
    lower: float
    upper: float

def ei_bounds_from_mu_sigma_intervals(
    mu_lo: float, mu_hi: float, var_lo: float, var_hi: float, f_best: float
) -> EIBounds:
    """
    Conservative EI bounds by evaluating EI at the 4 corners of (mu, sigma).
    """
    sig_lo = float(np.sqrt(max(var_lo, 1e-18)))
    sig_hi = float(np.sqrt(max(var_hi, 1e-18)))
    mus = [mu_lo, mu_hi]
    sigs = [sig_lo, sig_hi]
    vals = []
    for m in mus:
        for s in sigs:
            v = expected_improvement(np.array([[m]]), np.array([[s]]), f_best)
            vals.append(float(v))
    return EIBounds(lower=min(vals), upper=max(vals))
