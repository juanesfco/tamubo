"""Ordinary engineering helper module used by Example 16.

This module intentionally owns the SciPy implicit solves.  The calling example
imports this module normally; AutoIA must prepare the reached pure-Python helper
locally and lazily without modifying this module or SciPy globally.
"""
import numpy as np
from scipy.optimize import root_scalar, fsolve


def colebrook_resid(f, Re, rel_roughness):
    """Colebrook-White residual for the Darcy friction factor."""
    if f <= 0:
        return np.nan
    term = (rel_roughness / 3.7) + (2.51 / (Re * np.sqrt(f)))
    return 1.0 / np.sqrt(f) + 2.0 * np.log10(term)


def calculate_friction_factor(Re, rel_roughness):
    """Implicit Darcy friction factor solved inside an imported helper module."""
    result = root_scalar(
        colebrook_resid,
        bracket=[0.0008, 0.1],
        args=(Re, rel_roughness),
    )
    return result.root


def coupled_residual(y, a, b):
    """Small nonlinear square residual system used to demonstrate fsolve."""
    return np.array([
        y[0] + 0.2 * y[1] - a,
        0.1 * y[0] + y[1] + 0.05 * y[0] * y[1] - b,
    ])


def solve_coupled_system(a, b):
    """Nonlinear system solved inside an imported helper module."""
    return fsolve(coupled_residual, [1.0, 1.5], args=(a, b))
