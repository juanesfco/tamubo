"""15 — Common implicit engineering solves: root_scalar and fsolve.

SciPy is optional.  When present, AutoIA recognizes the mathematical meaning of
these solve calls.  SciPy's floating-point iteration supplies only a proposal;
AutoIA independently certifies the implicit solution.
"""
import numpy as np

from autoia import Interval, autoia, __version__

try:
    from scipy.optimize import root_scalar, fsolve
except ImportError:
    print("Example 15 skipped: SciPy is not installed (SciPy remains optional for AutoIA).")
    raise SystemExit(0)


# Ordinary engineering code: Colebrook-White implicit friction factor.
def colebrook_resid(f, Re, rel_roughness):
    if f <= 0:
        return np.nan
    term = (rel_roughness / 3.7) + (2.51 / (Re * np.sqrt(f)))
    return 1.0 / np.sqrt(f) + 2.0 * np.log10(term)


def friction_factor_model(p):
    Re, rel_roughness = p
    result = root_scalar(
        colebrook_resid,
        bracket=[0.0008, 0.1],
        args=(Re, rel_roughness),
    )
    return result.root


# A small square nonlinear system for fsolve.
def coupled_residual(y, a, b):
    return np.array([
        y[0] + 0.2 * y[1] - a,
        0.1 * y[0] + y[1] + 0.05 * y[0] * y[1] - b,
    ])


def coupled_model(p):
    return fsolve(coupled_residual, [1.0, 1.5], args=(p[0], p[1]))


if __name__ == "__main__":
    print("AutoIA version:", __version__)

    friction = autoia(
        friction_factor_model,
        [Interval(90000.0, 110000.0), Interval(0.0008, 0.0012)],
    )
    print("\nColebrook root_scalar")
    print(friction)
    root_cert = [
        c for c in friction.certificates
        if c.get("operation") == "scipy.optimize.root_scalar"
    ][-1]
    print("unique in supplied bracket:", root_cert["unique_in_bracket"])
    print("implicit correction radius:", root_cert["correction_radius"])

    coupled = autoia(
        coupled_model,
        [Interval(0.9, 1.1), Interval(1.4, 1.6)],
    )
    print("\nNonlinear-system fsolve")
    print(coupled)
    sys_cert = [
        c for c in coupled.certificates
        if c.get("operation") == "scipy.optimize.fsolve"
    ][-1]
    print("local contraction rho upper:", sys_cert["rho_upper"])
    print("implicit correction radius:", sys_cert["correction_radius"])
