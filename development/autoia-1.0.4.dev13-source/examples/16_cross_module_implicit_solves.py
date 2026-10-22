"""16 — Cross-module implicit solves without rewriting the helper module.

This example is intentionally split across two ordinary Python modules.
`cross_module_implicit_model.py` owns the SciPy `root_scalar` and `fsolve`
calls.  This file imports that helper and asks AutoIA to certify wrappers that
call it normally.

The lesson is architectural: source-backed pure-Python helpers are prepared
locally and lazily only when AutoIA semantic values reach them.  The imported
module, NumPy, and SciPy remain unmodified.
"""
from autoia import Interval, autoia, __version__
import cross_module_implicit_model as engineering


def friction_factor_model(p):
    """External wrapper around an imported engineering helper."""
    return engineering.calculate_friction_factor(p[0], p[1])


def coupled_model(p):
    """External wrapper around an imported helper that owns fsolve."""
    return engineering.solve_coupled_system(p[0], p[1])


if __name__ == "__main__":
    print("AutoIA version:", __version__)

    friction = autoia(
        friction_factor_model,
        [Interval(90000.0, 110000.0), Interval(0.0008, 0.0012)],
    )
    print("\nImported-helper Colebrook root_scalar")
    print(friction)
    root_cert = [
        c for c in friction.certificates
        if c.get("operation") == "scipy.optimize.root_scalar"
    ][-1]
    print("unique in supplied bracket:", root_cert["unique_in_bracket"])
    print("machine rigorous:", friction.machine_rigorous)

    coupled = autoia(
        coupled_model,
        [Interval(0.9, 1.1), Interval(1.4, 1.6)],
    )
    print("\nImported-helper nonlinear-system fsolve")
    print(coupled)
    sys_cert = [
        c for c in coupled.certificates
        if c.get("operation") == "scipy.optimize.fsolve"
    ][-1]
    print("local contraction rho upper:", sys_cert["rho_upper"])
    print("machine rigorous:", coupled.machine_rigorous)
