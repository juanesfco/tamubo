"""21 — Cross-module aerodynamic-style lookup table.

The imported helper owns a preconstructed SciPy RegularGridInterpolator. AutoIA
prepares the reached helper locally and certifies the uncertain query without
modifying the source module or SciPy.
"""
from autoia import Interval, autoia
import cross_module_interpolation_model as aero


def model(p):
    return aero.lift_coefficient(p[0], p[1])


if __name__ == "__main__":
    result = autoia(
        model,
        [Interval(0.35, 0.9), Interval(-1.0, 6.0)],
    )
    print(result)
    print("cell/path branches:", len(getattr(result, "branch_results", ())))
    print("machine rigorous:", result.machine_rigorous)
