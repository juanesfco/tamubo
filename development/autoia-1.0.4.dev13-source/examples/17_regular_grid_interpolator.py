"""17 — Fixed-data rectilinear table interpolation.

AutoIA recognizes SciPy RegularGridInterpolator with method='linear'.  Fixed
numeric grid/table data remain ordinary SciPy data.  Uncertain query coordinates
are certified by path-aware cell selection plus the exact multilinear formula
inside each feasible cell.
"""
import numpy as np
from autoia import Interval, autoia

try:
    from scipy.interpolate import RegularGridInterpolator
except ImportError:
    print("Example 17 skipped: SciPy is not installed.")
    raise SystemExit(0)

X = np.array([0.0, 0.5, 1.0, 1.5])
Y = np.array([-1.0, 0.0, 1.0])
VALUES = np.empty((len(X), len(Y), 2))
for i, x in enumerate(X):
    for j, y in enumerate(Y):
        VALUES[i, j, 0] = 2.0 + 0.5*x - y + 3.0*x*y
        VALUES[i, j, 1] = 1.0 + x*x + 0.25*y  # only nodal values matter to the interpolant

TABLE = RegularGridInterpolator((X, Y), VALUES, method="linear")


def model(p):
    return TABLE((p[0], p[1]))


if __name__ == "__main__":
    # This box crosses x=0.5 and y=0, so more than one interpolation cell is feasible.
    box = [Interval(0.2, 0.9), Interval(-0.4, 0.6)]
    result = autoia(model, box)
    print(result)
    print("output shape:", result.output_shape)
    print("path/cell branches:", len(getattr(result, "branch_results", ())))
    print("machine rigorous:", result.machine_rigorous)

    # Sampling is diagnostic only; it is not part of the proof.
    samples = []
    for x in np.linspace(0.2, 0.9, 41):
        for y in np.linspace(-0.4, 0.6, 41):
            samples.append(np.asarray(TABLE((x, y)), dtype=float))
    samples = np.asarray(samples)
    print("sample lower (diagnostic only):", samples.min(axis=0))
    print("sample upper (diagnostic only):", samples.max(axis=0))
