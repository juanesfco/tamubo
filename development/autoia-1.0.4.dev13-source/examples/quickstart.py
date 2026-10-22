"""Minimal AutoIA Version-1-style quickstart.

Run from the project root with:
    python examples/quickstart.py
"""

import numpy as np
from autoia import Interval, autoia


def spring_model(p):
    k1, k2, F = p
    A = np.array([
        [k1 + k2, -k2],
        [-k2, k2],
    ])
    b = np.array([F, 0.0])
    return np.linalg.solve(A, b)


if __name__ == "__main__":
    # No representation or polynomial order is required for ordinary use.
    # AutoIA begins from its release baseline and only retries a stronger
    # configuration when its own structural diagnostics justify the cost.
    result = autoia(
        spring_model,
        [
            Interval(90.0, 110.0),
            Interval(900.0, 1100.0),
            Interval(9.5, 10.5),
        ],
    )
    print(result)
    print("machine_rigorous =", result.machine_rigorous)
