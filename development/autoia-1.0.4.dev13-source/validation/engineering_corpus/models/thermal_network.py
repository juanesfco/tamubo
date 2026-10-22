"""Three-node steady thermal resistance network solved with NumPy."""

import numpy as np


def thermal_network(p):
    """Return three node temperatures and source heat rejection.

    Parameters are ``[g12, g23, g1a, g3a, q1, q2, q3, ambient]`` where the
    ``g`` quantities are positive thermal conductances.
    """
    g12, g23, g1a, g3a, q1, q2, q3, ambient = p

    A = np.array([
        [g12 + g1a, -g12, 0.0],
        [-g12, g12 + g23, -g23],
        [0.0, -g23, g23 + g3a],
    ])
    b = np.array([
        q1 + g1a * ambient,
        q2,
        q3 + g3a * ambient,
    ])
    temperature = np.linalg.solve(A, b)
    rejected = g1a * (temperature[0] - ambient) + g3a * (temperature[2] - ambient)
    return np.array([temperature[0], temperature[1], temperature[2], rejected])
