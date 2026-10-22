"""Rectangular cantilever beam calculations in ordinary NumPy/Python.

No AutoIA imports are used here.  The function is intentionally written like a
small engineering analysis routine that can also be called with ordinary float
arrays.
"""

import numpy as np


def cantilever_beam(p):
    """Return root bending stress, tip deflection, and section frequency proxy.

    Parameters are ``[force, length, width, height, youngs_modulus, density]``.
    """
    force, length, width, height, youngs_modulus, density = p

    area = width * height
    second_moment = width * height**3 / 12.0
    root_moment = force * length
    root_stress = root_moment * (0.5 * height) / second_moment
    tip_deflection = force * length**3 / (3.0 * youngs_modulus * second_moment)

    # First Euler-Bernoulli cantilever bending frequency.
    beta1 = 1.875104068711961
    omega1 = beta1**2 * np.sqrt(
        youngs_modulus * second_moment / (density * area * length**4)
    )
    frequency_hz = omega1 / (2.0 * np.pi)

    return np.array([root_stress, tip_deflection, frequency_hz])
