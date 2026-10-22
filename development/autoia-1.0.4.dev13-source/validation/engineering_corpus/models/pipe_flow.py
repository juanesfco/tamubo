"""Darcy-Weisbach pipe pressure loss with a conventional regime branch."""

import numpy as np


def pipe_pressure_drop(p):
    """Return Reynolds number, Darcy friction factor, and pressure drop.

    Parameters are ``[density, velocity, viscosity, diameter, length]``.
    The simple Blasius turbulent correlation is intentionally retained as it is
    commonly used for smooth-pipe estimates over its applicable range.
    """
    density, velocity, viscosity, diameter, length = p
    reynolds = density * velocity * diameter / viscosity

    if reynolds < 2300.0:
        friction = 64.0 / reynolds
    else:
        friction = 0.3164 / reynolds**0.25

    pressure_drop = friction * (length / diameter) * 0.5 * density * velocity**2
    return np.array([reynolds, friction, pressure_drop])
