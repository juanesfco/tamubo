"""Thin-walled closed-end pressure-vessel stress calculation."""

import numpy as np


def pressure_vessel(p):
    """Return hoop, longitudinal, and von-Mises membrane stresses.

    Parameters are ``[pressure, radius, thickness]``.
    """
    pressure, radius, thickness = p
    hoop = pressure * radius / thickness
    longitudinal = pressure * radius / (2.0 * thickness)
    von_mises = np.sqrt(
        hoop**2 - hoop * longitudinal + longitudinal**2
    )
    return np.array([hoop, longitudinal, von_mises])
