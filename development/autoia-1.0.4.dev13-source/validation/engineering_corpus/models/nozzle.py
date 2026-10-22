"""Ideal-gas isentropic nozzle relations with fixed gas properties."""

import numpy as np

GAMMA = 1.4
R_GAS = 287.05


def nozzle(p):
    """Return static temperature, exit speed, density, and mass flux.

    Parameters are ``[stagnation_temperature, stagnation_pressure,
    pressure_ratio]``, where ``pressure_ratio = p_exit / p_stagnation``.
    """
    stagnation_temperature, stagnation_pressure, pressure_ratio = p
    exponent = (GAMMA - 1.0) / GAMMA
    static_temperature = stagnation_temperature * pressure_ratio**exponent
    exit_speed = np.sqrt(
        2.0 * GAMMA / (GAMMA - 1.0)
        * R_GAS
        * stagnation_temperature
        * (1.0 - pressure_ratio**exponent)
    )
    exit_pressure = stagnation_pressure * pressure_ratio
    density = exit_pressure / (R_GAS * static_temperature)
    mass_flux = density * exit_speed
    return np.array([static_temperature, exit_speed, density, mass_flux])
