"""First-order lumped thermal transient."""

import numpy as np


def thermal_rc(p):
    """Return body temperature and heat-flow magnitude after elapsed time.

    Parameters are ``[ambient_temperature, initial_temperature,
    thermal_resistance, thermal_capacitance, time]``.
    """
    ambient, initial, resistance, capacitance, time = p
    decay = np.exp(-time / (resistance * capacitance))
    temperature = ambient + (initial - ambient) * decay
    heat_flow = (temperature - ambient) / resistance
    return np.array([temperature, heat_flow])
