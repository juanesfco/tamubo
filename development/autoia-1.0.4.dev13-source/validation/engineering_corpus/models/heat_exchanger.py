"""Counterflow heat-exchanger log-mean-temperature-difference calculation."""

import numpy as np


def heat_exchanger(p):
    """Return terminal temperature differences, LMTD, and heat-transfer rate.

    Parameters are ``[hot_in, hot_out, cold_in, cold_out, overall_u, area]``.
    The validation box is chosen so both terminal temperature differences remain
    positive and separated from one another.
    """
    hot_in, hot_out, cold_in, cold_out, overall_u, area = p
    delta_t_1 = hot_in - cold_out
    delta_t_2 = hot_out - cold_in
    lmtd = (delta_t_1 - delta_t_2) / np.log(delta_t_1 / delta_t_2)
    heat_rate = overall_u * area * lmtd
    return np.array([delta_t_1, delta_t_2, lmtd, heat_rate])
