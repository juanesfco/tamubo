"""Ordinary imported engineering helper for Example 21."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

MACH = np.array([0.2, 0.5, 0.8, 1.1])
ALPHA = np.array([-4.0, 0.0, 4.0, 8.0])
CL = np.empty((len(MACH), len(ALPHA)))
for i, m in enumerate(MACH):
    for j, a in enumerate(ALPHA):
        CL[i, j] = 0.08*a*(1.0 - 0.12*m) + 0.02*m

CL_TABLE = RegularGridInterpolator((MACH, ALPHA), CL, method="linear")


def lift_coefficient(mach, alpha_deg):
    return CL_TABLE((mach, alpha_deg))
