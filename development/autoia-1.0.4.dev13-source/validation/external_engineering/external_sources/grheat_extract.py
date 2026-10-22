"""External planar Green-function method from the MIT-licensed ``grheat`` project.

Project: grheat, Scott Prahl
Upstream file: grheat/plane_source.py
Release documented upstream: 0.5.1 (2026)
License: MIT

The ``_instantaneous`` computational method is retained from upstream.  The
small constructor below is validation scaffolding only; it supplies the same
attributes that the original Plane/HeatSource classes supply.
"""

import numpy as np


class PlaneInstantaneousExternal:
    def __init__(self, zp, tp=0.0, diffusivity=1.4e-7, capacity=4.18e6, boundary="infinite"):
        self.zp = zp
        self.tp = tp
        self.diffusivity = diffusivity
        self.capacity = capacity
        self.boundary = boundary

    def _instantaneous(self, z: float | np.ndarray, t: float, tp: float) -> float | np.ndarray:
        z2 = (z - self.zp) ** 2
        if t <= tp:
            return 0 * z2
        factor = self.capacity * 2 * np.sqrt(np.pi * self.diffusivity * (t - tp))
        T = 1 / factor * np.exp(-z2 / (4 * self.diffusivity * (t - tp)))
        if self.boundary != "infinite":
            z2 = (z + self.zp) ** 2
            T1 = 1 / factor * np.exp(-z2 / (4 * self.diffusivity * (t - tp)))
            if self.boundary == "adiabatic":
                T += T1
            if self.boundary == "zero":
                T -= T1
        return T
