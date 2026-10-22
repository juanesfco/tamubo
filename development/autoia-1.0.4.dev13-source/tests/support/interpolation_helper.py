import numpy as np
from scipy.interpolate import RegularGridInterpolator

X = np.array([0.0, 0.5, 1.0])
Y = np.array([-1.0, 0.0, 1.0])
VALUES = np.empty((len(X), len(Y)))
for i, x in enumerate(X):
    for j, y in enumerate(Y):
        VALUES[i, j] = 2.0 + 0.5*x - y + 3.0*x*y

INTERP = RegularGridInterpolator((X, Y), VALUES, method="linear")


def evaluate(x, y):
    return INTERP((x, y))
