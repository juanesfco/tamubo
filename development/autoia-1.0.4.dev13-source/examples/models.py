"""Small ordinary-Python engineering models used in AutoIA examples."""
from __future__ import annotations

import numpy as np


def spring_model(p):
    k1, k2, F = p
    A = np.array([[k1 + k2, -k2], [-k2, k2]])
    b = np.array([F, 0.0])
    return np.linalg.solve(A, b)


def coupled_model(p):
    p1, p2 = p
    A = np.array([[2.0 + p1, 0.2*p2], [0.2*p2, 1.5-p1]])
    b = np.array([1.0 + 0.1*p2, 0.5+p1])
    u = np.linalg.solve(A, b)
    return np.array([u[0], u[0]*u[1], u[1]**2], dtype=object)
