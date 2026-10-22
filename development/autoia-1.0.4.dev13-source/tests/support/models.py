"""Test-only engineering models used by historical AutoIA regression tests.

These are ordinary model functions, deliberately kept outside the installed
``autoia`` package.  They exercise semantic and generic execution paths.
"""
from __future__ import annotations

import numpy as np


def spring_model(p):
    k1, k2, F = p
    A = np.array([
        [k1 + k2, -k2],
        [-k2,          k2],
    ])
    b = np.array([F, 0.0])
    return np.linalg.solve(A, b)


def spring_with_postprocessing(p):
    k1, k2, F = p
    A = np.array([
        [k1 + k2, -k2],
        [-k2,          k2],
    ])
    b = np.array([F, 0.0])
    u = np.linalg.solve(A, b)
    return np.array([u[0], F*u[0], u[0]**2], dtype=object)


def generic_coupled_model(p):
    p1, p2 = p
    A = np.array([
        [2.0 + p1, 0.2 * p2],
        [0.2 * p2, 1.5 - p1],
    ])
    b = np.array([1.0 + 0.1 * p2, 0.5 + p1])
    u = np.linalg.solve(A, b)
    return np.array([u[0], u[0]*u[1], u[1]**2], dtype=object)


def manual_solve_2x2(A, b):
    """Hand-coded exact 2x2 algebraic solve with no semantic solve call."""
    a = A[0, 0]
    bb = A[0, 1]
    c = A[1, 0]
    d = A[1, 1]
    det = a*d - bb*c
    x0 = (b[0]*d - bb*b[1]) / det
    x1 = (a*b[1] - b[0]*c) / det
    return np.array([x0, x1], dtype=object)


def spring_model_manual(p):
    k1, k2, F = p
    A = np.array([
        [k1 + k2, -k2],
        [-k2,          k2],
    ])
    b = np.array([F, 0.0])
    return manual_solve_2x2(A, b)


def generic_coupled_model_manual(p):
    p1, p2 = p
    A = np.array([
        [2.0 + p1, 0.2 * p2],
        [0.2 * p2, 1.5 - p1],
    ])
    b = np.array([1.0 + 0.1 * p2, 0.5 + p1])
    u = manual_solve_2x2(A, b)
    return np.array([u[0], u[0]*u[1], u[1]**2], dtype=object)


def generic_ratio_model(p):
    x, y = p
    numerator = 1.0 + 0.3*x - 0.2*y + 0.1*x*y
    denominator = 2.0 + 0.15*x + 0.1*y - 0.05*x*y + 0.02*x*x
    return numerator / denominator


def hidden_monotone_model(p):
    total = 0.0
    for x in p:
        total = total + x + x*x + (x*x*x)/3.0
    return total
