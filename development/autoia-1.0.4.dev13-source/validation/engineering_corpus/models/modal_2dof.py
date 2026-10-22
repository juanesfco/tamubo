"""Conventional two-degree-of-freedom modal calculation.

This is intentionally an expected unsupported Version-1 candidate.  It is kept
in the corpus because generalized eigenvalue analysis is common engineering
Python and is explicitly deferred by the current roadmap.
"""

import numpy as np


def modal_2dof(p):
    """Return the two natural frequencies from a standard 2-DOF mass-spring model."""
    m1, m2, k1, k2, k12 = p
    M = np.array([[m1, 0.0], [0.0, m2]])
    K = np.array([[k1 + k12, -k12], [-k12, k2 + k12]])
    dynamic = np.linalg.solve(M, K)
    eigenvalues = np.linalg.eigvals(dynamic)
    frequencies = np.sqrt(eigenvalues) / (2.0 * np.pi)
    return frequencies
