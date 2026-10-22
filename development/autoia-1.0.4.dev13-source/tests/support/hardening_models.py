"""Module-level deterministic models used by cross-platform hardening audits."""

from __future__ import annotations

import numpy as np


def prepared_process_model(p):
    x, y = p
    return np.exp(0.1*x) + x*y + 0.25*y*y


def near_singular_semantic_model(p):
    x = p[0]
    delta = 2.0 ** -30
    A = np.array([[1.0, 0.0], [0.0, delta + x]], dtype=object)
    b = np.array([1.0, 1.0], dtype=object)
    return np.linalg.solve(A, b)


def near_singular_generic_model(p):
    x = p[0]
    delta = 2.0 ** -30
    return np.array([1.0, 1.0/(delta + x)], dtype=object)


def one_ulp_branch_model(p):
    x = p[0]
    if x <= 1.0:
        return x - 1.0
    return 2.0*(x - 1.0)
