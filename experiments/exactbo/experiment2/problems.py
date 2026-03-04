"""Problem registry for experiment 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ProblemSpec:
    """Container for a benchmark problem definition."""

    name: str
    d: int
    bounds: np.ndarray
    X0: np.ndarray
    y_star: float
    objective: Callable[[np.ndarray], np.ndarray]


def objective_minimization_2d(X: np.ndarray) -> np.ndarray:
    """Two-dimensional synthetic minimization objective."""
    X = np.asarray(X, dtype=float)
    X_eval = X.reshape(1, -1) if X.ndim == 1 else X
    x, y = X_eval[:, 0], X_eval[:, 1]

    alpha = 0.1
    A = np.array([4.0, 3.0, 2.0], dtype=float)
    B = np.array([0.08, 0.05, 0.02], dtype=float)
    C = np.array([[0.9, 0.3], [0.1, 0.8], [0.6, 0.7]], dtype=float)
    D = 2.0

    val = alpha * (x**2 + y**2)
    for Ai, Bi, (xi, yi) in zip(A, B, C):
        r2 = (x - xi) ** 2 + (y - yi) ** 2
        val -= Ai * np.exp(-r2 / Bi)
    val += D
    return val


_PROBLEMS: dict[str, ProblemSpec] = {
    "problem2d": ProblemSpec(
        name="problem2d",
        d=2,
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float),
        X0=np.array(
            [
                [0.25, 0.25],
                [0.25, 0.75],
                [0.75, 0.25],
                [0.75, 0.75],
            ],
            dtype=float,
        ),
        y_star=-1.9101873427908376,
        objective=objective_minimization_2d,
    ),
}


def load_problem(name: str) -> ProblemSpec:
    """Return a problem by name."""
    if name not in _PROBLEMS:
        available = ", ".join(sorted(_PROBLEMS))
        raise ValueError(f"Unknown problem '{name}'. Available problems: {available}.")

    problem = _PROBLEMS[name]
    return ProblemSpec(
        name=problem.name,
        d=problem.d,
        bounds=problem.bounds.copy(),
        X0=problem.X0.copy(),
        y_star=float(problem.y_star),
        objective=problem.objective,
    )


def list_problem_names() -> list[str]:
    """List available problem names."""
    return sorted(_PROBLEMS.keys())
