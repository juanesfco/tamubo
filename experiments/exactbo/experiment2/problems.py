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


def objective_least_squares_5d(X: np.ndarray) -> np.ndarray:
    """Five-dimensional nonlinear least-squares objective."""
    X = np.asarray(X, dtype=float)
    X_eval = X.reshape(1, -1) if X.ndim == 1 else X
    x0, x1, x2, x3, x4 = [X_eval[:, i] for i in range(5)]

    t1 = -4.583 - 3.933 * x0 + 0.107 * x1 + 0.126 * x2 - 9.99 * x4
    t2 = 1.4185 - 0.987 * x1 - 22.95 * x3
    t3 = -0.0921 + 0.002 * x0 - 0.235 * x2 + 5.67 * x4
    t4 = 0.0084 + x1 - x3
    t5 = -0.00071 - x2 - 0.196 * x4

    t6 = -0.727 * x1 * x2 + 8.39 * x2 * x3 - 684.4 * x3 * x4 + 63.5 * x3 * x1
    t7 = 0.949 * x0 * x2 + 0.173 * x0 * x4
    t8 = -0.716 * x0 * x1 - 1.578 * x0 * x3 + 1.132 * x3 * x1
    t9 = -x0 * x4
    t10 = x0 * x3

    return (t1 + t6) ** 2 + (t2 + t7) ** 2 + (t3 + t8) ** 2 + (t4 + t9) ** 2 + (t5 + t10) ** 2


_PROBLEM10D_A = np.array(
    [
        [-0.8123, 0.2413, -0.2964, 0.2484, 2.3081, -1.3713, 3.0762, -4.4416, -0.0310, -0.1894],
        [0.6772, 0.0609, -1.2558, -0.0741, 3.4272, 1.2321, -3.4688, -6.2450, -0.6963, -0.7903],
        [-0.3365, 0.5358, 0.8026, 1.9580, 1.4212, 1.9957, -1.1197, 4.1086, 0.5639, -0.7754],
        [-0.4282, 0.4834, -0.4652, 1.4991, -2.8462, 0.6664, -1.0966, -4.4469, -0.7988, 0.5689],
        [0.5119, 0.6563, 0.3090, 1.1751, 1.7230, 1.9228, 1.5512, -1.6881, -0.4119, -0.4169],
        [-1.0177, -0.4945, 0.6911, -2.2498, -3.0143, -2.1242, 0.8211, 0.7823, -0.5253, 0.2071],
        [-0.9385, 0.4491, -1.1320, -2.1417, -2.9591, 1.8822, 2.4259, -4.0952, 0.0617, 0.9288],
        [0.3490, 0.1322, -1.0659, -2.0633, 1.0886, 3.7698, -1.1095, 2.1446, -0.8170, -0.1350],
    ],
    dtype=float,
)
_PROBLEM10D_B = np.array(
    [1.7367, 6.9483, -1.1465, 2.6396, -0.5015, -0.2883, 2.1894, 1.8491],
    dtype=float,
)
_PROBLEM10D_X_STAR = np.array(
    [-0.5232654244, 0.0, 0.0, -0.4590488748, 0.2186127289, 0.2388229495, -0.6444044675, -0.5958475955, 0.0, 0.0],
    dtype=float,
)
_PROBLEM10D_BOUNDS = np.tile(np.array([[-1.0, 1.0]], dtype=float), (10, 1))


def objective_convex_l1_10d(X: np.ndarray) -> np.ndarray:
    """Ten-dimensional L1-regularized least-squares objective."""
    X = np.asarray(X, dtype=float)
    X_eval = X.reshape(1, -1) if X.ndim == 1 else X
    residual = X_eval @ _PROBLEM10D_A.T - _PROBLEM10D_B
    smooth = 0.5 * np.sum(residual**2, axis=1)
    penalty = np.sum(np.abs(X_eval), axis=1)
    return smooth + penalty


_PROBLEM10D_Y_STAR = float(objective_convex_l1_10d(_PROBLEM10D_X_STAR)[0])


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
    "problem5d": ProblemSpec(
        name="problem5d",
        d=5,
        bounds=np.array(
            [
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
            ],
            dtype=float,
        ),
        X0=np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float),
        y_star=0.0,
        objective=objective_least_squares_5d,
    ),
    "problem10d": ProblemSpec(
        name="problem10d",
        d=10,
        bounds=_PROBLEM10D_BOUNDS,
        X0=np.zeros((1, 10), dtype=float),
        y_star=_PROBLEM10D_Y_STAR,
        objective=objective_convex_l1_10d,
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
