from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .backend import BackendInfo, BackendName, resolve_backend
from .loop_exactbo import ExactBOLoop

Array = np.ndarray


@dataclass
class ExactBORunResult:
    """Unified result payload for numpy and cupynumeric backends."""

    X: Array
    y: Array
    backend: BackendInfo


def _normalize_epsilon(epsilon: Array | float, dim: int) -> Array:
    """Normalize epsilon to a per-dimension array."""

    eps = np.asarray(epsilon, dtype=float)
    if eps.ndim == 0:
        return np.full((dim,), float(eps), dtype=float)
    if eps.shape == (dim,):
        return eps
    raise ValueError(f"epsilon must be scalar or shape ({dim},), got {eps.shape}")


def _evaluate_objective(f: Callable[[Array], Array], x: Array) -> Array:
    """Evaluate objective and return a flattened array."""

    x = np.asarray(x, dtype=float)
    x_eval = x.reshape(1, -1) if x.ndim == 1 else x
    return np.asarray(f(x_eval), dtype=float).ravel()


def run_exactbo(
    x0: Array,
    bounds: Array,
    epsilon: Array | float,
    gp,
    f: Callable[[Array], Array],
    max_iters: int,
    max_partitions: int,
    *,
    backend: BackendName = "auto",
    split_type: str = "centered",
    verbose: bool = False,
) -> ExactBORunResult:
    """
    Run ExactBO with backend selection and CPU fallback.

    Parameters
    ----------
    x0 : ndarray, shape (n0, d)
        Initial evaluated points.
    bounds : ndarray, shape (d, 2)
        Search-space bounds.
    epsilon : float or ndarray, shape (d,)
        Partition termination threshold(s).
    gp : sklearn-like regressor
        Surrogate model with .fit/.predict plus sklearn GP attributes.
    f : callable
        Objective function.
    max_iters : int
        Outer BO iterations.
    max_partitions : int
        Max partition loops per BO iteration.
    backend : {"auto", "numpy", "cupynumeric"}, default="auto"
        Execution backend.
    split_type : {"centered", "full"}, default="centered"
        Split strategy for numpy backend. cupynumeric backend currently uses
        a DIRECT/centered split implementation.
    verbose : bool, default=False
        Print loop-level progress.

    Returns
    -------
    ExactBORunResult
        Final design points, objective values, and backend resolution info.
    """

    x0 = np.asarray(x0, dtype=float)
    bounds = np.asarray(bounds, dtype=float)

    if x0.ndim != 2:
        raise ValueError(f"x0 must be 2D with shape (n0, d), got {x0.shape}")
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"bounds must have shape (d, 2), got {bounds.shape}")

    dim = bounds.shape[0]
    if x0.shape[1] != dim:
        raise ValueError(
            f"x0 second dimension ({x0.shape[1]}) must match bounds dimension ({dim})"
        )

    eps = _normalize_epsilon(epsilon, dim)
    backend_info = resolve_backend(backend)

    if backend_info.selected == "cupynumeric":
        from .vectorized_loop import exactbo_loop_cupynumeric

        x_data, y_data = exactbo_loop_cupynumeric(
            x0=x0,
            bounds=bounds,
            epsilon=eps,
            gp=gp,
            f=f,
            max_iters=max_iters,
            max_partitions=max_partitions,
            verbose=verbose,
        )
        return ExactBORunResult(X=np.asarray(x_data), y=np.asarray(y_data), backend=backend_info)

    y0 = _evaluate_objective(f, x0)

    loop = ExactBOLoop(model=gp, bounds=bounds, precision=eps, log="none")
    loop.set_oracle(lambda x: _evaluate_objective(f, x))

    res = loop.run(
        X0=x0,
        y0=y0,
        budget=max_iters,
        max_splits=max_partitions,
        split_type=split_type,
    )

    return ExactBORunResult(X=res.X, y=res.y, backend=backend_info)
