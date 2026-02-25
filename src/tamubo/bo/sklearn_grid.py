from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from tamubo.acquisition_functions import expected_improvement

from tamubo.utils import (
    BOResult,
    _as_result,
    _build_cartesian_grid,
    _evaluate_objective,
    _init_log,
    _normalize_inputs,
)

__all__ = ["run_sklearn_grid_ei"]


def _default_sklearn_gp() -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3)) 
        * RBF(length_scale=0.2,length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
    )
    return GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True)


def run_sklearn_grid_ei(
    X0: np.ndarray,
    bounds: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    max_iters: int,
    *,
    gp: GaussianProcessRegressor | None = None,
    grid_resolution: int = 50,
    validation: bool = True,
    verbose: bool = False,
    logMask: bool = False,
) -> BOResult:
    """
    BO workflow: sklearn GP (RBF kernel) + EI maximization via cartesian grid search.

    Parameters
    ----------
    X0 : ndarray, shape (N0, d)
        Initial evaluated points.
    bounds : ndarray, shape (d, 2)
        Search-space bounds, [lower, upper] per dimension.
    f : callable
        Objective function.
    max_iters : int
        Number of BO iterations.
    gp : GaussianProcessRegressor, optional
        If omitted, a default RBF-based GP is created.
    grid_resolution : int, default=50
        Number of evenly spaced points per dimension used for EI grid search.
    validation : bool, default=True
        Run shape/value checks.
    verbose : bool, default=False
        Print per-iteration progress.
    logMask : bool, default=False
        Enable logging of intermediate data.
    """
    X, search_bounds, _ = _normalize_inputs(X0, bounds, validation=validation)
    iterations = int(max_iters)
    if validation and iterations < 0:
        raise ValueError(f"max_iters must be >= 0, got {iterations}")

    gp_model = gp if gp is not None else _default_sklearn_gp()
    grid = _build_cartesian_grid(search_bounds, grid_resolution, validation=validation)
    log = _init_log(logMask)

    y = _evaluate_objective(f, X)
    for iteration in range(iterations):
        if verbose:
            print(f"Iteration {iteration + 1}/{iterations}")
            print(f"Current training data: \nX: {X}, \ny: {y}")

        if logMask:
            log[f"i{iteration}"] = {"X": X.copy(), "y": y.copy()}

        gp_model.fit(X, y)
        mu_grid, sigma_grid = gp_model.predict(grid, return_std=True)
        ei_grid = expected_improvement(mu_grid, sigma_grid, np.min(y), backend="numpy")
        idx_best = int(np.argmax(ei_grid))
        Xn = grid[idx_best]
        yn = _evaluate_objective(f, Xn)

        if verbose:
            print(f"Evaluated new point: {Xn} -> {yn}")

        if logMask:
            log[f"i{iteration}"].update(
                {
                    "Xn": Xn.copy(),
                    "yn": yn.copy(),
                    "ei_max": float(ei_grid[idx_best]),
                    "grid_resolution": int(grid_resolution),
                    "grid_points": int(grid.shape[0]),
                }
            )

        X = np.vstack((X, Xn))
        y = np.hstack((y, yn))

    return _as_result(X, y, log)

