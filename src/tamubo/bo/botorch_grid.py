from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import Interval
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from tamubo.utils import (
    BOResult,
    _as_result,
    _build_cartesian_grid,
    _evaluate_objective,
    _init_log,
    _normalize_inputs,
)

__all__ = ["run_botorch_grid_ei"]


def run_botorch_grid_ei(
    X0: np.ndarray,
    bounds: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    max_iters: int,
    *,
    grid_resolution: int = 50,
    validation: bool = True,
    verbose: bool = False,
    logMask: bool = False,
    device: str = "cuda",
) -> BOResult:
    """
    BO workflow: BoTorch SingleTaskGP(RBF) + EI maximization via cartesian grid search.

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
    grid_resolution : int, default=50
        Number of evenly spaced points per dimension used for EI grid search.
    validation : bool, default=True
        Run shape/value checks.
    verbose : bool, default=False
        Print per-iteration progress.
    logMask : bool, default=False
        Enable logging of intermediate data.
    device : str, default="cuda"
        Torch device passed to model/acquisition tensors.
    """
    X, search_bounds, _ = _normalize_inputs(X0, bounds, validation=validation)
    iterations = int(max_iters)
    if validation and iterations < 0:
        raise ValueError(f"max_iters must be >= 0, got {iterations}")

    grid = _build_cartesian_grid(search_bounds, grid_resolution, validation=validation)
    torch_device = torch.device(device)
    grid_t = torch.as_tensor(grid, dtype=torch.double, device=torch_device)
    log = _init_log(logMask)

    y = _evaluate_objective(f, X)
    for iteration in range(iterations):
        if verbose:
            print(f"Iteration {iteration + 1}/{iterations}")
            print(f"Current training data: \nX: {X}, \ny: {y}")

        if logMask:
            log[f"i{iteration}"] = {"X": X.copy(), "y": y.copy()}

        X_t = torch.as_tensor(X, dtype=torch.double, device=torch_device)
        y_t = torch.as_tensor(y.reshape(-1, 1), dtype=torch.double, device=torch_device)

        covar_module = ScaleKernel(
            RBFKernel(
                lengthscale_constraint=Interval(1e-2, 1e2),
            ),
            outputscale_constraint=Interval(1e-2, 1e3),
        )
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-10, 1e1))

        model = SingleTaskGP(
            train_X=X_t,
            train_Y=y_t,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
        )
        model.covar_module.outputscale = 1.0
        model.covar_module.base_kernel.lengthscale = 0.2
        model.likelihood.noise = 1e-3

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()

        acqf = LogExpectedImprovement(model=model, best_f=float(np.min(y)), maximize=False)
        with torch.no_grad():
            ei_grid = acqf(grid_t.unsqueeze(-2))
            idx_best = int(torch.argmax(ei_grid).item())
            ei_best = np.exp(float(ei_grid[idx_best].item()))

        Xn = grid[idx_best]
        yn = _evaluate_objective(f, Xn)

        if verbose:
            print(f"Evaluated new point: {Xn} -> {yn}")

        if logMask:
            log[f"i{iteration}"].update(
                {
                    "Xn": Xn.copy(),
                    "yn": yn.copy(),
                    "ei_max": ei_best,
                    "grid_resolution": int(grid_resolution),
                    "grid_points": int(grid.shape[0]),
                }
            )

        X = np.vstack((X, Xn))
        y = np.hstack((y, yn))

    return _as_result(X, y, log)
