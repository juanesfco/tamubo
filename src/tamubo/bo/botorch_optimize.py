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
from botorch.optim import optimize_acqf

from tamubo.utils import BOResult, _as_result, _evaluate_objective, _init_log, _normalize_inputs

__all__ = ["run_botorch_optimize_ei"]


def run_botorch_optimize_ei(
    X0: np.ndarray,
    bounds: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    max_iters: int,
    *,
    num_restarts: int = 10,
    raw_samples: int = 128,
    maxiter: int = 200,
    validation: bool = True,
    verbose: bool = False,
    logMask: bool = False,
    device: str = "cuda",
) -> BOResult:
    """
    BO workflow: BoTorch SingleTaskGP(RBF) + EI maximization via optimize_acqf.

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
    num_restarts : int, default=10
        Number of multistart optimization restarts in optimize_acqf.
    raw_samples : int, default=128
        Number of raw initialization samples in optimize_acqf.
    maxiter : int, default=200
        Maximum optimizer iterations for each restart.
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
    if validation:
        if int(num_restarts) <= 0:
            raise ValueError(f"num_restarts must be > 0, got {num_restarts}")
        if int(raw_samples) <= 0:
            raise ValueError(f"raw_samples must be > 0, got {raw_samples}")
        if int(maxiter) <= 0:
            raise ValueError(f"maxiter must be > 0, got {maxiter}")

    torch_device = torch.device(device)
    bounds_t = torch.as_tensor(search_bounds.T, dtype=torch.double, device=torch_device)
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
        candidate, acq_value = optimize_acqf(
            acq_function=acqf,
            bounds=bounds_t,
            q=1,
            num_restarts=int(num_restarts),
            raw_samples=int(raw_samples),
            options={"maxiter": int(maxiter)},
        )

        Xn = candidate.detach().cpu().numpy().reshape(-1)
        yn = _evaluate_objective(f, Xn)
        ei_best = np.exp(float(acq_value.detach().cpu().item()))

        if verbose:
            print(f"Evaluated new point: {Xn} -> {yn}")

        if logMask:
            log[f"i{iteration}"].update(
                {
                    "Xn": Xn.copy(),
                    "yn": yn.copy(),
                    "ei_max": ei_best,
                    "num_restarts": int(num_restarts),
                    "raw_samples": int(raw_samples),
                    "maxiter": int(maxiter),
                }
            )

        X = np.vstack((X, Xn))
        y = np.hstack((y, yn))

    return _as_result(X, y, log)

