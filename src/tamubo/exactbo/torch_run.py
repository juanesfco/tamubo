from __future__ import annotations

from typing import Callable
import time

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

try:
    import torch
    from botorch.acquisition import ExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.constraints import Interval
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood
except ModuleNotFoundError as exc:
    raise ImportError(
        "BoTorch and its dependencies are required to use tamubo.exactbo.torch_run. "
        "Please install BoTorch and try again."
    ) from exc

from tamubo.utils import (
    BOResult,
    _as_result,
    _evaluate_objective,
    _from_unit_cube,
    _init_log,
    _normalize_inputs,
    _normalize_problem_to_unit_cube,
)

from .botorch import optimize_acqf_exactbo

__all__ = ["run_botorch_exactbo_ei"]


def _default_sklearn_gp(dim: int) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(length_scale=np.full(dim, 0.2), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
    )
    return GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True)


def _normalize_epsilon(epsilon: np.ndarray | float, dim: int) -> np.ndarray:
    eps = np.asarray(epsilon, dtype=np.float64)
    if eps.ndim == 0:
        return np.full((dim,), float(eps), dtype=np.float64)
    if eps.shape == (dim,):
        return eps
    raise ValueError(f"epsilon_X must be scalar or shape ({dim},), got {eps.shape}")


def _gpu_warmup(device: torch.device) -> None:
    if device.type == "cuda":
        dummy = torch.randn(10, 10, dtype=torch.double, device=device)
        _ = dummy @ dummy
        torch.cuda.synchronize()


def run_botorch_exactbo_ei(
    X0: np.ndarray,
    bounds: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    max_iters: int,
    *,
    gp_sk: bool | GaussianProcessRegressor = False,
    epsilon_X: np.ndarray | float = 1e-3,
    epsilon_ei: float = 1e-6,
    max_partitions: int = 100,
    validation: bool = True,
    verbose: bool = False,
    logMask: bool = False,
    device: str = "cuda",
    normalize_to_unit_cube: bool = False,
) -> BOResult:
    """
    Full BoTorch ExactBO workflow.

    This mirrors the existing BoTorch benchmark runners, but replaces
    ``optimize_acqf`` with the ExactBO partition search.
    """
    X, search_bounds, dim = _normalize_inputs(X0, bounds, validation=validation)
    epsilon_X = _normalize_epsilon(epsilon_X, dim)

    iterations = int(max_iters)
    if validation and iterations < 0:
        raise ValueError(f"max_iters must be >= 0, got {iterations}")

    objective = f
    physical_bounds = None
    if normalize_to_unit_cube:
        X, search_bounds, objective, physical_bounds = _normalize_problem_to_unit_cube(
            X,
            search_bounds,
            f,
            validation=validation,
        )

    torch_device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    _gpu_warmup(torch_device)
    bounds_t = torch.as_tensor(search_bounds.T, dtype=torch.double, device=torch_device)
    log = _init_log(logMask)

    y = _evaluate_objective(objective, X)
    for iteration in range(iterations):
        X_display = (
            _from_unit_cube(X, physical_bounds, validation=False)
            if physical_bounds is not None
            else X
        )

        if verbose:
            print(f"Iteration {iteration + 1}/{iterations}")
            print(f"Current training data: \nX: {X_display}, \ny: {y}")

        if logMask:
            log[f"i{iteration}"] = {"X": X_display.copy(), "y": y.copy()}

        if gp_sk is True:
            gp = _default_sklearn_gp(dim)
            gp.fit(X, y)
            kernel = gp.kernel_
            sigma_f2 = float(kernel.k1.k1.constant_value)
            sigma_n2 = float(kernel.k2.noise_level)
            lengthscales = np.asarray(kernel.k1.k2.length_scale, dtype=float).reshape(-1)
        elif isinstance(gp_sk, GaussianProcessRegressor):
            gp_sk.fit(X, y)
            kernel = gp_sk.kernel_
            sigma_f2 = float(kernel.k1.k1.constant_value)
            sigma_n2 = float(kernel.k2.noise_level)
            lengthscales = np.asarray(kernel.k1.k2.length_scale, dtype=float).reshape(-1)

        X_t = torch.as_tensor(X, dtype=torch.double, device=torch_device)
        y_t = torch.as_tensor(y.reshape(-1, 1), dtype=torch.double, device=torch_device)
        n_train = X.shape[0]
        ddof_correction = (n_train - 1) / n_train if n_train > 1 else 1.0

        covar_module = ScaleKernel(
            RBFKernel(
                ard_num_dims=dim,
                lengthscale_constraint=Interval(1e-3, 1e3),
            ),
            outputscale_constraint=Interval(1e-2 * ddof_correction, 1e3 * ddof_correction),
        )
        likelihood = GaussianLikelihood(
            noise_constraint=Interval(1e-10 * ddof_correction, 1e1 * ddof_correction)
        )

        model = SingleTaskGP(
            train_X=X_t,
            train_Y=y_t,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
        )

        if gp_sk is True or isinstance(gp_sk, GaussianProcessRegressor):
            ls_t = torch.as_tensor(lengthscales, dtype=torch.double, device=torch_device).reshape(1, -1)
            model.covar_module.outputscale = sigma_f2 * ddof_correction
            model.likelihood.noise = sigma_n2 * ddof_correction
            model.covar_module.base_kernel.lengthscale = ls_t
            for parameter in model.parameters():
                parameter.requires_grad = False
        else:
            model.covar_module.outputscale = 1.0 * ddof_correction
            model.likelihood.noise = 1e-3 * ddof_correction
            model.covar_module.base_kernel.lengthscale = 0.2 * torch.ones(
                dim,
                dtype=torch.double,
                device=torch_device,
            )

            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

        model.eval()

        if torch_device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        acqf = ExpectedImprovement(model=model, best_f=float(np.min(y)), maximize=False)
        if logMask:
            candidate, acq_value, opt_log = optimize_acqf_exactbo(
                acqf,
                bounds_t,
                epsilon_X,
                float(epsilon_ei),
                max_partitions,
                return_log=True,
                validation=validation,
                verbose=verbose,
                logMask=True,
            )
        else:
            candidate, acq_value = optimize_acqf_exactbo(
                acqf,
                bounds_t,
                epsilon_X,
                float(epsilon_ei),
                max_partitions,
                validation=validation,
                verbose=verbose,
            )
            opt_log = None

        if torch_device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        Xn = candidate.detach().cpu().numpy().reshape(-1)
        yn = _evaluate_objective(objective, Xn)
        Xn_display = (
            _from_unit_cube(Xn, physical_bounds, validation=False)
            if physical_bounds is not None
            else Xn
        )
        ei_best = float(acq_value.detach().cpu().item())

        if verbose:
            print(f"Evaluated new point: {Xn_display} -> {yn}")

        if logMask:
            log[f"i{iteration}"].update(opt_log)
            log[f"i{iteration}"].update(
                {
                    "Xn": Xn_display.copy(),
                    "yn": yn.copy(),
                    "ei_max": ei_best,
                    "time": dt,
                }
            )

        X = np.vstack((X, Xn))
        y = np.hstack((y, yn))

    X_result = (
        _from_unit_cube(X, physical_bounds, validation=False)
        if physical_bounds is not None
        else X
    )
    return _as_result(X_result, y, backend=torch_device.type, log=log)
