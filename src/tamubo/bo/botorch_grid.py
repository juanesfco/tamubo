from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
import time

try:
    import torch
    from botorch.acquisition import LogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.constraints import Interval
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood
except ModuleNotFoundError:
    raise ImportError(
        "BoTorch and its dependencies are required to use the botorch_grid module. "
        "Please install BoTorch and try again."
    )

from tamubo.utils import (
    BOResult,
    _as_result,
    _evaluate_objective,
    _from_unit_cube,
    _init_log,
    _normalize_inputs,
    _normalize_problem_to_unit_cube,
)

__all__ = ["run_botorch_grid_ei"]

def _default_sklearn_gp(d) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3)) 
        * RBF(length_scale=np.full(d, 0.2),length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
    )
    return GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True)


def _build_cartesian_grid_torch(
    bounds: np.ndarray,
    grid_resolution: int,
    *,
    device: torch.device,
    validation: bool = True,
) -> torch.Tensor:
    resolution = int(grid_resolution)
    if validation and resolution < 2:
        raise ValueError(f"grid_resolution must be >= 2, got {resolution}")

    axes = [
        torch.linspace(float(lower), float(upper), resolution, dtype=torch.double, device=device)
        for lower, upper in np.asarray(bounds, dtype=np.float64)
    ]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([axis.reshape(-1) for axis in mesh], dim=1)

def _gpu_warmup(device: torch.device) -> None:
    if device.type == "cuda":
        # A small, non-essential matrix multiplication for warm-up
        dummy_tensor = torch.randn(10, 10).to(device)
        _ = torch.matmul(dummy_tensor, dummy_tensor)
        # Synchronize to ensure the operation completes
        torch.cuda.synchronize()


def run_botorch_grid_ei(
    X0: np.ndarray,
    bounds: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    max_iters: int,
    *,
    gp_sk: bool | GaussianProcessRegressor = False,
    grid_resolution: int = 50,
    validation: bool = True,
    verbose: bool = False,
    logMask: bool = False,
    device: str = "cuda",
    normalize_to_unit_cube: bool = False,
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
    gp_sk : bool or sklearn GP, default=False
        If True, fit a default sklearn GP to the data and copy its parameters to the 
        BoTorch model, without optimizing the BoTorch model's parameters. If a sklearn 
        GP instance is provided, fit it to the data and copy its parameters to the 
        BoTorch model. If False, use the default BoTorch GP with its parameters 
        optimized via MLL.  
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
    normalize_to_unit_cube : bool, default=False
        If True, optimize internally on [0, 1]^d and evaluate the objective after
        mapping candidates back to the original finite bounds.
    """
    X, search_bounds, d = _normalize_inputs(X0, bounds, validation=validation)
    objective = f
    physical_bounds = None
    if normalize_to_unit_cube:
        X, search_bounds, objective, physical_bounds = _normalize_problem_to_unit_cube(
            X,
            search_bounds,
            f,
            validation=validation,
        )
    iterations = int(max_iters)
    if validation and iterations < 0:
        raise ValueError(f"max_iters must be >= 0, got {iterations}")

    torch_device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    _gpu_warmup(torch_device)
    grid_t = _build_cartesian_grid_torch(
        search_bounds, grid_resolution, device=torch_device, validation=validation
    )
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
            # Create default sklearn GP and fit to current data to extract kernel parameters
            gp = _default_sklearn_gp(d)
            # Fit the provided GP to the current data
            gp.fit(X, y)
            # Extract the kernel parameters and noise level from the fitted GP
            k = gp.kernel_
            sigma_f2 = float(k.k1.k1.constant_value)
            sigma_n2 = float(k.k2.noise_level)
            lengthscales = np.asarray(k.k1.k2.length_scale, dtype=float).reshape(-1)
        
        elif isinstance(gp_sk, GaussianProcessRegressor):
            # Fit the provided GP to the current data
            gp_sk.fit(X, y)
            k = gp_sk.kernel_
            sigma_f2 = float(k.k1.k1.constant_value)
            sigma_n2 = float(k.k2.noise_level)
            lengthscales = np.asarray(k.k1.k2.length_scale, dtype=float).reshape(-1)

        X_t = torch.as_tensor(X, dtype=torch.double, device=torch_device)
        y_t = torch.as_tensor(y.reshape(-1, 1), dtype=torch.double, device=torch_device)
        n = X.shape[0]
        ddof_correction = (n - 1) / n if n > 1 else 1.0

        covar_module = ScaleKernel(
            RBFKernel(
                ard_num_dims=d,
                lengthscale_constraint=Interval(1e-2, 1e3),
            ),
            outputscale_constraint=Interval(1e-2*ddof_correction, 1e3*ddof_correction),
        )
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-10*ddof_correction, 1e1*ddof_correction))

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
            for p in model.parameters():
                p.requires_grad = False
        else:
            model.covar_module.outputscale = 1.0*ddof_correction
            model.likelihood.noise = 1e-3*ddof_correction
            model.covar_module.base_kernel.lengthscale = 0.2*torch.ones(d, dtype=torch.double, device=torch_device)

            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
 
        model.eval()

        if torch_device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        acqf = LogExpectedImprovement(model=model, best_f=float(np.min(y)), maximize=False)
        with torch.no_grad():
            ei_grid = acqf(grid_t.unsqueeze(-2))
            idx_best = int(torch.argmax(ei_grid).item())
            ei_best = np.exp(float(ei_grid[idx_best].item()))

        Xn = grid_t[idx_best].detach().cpu().numpy()

        if torch_device.type == "cuda":
            torch.cuda.synchronize()

        dt = time.perf_counter() - t0

        yn = _evaluate_objective(objective, Xn)
        Xn_display = (
            _from_unit_cube(Xn, physical_bounds, validation=False)
            if physical_bounds is not None
            else Xn
        )

        if verbose:
            print(f"Evaluated new point: {Xn_display} -> {yn}")

        if logMask:
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
