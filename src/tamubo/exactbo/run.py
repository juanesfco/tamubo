from __future__ import annotations

from importlib import import_module
from dataclasses import dataclass
from typing import Callable

import numpy as np

from tamubo.utils import BackendInfo, BackendName, resolve_backend
from tamubo.acquisition_functions import expected_improvement
from .bounds import rbf_k_bounds, mu_bounds, sigma_bounds, ei_bounds
from .partition import split_boxes

@dataclass
class ExactBOResult:
    """Unified result payload for numpy and cupynumeric backends."""
    X: np.ndarray # (N,d)
    y: np.ndarray # (N,)
    backend: BackendInfo
    log: dict | None = None

@dataclass
class ExactBOPartitioningResult:
    """Unified result payload for the partitioning loop."""
    Xn: np.ndarray # (d,)
    logIteration: dict | None = None

def _normalize_epsilon(epsilon: np.ndarray | float, dim: int) -> np.ndarray:
    """Normalize epsilon to a per-dimension array."""
    eps = np.asarray(epsilon, dtype=float)
    if eps.ndim == 0:
        return np.full((dim,), float(eps), dtype=float)
    if eps.shape == (dim,):
        return eps
    raise ValueError(f"epsilon must be scalar or shape ({dim},), got {eps.shape}")

def _evaluate_objective(f: Callable[[np.ndarray], np.ndarray], X: np.ndarray) -> np.ndarray:
    """Evaluate objective and return a flattened array."""
    X = np.asarray(X, dtype=float)
    X_eval = X.reshape(1, -1) if X.ndim == 1 else X
    return np.asarray(f(X_eval), dtype=float).ravel()

def _array_module(backend: BackendName = "auto"):
    """Return the resolved array module (`numpy` or `cupynumeric`)."""
    backend_info = resolve_backend(backend)
    if backend_info.selected == "numpy":
        return np
    # Import cupynumeric only when it is the selected backend.
    return import_module("cupynumeric")


def exactbo(
    X0: np.ndarray,
    bounds: np.ndarray,
    epsilon_X: np.ndarray | float,
    epsilon_ei: float,
    gp,
    f: Callable[[np.ndarray], np.ndarray],
    max_iters: int,
    max_partitions: int,
    *,
    backend: BackendName = "auto",
    validation: bool = True,
    verbose: bool = False,
    logMask: bool = False,
) -> ExactBOResult:
    """
    Run ExactBO with backend selection and CPU fallback.

    Parameters
    ----------
    X0 : ndarray, shape (N0, d)
        Initial evaluated points.
    bounds : ndarray, shape (d, 2)
        Search-space bounds, [lower, upper] per dim.
    epsilon_X : float or ndarray, shape (d,)
        Partition termination threshold(s) for the input space.
    epsilon_ei : float
        Threshold for the expected improvement values.
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
    validation : bool, default=True
        Run additional checks for validation purposes (not optimized).
    verbose : bool, default=False
        Print loop-level progress.
    logMask : bool, default=False
        Enable logging of intermediate results.

    Returns
    -------
    ExactBORunResult
        Final design points, objective values, backend resolution info and log.
    """
    
    X = np.asarray(X0, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)

    if validation:
        if X.ndim != 2:
            raise ValueError(f"X0 must be 2D with shape (N0, d), got {X.shape}")
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"bounds must have shape (d, 2), got {bounds.shape}")

    dim = bounds.shape[0]
    if validation:
        if X.shape[1] != dim:
            raise ValueError(
                f"X0 second dimension ({X.shape[1]}) must match bounds dimension ({dim})"
            )

    epsilon_X = _normalize_epsilon(epsilon_X, dim)
    backend_info = resolve_backend(backend)

    if logMask:
        log = {}

    ei_actives = [] #ELIMINATE
    for iteration in range(max_iters):
        if verbose:
            print(f"Iteration {iteration + 1}/{max_iters}")
        # Evaluate function at current data points
        y = _evaluate_objective(f, X)  # (N,)
        if verbose:
            print(f"Current training data: \nX: {X}, \ny: {y}")
        if logMask:
            log[f"i{iteration}"] = {"X": X.copy(), "y": y.copy()}

        # Fit Gaussian Process
        gp.fit(X, y)
        if verbose:
            print("GP fitted")

        # Run partitioning to find next point and evaluate it
        partitioning_result, ei_active = exactbo_partitioning(X, bounds, epsilon_X, epsilon_ei, gp, max_partitions, backend=backend_info.selected, validation=validation, verbose=verbose, logMask=logMask) #ELIMINATE
        #partitioning_result = exactbo_partitioning(X, bounds, epsilon_X, epsilon_ei, gp, max_partitions, backend=backend_info.selected, validation=validation, verbose=verbose, logMask=logMask)
        Xn = np.asarray(partitioning_result.Xn, dtype=np.float64).ravel()  # (d,)
        yn = _evaluate_objective(f, Xn)  # (1,)
        if verbose:
            print(f"Evaluated new point: {Xn} -> {yn}")
        if logMask:
            log[f"i{iteration}"].update(partitioning_result.logIteration)
            log[f"i{iteration}"].update({"Xn": Xn.copy(), "yn": yn.copy()})

        # Update data
        X = np.vstack((X, Xn))  # (N+1,d)
        y = np.hstack((y, yn))  # (N+1,)

        ei_actives.append(ei_active) #ELIMIATE
    
    return ExactBOResult(X, y, backend_info, log if logMask else None), ei_actives #ELIMINATE
    #return ExactBOResult(X, y, backend_info, log if logMask else None)


def exactbo_partitioning(
    X: np.ndarray,
    bounds: np.ndarray,
    epsilon_X: np.ndarray,
    epsilon_ei: float,
    gp,
    max_partitions: int,
    *,
    backend: BackendName = "auto",
    validation: bool = True,
    verbose: bool = False,
    logMask: bool = False,
) -> ExactBOPartitioningResult:
    """
    Run ExactBO Partitioning with backend selection.

    Parameters
    ----------
    X : ndarray, shape (N, d)
        Evaluated points.
    bounds : ndarray, shape (d, 2)
        Search-space bounds, [lower, upper] per dim.
    epsilon_X : float or ndarray, shape (d,)
        Partition termination threshold(s) for the input space.
    epsilon_ei : float
        Threshold for the expected improvement values.
    gp : sklearn-like regressor
        Surrogate model with .fit/.predict plus sklearn GP attributes.
    iteration : int
        Current BO iteration.
    max_partitions : int
        Max partition loops per BO iteration.
    backend : {"auto", "numpy", "cupynumeric"}, default="auto"
        Backend used for array ops.
    validation : bool, default=True
        Run additional checks for validation purposes (not optimized).
    verbose : bool, default=False
        Print loop-level progress.
    log : bool, default=False
        Enable logging of intermediate results.

    Returns
    -------
    ExactBOPartitioningResult
        Next design point, backend resolution info and log.
    """
    xp = _array_module(backend)

    # Copy of X for partitioning, converted to backend array
    Xc = xp.asarray(X, dtype=xp.float64)

    # Initialize boxes
    ## One row per box (initially one box)
    ## One column per dimension
    bounds_L = xp.asarray([bounds[:, 0]], dtype=xp.float64)  # (n,d)
    bounds_U = xp.asarray([bounds[:, 1]], dtype=xp.float64)  # (n,d)

    # GP hyperparameters
    gp_kernel_params = gp.kernel_.get_params()
    sigma_f_2 = gp_kernel_params["k1__k1__constant_value"]
    length_scale = gp_kernel_params["k1__k2__length_scale"]
    alpha = xp.asarray(gp.alpha_, dtype=xp.float64)
    y_train_std = gp._y_train_std
    y_train_mean = gp._y_train_mean
    L = xp.asarray(gp.L_, dtype=xp.float64)  # (N,N)
    y_min = xp.min(gp.y_train_)
    y_min_unscaled = y_min * y_train_std + y_train_mean

    # Partition parameters
    N = Xc.shape[0]  # Number of data points
    d = bounds.shape[0]  # Number of dimensions
    w = bounds_U[0] - bounds_L[0]  # Bounds with per dimension (d,)
    epsilon_X = xp.asarray(epsilon_X, dtype=xp.float64)
    partition = 0
    w_max = w.copy()
    ei_max = 0
    active_boxes_mask = xp.ones((1,), dtype=bool)

    # Initialize log
    if logMask:
        log = {}
    
    while partition < max_partitions and xp.any(w_max > epsilon_X):
        # Total number of boxes
        n = bounds_L.shape[0]

        # Bounds
        ## Kernel
        K_lo = xp.zeros((n, N), dtype=xp.float64)
        K_hi = xp.zeros((n, N), dtype=xp.float64)
        for i in range(N):
            xi = Xc[i]
            K_lo[:, i], K_hi[:, i] = rbf_k_bounds(bounds_L.ravel(),bounds_U.ravel(),xi,n,d,sigma_f_2,length_scale,backend=backend,validation=validation)  # (n,) both
        ## Mean
        mu_lo, mu_hi = mu_bounds(alpha, K_lo, K_hi, n, N, y_train_mean=y_train_mean, y_train_std=y_train_std, backend=backend, validation=validation)  # (n,) both
        ## Sigma
        sig_lo, sig_hi = sigma_bounds(K_lo, K_hi, L, n, N, sigma_f_2, y_train_std=y_train_std, backend=backend, validation=validation)  # (n,) both
        ## EI
        ei_lo, ei_hi = ei_bounds(mu_lo, mu_hi, sig_lo, sig_hi, n, y_min_unscaled, backend=backend, validation=validation)  # (n,) both

        # Compute actual EI in the center of the hyperbox with highest upper EI bound
        idx_max_ei_hi = xp.argmax(ei_hi)
        max_ei_hi_box_L = bounds_L[idx_max_ei_hi, :]  # (d,)
        max_ei_hi_box_U = bounds_U[idx_max_ei_hi, :]  # (d,)
        max_ei_hi_box_center = (max_ei_hi_box_L + max_ei_hi_box_U) / 2.0  # (d,)
        mu_pred, sigma_pred = gp.predict(np.array(max_ei_hi_box_center).reshape(1, -1), return_std=True)
        ei_center = expected_improvement(mu_pred[0], sigma_pred[0], y_min_unscaled, backend=backend)  # scalar
        ei_max = max(ei_max, float(ei_center.ravel()[0]))

        # Active boxes are the ones where ei_hi is higher than ei_max
        active_boxes_mask = ei_hi > ei_max  # (n,)
        if not bool(xp.any(active_boxes_mask)):
            idx_max = xp.argmax(ei_hi)
            active_boxes_mask = xp.zeros(n, dtype=bool)
            active_boxes_mask[idx_max] = True

        # Update maximum width of active boxes
        w_max = xp.max(bounds_U[active_boxes_mask] - bounds_L[active_boxes_mask], axis=0)

        # Print status
        if verbose:
            print(
                f" Partition {partition}/{max_partitions-1}, Boxes: {n}, "
                f"Active: {xp.sum(active_boxes_mask)}, Max EI: {ei_max:.6f}, Max Width: {w_max}"
            )

        if logMask:
            log[f"p{partition}"] = {
                "bounds_L": np.asarray(bounds_L),
                "bounds_U": np.asarray(bounds_U),
                "active_boxes_mask": np.asarray(active_boxes_mask),
            }
        
        # Update partition count
        partition += 1

        # Split active boxes (don't if its the last partition)
        if partition < max_partitions and bool(xp.any(w_max > epsilon_X)):
            bounds_L, bounds_U = split_boxes(bounds_L, bounds_U, active_boxes_mask, w, n, d, backend=backend, validation=validation)

    # Check EI in the center of the active boxes and return the best point
    bound_U_active = bounds_U[active_boxes_mask]  # (m,d)
    bound_L_active = bounds_L[active_boxes_mask]  # (m,d)
    center_active = (bound_L_active + bound_U_active) / 2.0  # (m,d)
    mu_active, sigma_active = gp.predict(np.array(center_active), return_std=True)  # (m,) both
    ei_active = expected_improvement(xp.asarray(mu_active), xp.asarray(sigma_active), y_min_unscaled, backend=backend)  # (m,)
    idx_best = xp.argmax(ei_active)
    best_x = center_active[idx_best]

    return ExactBOPartitioningResult(Xn=np.asarray(best_x), logIteration=log if logMask else None), np.asarray(ei_active) #ELIMINATE
    #return ExactBOPartitioningResult(Xn=np.asarray(best_x), logIteration=log if logMask else None)
