from __future__ import annotations

from importlib import import_module
from typing import Any
import warnings

import numpy as np

from tamubo.utils import BackendName, resolve_backend

__all__ = ["gp_posterior"]


def _array_module(backend: BackendName = "auto"):
    """Return the resolved array module (`numpy` or `cupynumeric`)."""
    backend_info = resolve_backend(backend)
    if backend_info.selected == "numpy":
        return np
    return import_module("cupynumeric")


def _to_python_bool(value: Any) -> bool:
    """Convert numpy/cupynumeric scalar booleans into plain Python bool."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return bool(np.asarray(value).item())


def _rbf_kernel(X, Y, length_scale, sigma_f_squared, xp):
    """
    Compute an RBF covariance matrix:
        K[i, j] = sigma_f_squared * exp(-0.5 * ||(X_i - Y_j)/length_scale||^2)
    """
    X_scaled = X / length_scale
    Y_scaled = Y / length_scale

    X_sq = xp.sum(X_scaled * X_scaled, axis=1, keepdims=True)
    Y_sq = xp.sum(Y_scaled * Y_scaled, axis=1, keepdims=True).T

    sq_dists = X_sq + Y_sq - 2.0 * (X_scaled @ Y_scaled.T)
    sq_dists = xp.maximum(sq_dists, 0.0)

    return sigma_f_squared * xp.exp(-0.5 * sq_dists)


def gp_posterior(
    X,
    *,
    X_train,
    alpha,
    L,
    length_scale,
    sigma_f_squared: float,
    sigma_n_squared: float = 0.0,
    y_train_mean=0.0,
    y_train_std=1.0,
    return_std: bool = False,
    return_cov: bool = False,
    include_noise: bool = True,
    backend: BackendName = "auto",
    validation: bool = True,
):
    """
    Compute GP posterior moments from trained-state tensors/arrays.

    This mirrors sklearn's ``GaussianProcessRegressor.predict`` behavior for
    an RBF + WhiteKernel model using precomputed training-state quantities.

    Parameters
    ----------
    X : array-like of shape (n_test, d) or (d,)
        Query points.
    X_train : array-like of shape (n_train, d)
        Training points used to fit the GP.
    alpha : array-like of shape (n_train,) or (n_train, n_targets)
        Dual coefficients from a trained GP (e.g., ``gp.alpha_``).
    L : array-like of shape (n_train, n_train)
        Cholesky factor of the training kernel matrix (e.g., ``gp.L_``).
    length_scale : float or array-like of shape (d,)
        RBF length scale.
    sigma_f_squared : float
        Signal variance.
    sigma_n_squared : float, default=0.0
        White-noise variance.
    y_train_mean : float or array-like, default=0.0
        Target normalization mean from training.
    y_train_std : float or array-like, default=1.0
        Target normalization std from training.
    return_std : bool, default=False
        Return posterior standard deviation.
    return_cov : bool, default=False
        Return posterior covariance matrix.
    include_noise : bool, default=True
        If True, include white-noise variance in predictive variance/covariance.
    backend : {"auto", "numpy", "cupynumeric"}, default="auto"
        Array backend.
    validation : bool, default=True
        Validate input dimensions and consistency.
    """
    if return_std and return_cov:
        raise ValueError("At most one of return_std or return_cov can be True.")

    xp = _array_module(backend)

    X = xp.asarray(X, dtype=xp.float64)
    X_train = xp.asarray(X_train, dtype=xp.float64)
    alpha = xp.asarray(alpha, dtype=xp.float64)
    L = xp.asarray(L, dtype=xp.float64)
    sigma_f_squared = float(np.asarray(sigma_f_squared, dtype=np.float64).item())
    sigma_n_squared = float(np.asarray(sigma_n_squared, dtype=np.float64).item())
    length_scale = xp.asarray(length_scale, dtype=xp.float64)
    y_train_mean = xp.asarray(y_train_mean, dtype=xp.float64)
    y_train_std = xp.asarray(y_train_std, dtype=xp.float64)

    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    if alpha.ndim == 1:
        alpha = alpha.reshape(-1, 1)
        single_target = True
    elif alpha.ndim == 2:
        single_target = alpha.shape[1] == 1
    else:
        raise ValueError("alpha must be 1D or 2D.")

    if validation:
        if X.ndim != 2 or X_train.ndim != 2:
            raise ValueError("X and X_train must be 2D after normalization.")
        if X.shape[1] != X_train.shape[1]:
            raise ValueError(
                f"X and X_train must have matching feature dimension; got {X.shape[1]} and {X_train.shape[1]}."
            )
        if alpha.shape[0] != X_train.shape[0]:
            raise ValueError(
                f"alpha first dimension must match n_train; got {alpha.shape[0]} and {X_train.shape[0]}."
            )
        if L.ndim != 2 or L.shape[0] != L.shape[1]:
            raise ValueError("L must be a square 2D array.")
        if L.shape[0] != X_train.shape[0]:
            raise ValueError(
                f"L size must match n_train; got {L.shape} and n_train={X_train.shape[0]}."
            )
        if length_scale.ndim > 1:
            raise ValueError("length_scale must be scalar or 1D.")
        if length_scale.ndim == 1 and length_scale.shape[0] != X_train.shape[1]:
            raise ValueError(
                f"length_scale size must match feature dimension; got {length_scale.shape[0]} and {X_train.shape[1]}."
            )

    K_trans = _rbf_kernel(X, X_train, length_scale, sigma_f_squared, xp)

    y_mean = K_trans @ alpha
    y_mean = y_mean * y_train_std + y_train_mean

    if single_target:
        y_mean = y_mean.ravel()

    if not return_std and not return_cov:
        return y_mean

    V = xp.linalg.solve(L, K_trans.T)

    if return_cov:
        K_xx = _rbf_kernel(X, X, length_scale, sigma_f_squared, xp)
        if include_noise and sigma_n_squared != 0.0:
            diag_idx = xp.arange(X.shape[0])
            K_xx[diag_idx, diag_idx] += sigma_n_squared

        y_cov = K_xx - (V.T @ V)
        y_cov = 0.5 * (y_cov + y_cov.T)

        diag = xp.diag(y_cov)
        neg_diag = diag < 0.0
        if _to_python_bool(xp.any(neg_diag)):
            warnings.warn(
                "Predicted covariance had small negative diagonal entries; clipping to zero.",
                RuntimeWarning,
            )
            y_cov = y_cov.copy()
            diag_idx = xp.arange(y_cov.shape[0])
            y_cov[diag_idx, diag_idx] = xp.maximum(diag, 0.0)

        y_scale_sq = y_train_std * y_train_std
        if getattr(y_scale_sq, "ndim", 0) == 0:
            y_cov = y_cov * y_scale_sq
            return y_mean, y_cov

        y_cov = y_cov[:, :, None] * y_scale_sq
        return y_mean, y_cov

    base_var = sigma_f_squared + (sigma_n_squared if include_noise else 0.0)
    y_var = xp.full((X.shape[0],), base_var, dtype=xp.float64) - xp.sum(V * V, axis=0)

    neg_var = y_var < 0.0
    if _to_python_bool(xp.any(neg_var)):
        warnings.warn(
            "Predicted variances had small negative values; clipping to zero.",
            RuntimeWarning,
        )
        y_var = xp.maximum(y_var, 0.0)

    y_scale_sq = y_train_std * y_train_std
    if getattr(y_scale_sq, "ndim", 0) == 0:
        y_var = y_var * y_scale_sq
        y_std = xp.sqrt(y_var)
        return y_mean, y_std

    y_std = xp.sqrt(y_var[:, None] * y_scale_sq)
    return y_mean, y_std
