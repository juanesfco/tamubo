from __future__ import annotations

import numpy as np

from tamubo.acquisition_functions.ei import log_expected_improvement
from tamubo.utils import BackendName, get_array_module as _array_module, to_numpy


# Define rbf_k_bounds
def rbf_k_bounds(
    bounds_L, 
    bounds_U, 
    xi, 
    n: int, 
    d: int, 
    sigma_f_2: float, 
    length_scale, 
    *,
    backend: BackendName = "auto",
    validation: bool = True,
) -> tuple:
    """
    Compute lower/upper bounds of the RBF kernel between xi and each hyperbox.

    Parameters
    ----------
    bounds_L, bounds_U : np.ndarray or cupy.ndarray
        Lower/upper bounds for n boxes. Accepts shape (n,d),
        with box coordinates stored consecutively by dimension.
    xi : np.ndarray or cupy.ndarray
        Query points in R^d with shape (d,).
    n : int
        Number of boxes.
    d : int
        Dimension of the design space.
    sigma_f_2 : float
        Kernel variance from the trained GP.
    length_scale : float or array-like of shape (d,)
        RBF length scale from the trained GP.
    backend : {"auto", "numpy", "cupy"}, default="auto"
        Backend used for array ops.
    validation: default=True
        Validate dimensions of inputs.

    Returns
    -------
    (K_lo, K_hi) : tuple[np.ndarray or cupy.ndarray, np.ndarray or cupy.ndarray]
        Lower and upper kernel bounds for each box, each with shape (n,).
    """
    # Convert inputs to the appropriate array type based on the backend.
    xp = _array_module(backend)
    bounds_L = xp.asarray(bounds_L)
    bounds_U = xp.asarray(bounds_U)
    xi = xp.asarray(xi)
    if not isinstance(length_scale, (float, int)):
        length_scale = xp.asarray(length_scale)

    # Validate input shapes if requested.
    if validation:
        if bounds_L.shape != (n,d) or bounds_U.shape != (n,d):
            raise ValueError("bounds_L/R must have shape (n,d).")
        if xi.size != d:
            raise ValueError("xi must have size d.")

    # Serial computation for numpy (more efficient for small n).
    if xp is np:
        K_lo = []
        K_hi = []
        # For each box, compute the kernel bounds to xi
        for i in range(n):
            # Extract the bounds for the i-th box
            bounds_L_i = bounds_L[i] # (d,) 
            bounds_U_i = bounds_U[i] # (d,)

            # Scale with lengthscale
            bounds_L_i_scaled = bounds_L_i / length_scale
            bounds_U_i_scaled = bounds_U_i / length_scale
            xi_scaled = xi / length_scale

            # Minimum and maximum distance from xi to the box by dimension
            d_min = xp.maximum(xp.maximum(bounds_L_i_scaled - xi_scaled, xi_scaled - bounds_U_i_scaled), 0) # (d,)
            d_max = xp.maximum(xp.abs(bounds_L_i_scaled - xi_scaled), xp.abs(xi_scaled - bounds_U_i_scaled)) # (d,)

            # Minumum and maximum distance from xi to the box
            D_min = xp.linalg.norm(d_min)
            D_max = xp.linalg.norm(d_max)

            # Compute kernel bounds using the RBF formula
            K_lo.append(sigma_f_2 * xp.exp(-0.5 * D_max ** 2))
            K_hi.append(sigma_f_2 * xp.exp(-0.5 * D_min ** 2))

        return xp.array(K_lo), xp.array(K_hi)
    
    # Vectorized computation for cupy (more efficient for large n).
    else:
        # Create empty buffers for the intermediate distance calculations
        diff_lo = xp.empty((n, d), dtype=xp.float64) # (n,d)
        diff_hi = xp.empty((n, d), dtype=xp.float64) # (n,d)
        d_min = xp.empty((n, d), dtype=xp.float64) # (n,d)
        d_max = xp.empty((n, d), dtype=xp.float64) # (n,d)

        # diff_lo = bounds_L - xi, diff_hi = xi - bounds_U
        xp.subtract(bounds_L, xi, out=diff_lo) 
        xp.subtract(xi, bounds_U, out=diff_hi)

        # Scale diff_lo and diff_hi by lengthscale
        xp.divide(diff_lo, length_scale, out=diff_lo)
        xp.divide(diff_hi, length_scale, out=diff_hi)

        # d_min = max(max(diff_lo, diff_hi), 0)
        xp.maximum(diff_lo, diff_hi, out=d_min)
        xp.maximum(d_min, 0.0, out=d_min)

        # d_max = max(abs(diff_lo), abs(diff_hi))
        xp.abs(diff_lo, out=diff_lo)
        xp.abs(diff_hi, out=diff_hi)
        xp.maximum(diff_lo, diff_hi, out=d_max)

        # We only need squared norms for RBF exponent.
        xp.multiply(d_min, d_min, out=d_min)
        xp.multiply(d_max, d_max, out=d_max)
        
        # Maximum distance means lower kernel value, and vice versa.
        K_lo = xp.sum(d_max, axis=1) # (n,)
        K_hi = xp.sum(d_min, axis=1) # (n,)

        # Compute kernel bounds using the RBF formula
        coef = -0.5
        xp.multiply(K_lo, coef, out=K_lo)
        xp.multiply(K_hi, coef, out=K_hi)
        xp.exp(K_lo, out=K_lo)
        xp.exp(K_hi, out=K_hi)
        xp.multiply(K_lo, sigma_f_2, out=K_lo)
        xp.multiply(K_hi, sigma_f_2, out=K_hi)

        return (K_lo, K_hi)


# Define mu_bounds
def mu_bounds(
    alpha, 
    K_lo, 
    K_hi, 
    n: int, 
    N: int,
    *,
    y_train_mean: float = 0.0, 
    y_train_std: float = 1.0,
    scaled_output: bool = False,
    backend: BackendName = "auto", 
    validation: bool = True,
) -> tuple:
    """
    Compute lower/upper bounds on the GP posterior mean per box.
    Using: μ(x)=k(x)^T α, α = L^{-T} \\ (L \\ y).

    Parameters
    ----------
    alpha : np.ndarray or cupy.ndarray
        GP dual coefficients (typically gp.alpha_), shape (N,).
    K_lo, K_hi : np.ndarray or cupy.ndarray
        Lower/upper kernel bounds between each box and each training point,
        each with shape (n, N).
    n : int
        Number of boxes.
    N : int
        Number of training points.
    y_train_mean : float, default=0.0
        Training target mean used by the GP (normalize_y=True).
    y_train_std : float, default=1.0
        Training target std used by the GP (normalize_y=True).
    scaled_output : bool, default=False
        If True, return bounds in the GP's standardized target space.
        If False, return bounds in the original target scale.
    backend : {"auto", "numpy", "cupy"}, default="auto"
        Backend used for array ops.
    validation : bool, default=True
        If True, validate shapes and sizes.

    Returns
    -------
    (mu_lo, mu_hi) : tuple[np.ndarray or cupy.ndarray, np.ndarray or cupy.ndarray]
        Lower and upper bounds on the mean for each box, each with shape (n,).
    """
    # Convert inputs to the appropriate array type based on the backend.
    xp = _array_module(backend)
    alpha = xp.asarray(alpha)
    K_lo = xp.asarray(K_lo)
    K_hi = xp.asarray(K_hi)

    # Validate input shapes if requested.
    if validation:
        if alpha.size != N:
            raise ValueError("alpha must have size N.")
        if K_lo.shape != (n, N) or K_hi.shape != (n, N):
            raise ValueError("K_lo and K_hi must have shape (n, N).")

    # Serial computation for numpy (more efficient for small n).   
    if xp is np:
        mu_lo = []
        mu_hi = []
        # For each box, compute the lower and upper mean bounds
        for i in range(n):
            K_lo_i = K_lo[i]  # shape (N,)
            K_hi_i = K_hi[i]  # shape (N,)

            mu_lo_i = 0.0
            mu_hi_i = 0.0
            # For each training point, determine contribution to lower and upper bounds based on the sign of alpha[j].
            for j in range(N):
                # If alpha[j] >= 0, the lower bound contribution comes from K_lo and upper from K_hi.
                if alpha[j] >= 0:
                    mu_lo_i += K_lo_i[j] * alpha[j]
                    mu_hi_i += K_hi_i[j] * alpha[j]
                # If alpha[j] < 0, the lower bound contribution comes from K_hi and upper from K_lo.
                else:
                    mu_lo_i += K_hi_i[j] * alpha[j]
                    mu_hi_i += K_lo_i[j] * alpha[j]

            if scaled_output:
                mu_lo.append(mu_lo_i)
                mu_hi.append(mu_hi_i)
            else:
                mu_lo.append(y_train_mean + y_train_std * mu_lo_i)
                mu_hi.append(y_train_mean + y_train_std * mu_hi_i)
        
        return xp.array(mu_lo), xp.array(mu_hi)
    
    # Vectorized computation for cupy (more efficient for large n).
    else:
        # Split alpha into positive/negative parts to avoid (n, N) intermediates.
        alpha_pos = xp.empty_like(alpha)
        alpha_neg = xp.empty_like(alpha)
        xp.maximum(alpha, 0.0, out=alpha_pos)
        xp.minimum(alpha, 0.0, out=alpha_neg)

        # mu_lo = K_lo @ alpha_pos + K_hi @ alpha_neg
        mu_lo = K_lo @ alpha_pos
        tmp = K_hi @ alpha_neg
        xp.add(mu_lo, tmp, out=mu_lo)

        # mu_hi = K_hi @ alpha_pos + K_lo @ alpha_neg
        mu_hi = K_hi @ alpha_pos
        tmp = K_lo @ alpha_neg
        xp.add(mu_hi, tmp, out=mu_hi)

        if not scaled_output:
            xp.multiply(mu_lo, y_train_std, out=mu_lo)
            xp.add(mu_lo, y_train_mean, out=mu_lo)
            xp.multiply(mu_hi, y_train_std, out=mu_hi)
            xp.add(mu_hi, y_train_mean, out=mu_hi)

        return (mu_lo, mu_hi)




# Define sigma_bounds
def sigma_bounds(
    K_lo,
    K_hi,
    L,
    n: int,
    N: int,
    sigma_f_2: float,
    *,
    y_train_std: float = 1.0,
    scaled_output: bool = False,
    backend: BackendName = "auto",
    validation: bool = True,
    L_inv=None,
    lambda_max: float | None = None,
    q_bounds: tuple | None = None,
) -> tuple:
    """
    Compute lower/upper bounds on the GP (latent) posterior std per box.
    σ^2 = σ_f^2 - Q with Q = ||v||^2, v = L^{-1} k, L = cholesky(K + σ_n^2 I),
    and each kernel entry k_i in [K_lo_i, K_hi_i] (K_lo >= 0).

    Q is bounded two ways and the tighter bound is kept:
      (A) v = L^{-1} k is linear in k, so each v_j is bounded exactly over the
          K box by splitting the rows of L^{-1} by sign:
            v_lo = L^{-1}_+ K_lo + L^{-1}_- K_hi,  v_hi = L^{-1}_+ K_hi + L^{-1}_- K_lo,
          then Q_lo = Σ_j min v_j^2 (0 where [v_lo, v_hi] contains 0), Q_hi = Σ_j max v_j^2.
          Unlike interval forward substitution, no interval is reused across rows,
          so the v bounds do not widen with j. Cost: two (n, N) x (N, N) GEMMs.
      (B) Q = k^T (K + σ_n^2 I)^{-1} k >= ||k||^2 / λ_max(K + σ_n^2 I) >= ||K_lo||^2 / λ_max.

    Parameters
    ----------
    K_lo, K_hi : np.ndarray or cupy.ndarray
        Kernel bounds per box vs training points, shape (n, N). Not modified.
    L : np.ndarray or cupy.ndarray
        Cholesky factor (N, N) of K + σ_n^2 I (lower triangular).
    n : int
        Number of boxes.
    N : int
        Number of training points.
    sigma_f_2 : float
        Kernel variance σ_f^2.
    y_train_std : float, default=1.0
        Training target std used by the GP (normalize_y=True).
    scaled_output : bool, default=False
        If True, return bounds in the GP's standardized target space.
        If False, return bounds in the original target scale.
    backend : {"auto", "numpy", "cupy"}, default="auto"
        Backend used for array ops.
    validation : bool, default=True
        If True, validate shapes and sizes.
    L_inv : array, shape (N, N), optional
        Precomputed L^{-1}; computed from L when None.
    lambda_max : float, optional
        Precomputed largest eigenvalue of L L^T; computed from L when None.
    q_bounds : tuple (q_lo, q_hi) of arrays with shape (n,), optional
        Additional valid bounds on Q (e.g. from AutoBound) intersected with (A)/(B).

    Returns
    -------
    (sig_lo, sig_hi) : tuple[np.ndarray or cupy.ndarray, np.ndarray or cupy.ndarray]
        Lower/upper sigma bounds per box, each shape (n,).
    """
    # Convert inputs to the appropriate array type based on the backend.
    xp = _array_module(backend)
    K_lo = xp.asarray(K_lo)
    K_hi = xp.asarray(K_hi)
    L = xp.asarray(L)

    # Validate input shapes if requested.
    if validation:
        if K_lo.shape != K_hi.shape:
            raise ValueError("K_lo and K_hi must have the same shape.")
        if L.shape[0] != L.shape[1]:
            raise ValueError("L must be square.")
        if K_lo.shape[1] != L.shape[0]:
            raise ValueError("K_lo/K_hi second dim must match L size.")

    if L_inv is None or lambda_max is None:
        L_inv_L, lambda_max_L = sigma_bound_factors(L)
        L_inv = L_inv_L if L_inv is None else L_inv
        lambda_max = lambda_max_L if lambda_max is None else lambda_max
    L_inv = xp.asarray(L_inv, dtype=xp.float64)

    q_lo, q_hi = _q_bounds(K_lo, K_hi, L_inv, float(lambda_max), xp)
    if q_bounds is not None:
        xp.maximum(q_lo, q_bounds[0], out=q_lo)
        xp.minimum(q_hi, q_bounds[1], out=q_hi)

    # var = sigma_f_2 - Q, ensuring non-negativity; sigma = sqrt(var)
    sig_lo = q_hi  # Q_hi -> sig_lo, reusing the buffer
    sig_hi = q_lo  # Q_lo -> sig_hi
    xp.subtract(sigma_f_2, sig_lo, out=sig_lo)
    xp.subtract(sigma_f_2, sig_hi, out=sig_hi)
    xp.maximum(sig_lo, 1e-12, out=sig_lo)
    xp.maximum(sig_hi, 1e-12, out=sig_hi)
    xp.sqrt(sig_lo, out=sig_lo)
    xp.sqrt(sig_hi, out=sig_hi)
    if not scaled_output:
        xp.multiply(sig_lo, y_train_std, out=sig_lo)
        xp.multiply(sig_hi, y_train_std, out=sig_hi)

    return sig_lo, sig_hi

def sigma_bound_factors(L) -> tuple:
    """
    Return (L^{-1}, λ_max(L L^T)) on the host, the GP-only factors used by
    ``sigma_bounds``; compute once per trained GP and pass them in.
    """
    L = np.asarray(to_numpy(L), dtype=np.float64)
    from scipy.linalg import solve_triangular

    L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True)
    lambda_max = float(np.linalg.eigvalsh(L @ L.T)[-1])
    return L_inv, lambda_max

def _q_bounds(K_lo, K_hi, L_inv, lambda_max, xp):
    """Bounds (A) and (B) on Q = ||L^{-1} k||^2 described in ``sigma_bounds``."""
    L_inv_pos = xp.maximum(L_inv, 0.0).T  # (N, N), transposed for K @ L_inv^T
    L_inv_neg = xp.minimum(L_inv, 0.0).T

    # (A) v bounds, shape (n, N)
    v_lo = K_lo @ L_inv_pos
    v_lo += K_hi @ L_inv_neg
    v_hi = K_hi @ L_inv_pos
    v_hi += K_lo @ L_inv_neg

    # Q_lo = Σ min over [v_lo, v_hi] of v^2 = Σ (max(v_lo, 0)^2 + min(v_hi, 0)^2)
    # (at most one of the two terms is nonzero per entry).
    q_hi = xp.maximum(xp.abs(v_lo), xp.abs(v_hi))
    xp.multiply(q_hi, q_hi, out=q_hi)
    q_hi = xp.sum(q_hi, axis=1)
    xp.maximum(v_lo, 0.0, out=v_lo)
    xp.minimum(v_hi, 0.0, out=v_hi)
    xp.multiply(v_lo, v_lo, out=v_lo)
    xp.multiply(v_hi, v_hi, out=v_hi)
    v_lo += v_hi
    q_lo = xp.sum(v_lo, axis=1)
    del v_lo, v_hi

    # (B) Q >= ||K_lo||^2 / λ_max
    q_lo_B = xp.sum(K_lo * K_lo, axis=1)
    xp.divide(q_lo_B, lambda_max, out=q_lo_B)
    xp.maximum(q_lo, q_lo_B, out=q_lo)
    return q_lo, q_hi


def ei_bounds(
    mu_lo,
    mu_hi,
    sig_lo,
    sig_hi,
    n: int,
    y_min: float,
    *,
    backend: BackendName = "auto",
    validation: bool = True,
    pad: float = 1e-12,
    log: bool = False,
) -> tuple:
    """
    Exact EI (or log EI) range over the rectangle [mu_lo, mu_hi] x [sig_lo, sig_hi]:
      EI(mu, sigma) = sigma * h((f_min - mu) / sigma),  h(z) = phi(z) + z Phi(z),
    is decreasing in mu (dEI/dmu = -Phi(z)) and increasing in sigma
    (dEI/dsigma = phi(z)), so
      EI_lo = EI(mu_hi, sig_lo),  EI_hi = EI(mu_lo, sig_hi).
    This is the tightest bound given the mu/sigma intervals (interval arithmetic
    through Z, Phi and phi separately overestimates it). Evaluated through the
    stable log_h, so there is no cancellation or underflow for z << 0.

    Parameters
    ----------
    mu_lo, mu_hi : np.ndarray or cupy.ndarray
        Mean bounds per box, shape (n,).
    sig_lo, sig_hi : np.ndarray or cupy.ndarray
        Sigma bounds per box, shape (n,), sig_lo >= 0.
    n : int
        Number of boxes.
    y_min : float
        Minimum of training targets in the same scale as ``mu_*`` and ``sig_*``.
    backend : {"auto", "numpy", "cupy"}, default="auto"
        Backend used for array ops.
    validation : bool, optional
        If True, validate shapes/sizes.
    pad : float, optional, default=1e-12
        Sigma values below ``pad`` are treated as zero (EI -> max(f_min - mu, 0)).
    log : bool, default=False
        Return bounds on log EI instead of EI (-inf where EI is exactly 0).

    Returns
    -------
    (ei_lo, ei_hi) : tuple[np.ndarray or cupy.ndarray, np.ndarray or cupy.ndarray]
        EI (or log EI) bounds per box, shape (n,).
    """
    # Convert inputs to the appropriate array type based on the backend.
    xp = _array_module(backend)
    mu_lo = xp.asarray(mu_lo, dtype=xp.float64)
    mu_hi = xp.asarray(mu_hi, dtype=xp.float64)
    sig_lo = xp.asarray(sig_lo, dtype=xp.float64)
    sig_hi = xp.asarray(sig_hi, dtype=xp.float64)

    # Validate input shapes if requested.
    if validation:
        if mu_lo.shape != mu_hi.shape:
            raise ValueError("mu_lo and mu_hi must have the same shape.")
        if sig_lo.shape != sig_hi.shape:
            raise ValueError("sig_lo and sig_hi must have the same shape.")
        if mu_lo.shape != sig_lo.shape:
            raise ValueError("mu and sigma bounds must have the same shape.")

    ei_lo = _log_ei_corner(mu_hi, sig_lo, y_min, pad, xp, backend)
    ei_hi = _log_ei_corner(mu_lo, sig_hi, y_min, pad, xp, backend)
    if not log:
        xp.exp(ei_lo, out=ei_lo)
        xp.exp(ei_hi, out=ei_hi)
    return ei_lo, ei_hi

def _log_ei_corner(mu, sig, y_min, pad, xp, backend):
    """log EI(mu, sig) elementwise, with the sigma -> 0 limit log(max(y_min - mu, 0))."""
    tiny = sig < pad
    sig_safe = xp.where(tiny, 1.0, sig)
    out = log_expected_improvement(mu, sig_safe, y_min, backend=backend)
    if bool(xp.any(tiny)):
        improvement = xp.maximum(y_min - mu, 0.0)
        with np.errstate(divide="ignore"):
            out = xp.where(tiny, xp.log(improvement), out)
    return out
