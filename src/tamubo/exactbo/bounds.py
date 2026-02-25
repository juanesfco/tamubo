from __future__ import annotations

from importlib import import_module

import numpy as np
import math

from tamubo.utils import BackendName, resolve_backend

# Constants reused across helper calls.
_SQRT2 = math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)

def _array_module(backend: BackendName = "auto"):
    """Return the resolved array module (`numpy` or `cupynumeric`)."""
    backend_info = resolve_backend(backend)
    if backend_info.selected == "numpy":
        return np
    # Import cupynumeric only when it is the selected backend.
    return import_module("cupynumeric")


# Define rbf_k_bounds
def rbf_k_bounds(
    bounds_L, 
    bounds_U, 
    xi, 
    n: int, 
    d: int, 
    sigma_f_2: float, 
    length_scale: float, 
    *,
    backend: BackendName = "auto",
    validation: bool = True,
) -> tuple:
    """
    Compute lower/upper bounds of the RBF kernel between xi and each hyperbox.
    ADD CAPABILITY FOR ANISOTROPIC LENGTH SCALES IN THE FUTURE.

    Parameters
    ----------
    bounds_L, bounds_U : np.ndarray or cupynumeric.ndarray
        Lower/upper bounds for n boxes. Accepts shape (n,d),
        with box coordinates stored consecutively by dimension.
    xi : np.ndarray or cupynumeric.ndarray
        Query points in R^d with shape (d,).
    n : int
        Number of boxes.
    d : int
        Dimension of the design space.
    sigma_f_2 : float
        Kernel variance from the trained GP.
    length_scale : float
        RBF length scale from the trained GP.
    backend : {"auto", "numpy", "cupynumeric"}, default="auto"
        Backend used for array ops.
    validation: default=True
        Validate dimensions of inputs.

    Returns
    -------
    (K_lo, K_hi) : tuple[np.ndarray or cupynumeric.ndarray, np.ndarray or cupynumeric.ndarray]
        Lower and upper kernel bounds for each box, each with shape (n,).
    """
    # Convert inputs to the appropriate array type based on the backend.
    xp = _array_module(backend)
    bounds_L = xp.asarray(bounds_L)
    bounds_U = xp.asarray(bounds_U)
    xi = xp.asarray(xi)

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

            # Minimum and maximum distance from xi to the box by dimension
            d_min = xp.maximum(xp.maximum(bounds_L_i - xi, xi - bounds_U_i), 0) # (d,)
            d_max = xp.maximum(xp.abs(bounds_L_i - xi), xp.abs(xi - bounds_U_i)) # (d,)

            # Minumum and maximum distance from xi to the box
            D_min = xp.linalg.norm(d_min)
            D_max = xp.linalg.norm(d_max)

            # Compute kernel bounds using the RBF formula
            K_lo.append(sigma_f_2 * xp.exp(-1 / (2 * length_scale ** 2) * D_max ** 2))
            K_hi.append(sigma_f_2 * xp.exp(-1 / (2 * length_scale ** 2) * D_min ** 2))

        return xp.array(K_lo), xp.array(K_hi)
    
    # Vectorized computation for cupynumeric (more efficient for large n).
    else:
        # Create empty buffers for the intermediate distance calculations
        diff_lo = xp.empty((n, d), dtype=xp.float64) # (n,d)
        diff_hi = xp.empty((n, d), dtype=xp.float64) # (n,d)
        d_min = xp.empty((n, d), dtype=xp.float64) # (n,d)
        d_max = xp.empty((n, d), dtype=xp.float64) # (n,d)

        # diff_lo = bounds_L - xi, diff_hi = xi - bounds_U
        xp.subtract(bounds_L, xi, out=diff_lo) 
        xp.subtract(xi, bounds_U, out=diff_hi)

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
        coef = -0.5 / (length_scale ** 2)
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
    backend: BackendName = "auto", 
    validation: bool = True,
) -> tuple:
    """
    Compute lower/upper bounds on the GP posterior mean per box, in original scale.
    Using: μ(x)=k(x)^T α, α = L^{-T} \\ (L \\ y).

    Parameters
    ----------
    alpha : np.ndarray or cupynumeric.ndarray
        GP dual coefficients (typically gp.alpha_), shape (N,).
    K_lo, K_hi : np.ndarray or cupynumeric.ndarray
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
    backend : {"auto", "numpy", "cupynumeric"}, default="auto"
        Backend used for array ops.
    validation : bool, default=True
        If True, validate shapes and sizes.

    Returns
    -------
    (mu_lo, mu_hi) : tuple[np.ndarray or cupynumeric.ndarray, np.ndarray or cupynumeric.ndarray]
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

            # unnormalize to original target scale
            mu_lo.append(y_train_mean + y_train_std * mu_lo_i)
            mu_hi.append(y_train_mean + y_train_std * mu_hi_i)
        
        return xp.array(mu_lo), xp.array(mu_hi)
    
    # Vectorized computation for cupynumeric (more efficient for large n).
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

        # Rescale and shift back to original y space.
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
    backend: BackendName = "auto", 
    validation: bool = True,
) -> tuple:
    """
    Compute lower/upper bounds on the GP posterior std per box, in original scale.
    Given L = cholesky(K + σ_n^2 I) and per-component kernel intervals K_lo, K_hi (nonnegative), 
    bound v for the solution of L v = k, and then bound Q = ||v||^2, then σ^2 = σ_f^2 - Q, 
    and finally bound σ = sqrt(σ^2).
    Recurrence:
      v_1 = k_1 / L_11
      v_j = (k_j - Σ_{i<j} L_{j,i} v_i) / L_{j,j}

    Parameters
    ----------
    K_lo, K_hi : np.ndarray or cupynumeric.ndarray
        Kernel bounds per box vs training points, shape (n, N).
    L : np.ndarray or cupynumeric.ndarray
        Cholesky factor (N, N) of K + σ_n^2 I (lower triangular).
    n : int
        Number of boxes.
    N : int
        Number of training points.
    sigma_f_2 : float
        Kernel variance σ_f^2.
    y_train_std : float, default=1.0
        Training target std used by the GP (normalize_y=True).
    backend : {"auto", "numpy", "cupynumeric"}, default="auto"
        Backend used for array ops.
    validation : bool, default=True
        If True, validate shapes and sizes.

    Returns
    -------
    (sig_lo, sig_hi) : tuple[np.ndarray or cupynumeric.ndarray, np.ndarray or cupynumeric.ndarray]
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

    # Check if using numpy or cupynumeric for the computation.
    if xp is np:
        # Serial computation for numpy (more efficient for small n).
        return _sigma_bounds_numpy(K_lo, K_hi, L, n, N, sigma_f_2, y_train_std)
    else:
        # Vectorized computation for cupynumeric (more efficient for large n).
        return _sigma_bounds_cupynumeric(K_lo, K_hi, L, n, N, sigma_f_2, y_train_std)
    
def _sigma_bounds_numpy(K_lo, K_hi, L, n, N, sigma_f_2, y_train_std):
    sig_lo = []
    sig_hi = []
    # For each box, compute the lower and upper sigma bounds
    for i in range(n):
        K_lo_i = K_lo[i]  # shape (N,)
        K_hi_i = K_hi[i]  # shape (N,)

        # Forward solve for v bounds: L v = k, where k is in [K_lo_i, K_hi_i].
        v_lo = np.zeros(N)
        v_hi = np.zeros(N)
        for j in range(N):
            S_lo = 0
            S_hi = 0
            # S_k = Σ_{k<j} L_{j,k} v_k, with each term interval-bounded
            for k in range(j):
                Ljk = L[j, k]
                Ljkv_lo_k = Ljk * v_lo[k]
                Ljkv_hi_k = Ljk * v_hi[k]
                # Depending on the signs of Ljk and v_k, the contribution to S_lo and S_hi can swap.
                Sk_lo = min(Ljkv_lo_k, Ljkv_hi_k)
                Sk_hi = max(Ljkv_lo_k, Ljkv_hi_k)
                S_lo += Sk_lo
                S_hi += Sk_hi

            # N_j = k_j - S_j, where k_j is in [K_lo_i[j], K_hi_i[j]]
            N_lo = K_lo_i[j] - S_hi
            N_hi = K_hi_i[j] - S_lo

            # v_j = N_j/L_{jj}
            Ljj = L[j, j]
            v_lo[j] = N_lo / Ljj
            v_hi[j] = N_hi / Ljj

        # Q = v^T v, with each term interval-bounded
        Q_lo = 0
        Q_hi = 0
        for j in range(N):
            v2_lo = v_lo[j] * v_lo[j]
            v2_hi = v_hi[j] * v_hi[j]
            # If v_j_lo < 0 < v_j_hi, then the minimum of v_j^2 is 0 and the maximum is max(v2_lo, v2_hi).
            if v_lo[j] < 0 and v_hi[j] > 0:
                Q_lo += 0
                Q_hi += max(v2_lo, v2_hi)
            else:
                Q_lo += min(v2_lo, v2_hi)
                Q_hi += max(v2_lo, v2_hi)

        # var = sigma_f_2 - Q, ensuring non-negativity
        var_lo = max(0, sigma_f_2 - Q_hi)
        var_hi = max(0, sigma_f_2 - Q_lo)

        # sig = sqrt(var)
        sig_lo_i = np.sqrt(var_lo)
        sig_hi_i = np.sqrt(var_hi)

        # Ensure non-negativity and unnormalize to original target scale
        sig_lo.append(y_train_std * sig_lo_i)
        sig_hi.append(y_train_std * sig_hi_i)

    return np.array(sig_lo), np.array(sig_hi)

def _sigma_bounds_cupynumeric(K_lo, K_hi, L, n, N, sigma_f_2, y_train_std):
    import cupynumeric as cp
    # Create buffers for intermediate computations
    v_lo = cp.empty((n, N), dtype=cp.float64)
    v_hi = cp.empty((n, N), dtype=cp.float64)

    sig_lo = cp.zeros(n, dtype=cp.float64) # Q_hi accumulator, initialized to 0
    sig_hi = cp.zeros(n, dtype=cp.float64) # Q_lo accumulator, initialized to 0

    S_lo = cp.empty(n, dtype=cp.float64)
    S_hi = cp.empty(n, dtype=cp.float64)
    tmp0 = cp.empty(n, dtype=cp.float64)
    tmp1 = cp.empty(n, dtype=cp.float64)
    mask_cross = cp.empty(n, dtype=bool)
    mask_pos = cp.empty(n, dtype=bool)

    # Forward substitution to compute v_lo and v_hi
    for j in range(N):
        # Initialize the sum S for the j-th column of L.
        S_lo[...] = 0.0
        S_hi[...] = 0.0
        # S_j = sum_{i=0}^{j-1} L[j, i] * v[:, i]
        for i in range(j):
            Lji = float(L[j, i])
            # The sign of Lji determines whether to use v_lo or v_hi for the bounds of S.
            if Lji >= 0.0:
                # S_lo += Lji * v_lo[:, i]
                cp.multiply(v_lo[:, i], Lji, out=tmp0)
                cp.add(S_lo, tmp0, out=S_lo)
                # S_hi += Lji * v_hi[:, i]
                cp.multiply(v_hi[:, i], Lji, out=tmp0)
                cp.add(S_hi, tmp0, out=S_hi)
            else:
                # S_lo += Lji * v_hi[:, i]
                cp.multiply(v_hi[:, i], Lji, out=tmp0)
                cp.add(S_lo, tmp0, out=S_lo)
                # S_hi += Lji * v_lo[:, i]
                cp.multiply(v_lo[:, i], Lji, out=tmp0)
                cp.add(S_hi, tmp0, out=S_hi)

        # N_j = K_j - S_j
        cp.subtract(K_lo[:, j], S_hi, out=tmp0)
        cp.subtract(K_hi[:, j], S_lo, out=tmp1)
        # v_j = N_j / L[j, j]
        Ljj = float(L[j, j])
        cp.divide(tmp0, Ljj, out=v_lo[:, j])
        cp.divide(tmp1, Ljj, out=v_hi[:, j])

        # Q_hi += max(v_lo^2, v_hi^2)
        cp.multiply(v_lo[:, j], v_lo[:, j], out=tmp0)
        cp.multiply(v_hi[:, j], v_hi[:, j], out=tmp1)
        cp.minimum(tmp0, tmp1, out=S_lo)  # staged for Q_lo
        cp.maximum(tmp0, tmp1, out=tmp0)
        cp.add(sig_lo, tmp0, out=sig_lo) # accumulate Q_hi

        # Q_lo += min(v_lo^2, v_hi^2), except 0 when interval crosses zero.
        cp.less(v_lo[:, j], 0.0, out=mask_cross)
        cp.greater(v_hi[:, j], 0.0, out=mask_pos)
        cp.logical_and(mask_cross, mask_pos, out=mask_cross)
        S_lo[mask_cross] = 0.0
        cp.add(sig_hi, S_lo, out=sig_hi) # accumulate Q_lo

    # var = sigma_f_2 - Q
    cp.subtract(sigma_f_2, sig_lo, out=sig_lo)
    cp.subtract(sigma_f_2, sig_hi, out=sig_hi)
    # var > 0
    cp.maximum(sig_lo, 0.0, out=sig_lo)
    cp.maximum(sig_hi, 0.0, out=sig_hi)
    # sigma = sqrt(var)
    cp.sqrt(sig_lo, out=sig_lo)
    cp.sqrt(sig_hi, out=sig_hi)
    # Scale by y_train_std to get unscaled sigma bounds.
    cp.multiply(sig_lo, y_train_std, out=sig_lo)
    cp.multiply(sig_hi, y_train_std, out=sig_hi)

    return sig_lo, sig_hi


def ei_bounds(
    mu_lo, 
    mu_hi, 
    sig_lo, 
    sig_hi,
    n: int, 
    y_min_unscaled: float, 
    *,
    backend: BackendName = "auto", 
    validation: bool = True,
    pad: float = 1e-5,
) -> tuple:
    """
    IA bounds for EI across all boxes:
      EI = N * Phi(Z) + sigma * phi(Z), where N = f_min - mu, Z = N / sigma.

    Parameters
    ----------
    mu_lo, mu_hi : np.ndarray or cupynumeric.ndarray
        Mean bounds per box, shape (n,).
    sig_lo, sig_hi : np.ndarray or cupynumeric.ndarray
        Sigma bounds per box, shape (n,), sig_lo >= 0.
    n : int
        Number of boxes.
    y_min_unscaled : float
        Minimum of training targets.
    backend : {"auto", "numpy", "cupynumeric"}, default="auto"
        Backend used for array ops.
    validation : bool, optional
        If True, validate shapes/sizes.
    pad : float, optional, default=1e-5
        Small positive number to avoid division by zero.

    Returns
    -------
    (ei_lo, ei_hi) : tuple[np.ndarray or cupynumeric.ndarray, np.ndarray or cupynumeric.ndarray]
        EI bounds per box, shape (n,).
    """
    # Convert inputs to the appropriate array type based on the backend.
    xp = _array_module(backend)
    mu_lo = xp.asarray(mu_lo)
    mu_hi = xp.asarray(mu_hi)
    sig_lo = xp.asarray(sig_lo)
    sig_hi = xp.asarray(sig_hi)

    # Validate input shapes if requested.
    if validation:
        if mu_lo.shape != mu_hi.shape:
            raise ValueError("mu_lo and mu_hi must have the same shape.")
        if sig_lo.shape != sig_hi.shape:
            raise ValueError("sig_lo and sig_hi must have the same shape.")
        if mu_lo.shape != sig_lo.shape:
            raise ValueError("mu and sigma bounds must have the same shape.")
        
    # Check if using numpy or cupynumeric for the computation.
    if xp is np:
        # Serial computation for numpy (more efficient for small n).
        return _ei_bounds_numpy(mu_lo, mu_hi, sig_lo, sig_hi, n, y_min_unscaled, pad=pad)
    else:
        # Vectorized computation for cupynumeric (more efficient for large n).
        return _ei_bounds_cupynumeric(mu_lo, mu_hi, sig_lo, sig_hi, n, y_min_unscaled, pad=pad)
    
def _ei_bounds_numpy(mu_lo, mu_hi, sig_lo, sig_hi, n, y_min_unscaled, pad):
    from scipy.stats import norm
    ei_lo = []
    ei_hi = []
    for i in range(n):
        # Get the bounds for the i-th box
        mu_lo_i, mu_hi_i = mu_lo[i], mu_hi[i]
        sig_lo_i, sig_hi_i = sig_lo[i], sig_hi[i]

        # N bounds
        N_lo = y_min_unscaled - mu_hi_i
        N_hi = y_min_unscaled - mu_lo_i

        # Handle sigma == 0 cases
        if sig_hi_i == 0:
            ei_lo.append(0.0)
            ei_hi.append(0.0)
            continue
        elif sig_lo_i == 0:
            # If sig_lo_i == 0 but sig_hi_i > 0, we can still compute bounds using sig_hi_i for the upper bound and 0 for the lower bound.
            sig_lo_i = pad  # small positive number to avoid division by zero
            flag_ei_lo_0 = True
        else:
            flag_ei_lo_0 = False

        # J = 1/sigma bounds
        J_lo = 1.0 / sig_hi_i
        J_hi = 1.0 / sig_lo_i
        if flag_ei_lo_0:
            sig_lo_i = 0.0  # reset to zero for the lower bound case

        # Z = N * J bounds
        Z_lo = min(min(N_lo * J_lo, N_lo * J_hi), min(N_hi * J_lo, N_hi * J_hi))
        Z_hi = max(max(N_lo * J_lo, N_lo * J_hi), max(N_hi * J_lo, N_hi * J_hi))

        # Phi bounds (monotone)
        Phi_lo = norm.cdf(Z_lo)
        Phi_hi = norm.cdf(Z_hi)

        # phi bounds (unimodal, symmetric)
        norm_pdf_Z_lo = norm.pdf(Z_lo)
        norm_pdf_Z_hi = norm.pdf(Z_hi)
        phi_lo = min(norm_pdf_Z_lo, norm_pdf_Z_hi)
        if Z_lo <= 0 <= Z_hi:
            phi_hi = _INV_SQRT2PI
        else:
            phi_hi = max(norm_pdf_Z_lo, norm_pdf_Z_hi)

        # U = N * Phi, V = sigma * phi bounds
        U_lo = min(min(N_lo * Phi_lo, N_lo * Phi_hi), min(N_hi * Phi_lo, N_hi * Phi_hi))
        U_hi = max(max(N_lo * Phi_lo, N_lo * Phi_hi), max(N_hi * Phi_lo, N_hi * Phi_hi))    
        V_lo = min(min(sig_lo_i * phi_lo, sig_lo_i * phi_hi), min(sig_hi_i * phi_lo, sig_hi_i * phi_hi))
        V_hi = max(max(sig_lo_i * phi_lo, sig_lo_i * phi_hi), max(sig_hi_i * phi_lo, sig_hi_i * phi_hi))

        # EI = U + V bounds
        EI_lo = U_lo + V_lo
        EI_hi = U_hi + V_hi

        # If sigma was exactly zero, EI is zero
        if flag_ei_lo_0:
            ei_lo.append(0.0)
        else:
            ei_lo.append(max(EI_lo, 0))
        ei_hi.append(max(EI_hi, 0))

    return np.array(ei_lo), np.array(ei_hi)

def _ei_bounds_cupynumeric(mu_lo, mu_hi, sig_lo, sig_hi, n, y_min_unscaled, pad):
    import cupynumeric as cp
    inv_pad = 1.0 / pad

    # Create buffers for intermediate computations.
    # N shares with V
    N_lo = cp.empty(n, dtype=cp.float64)
    N_hi = cp.empty(n, dtype=cp.float64)
    # J shares with Phi
    J_lo = cp.empty(n, dtype=cp.float64)
    J_hi = cp.empty(n, dtype=cp.float64)
    # Z shares with phi
    Z_lo = cp.empty(n, dtype=cp.float64)
    Z_hi = cp.empty(n, dtype=cp.float64)
    tmp = cp.empty(n, dtype=cp.float64)
    mask0 = cp.empty(n, dtype=bool)
    mask1 = cp.empty(n, dtype=bool)

    # N = y_min - mu
    cp.subtract(y_min_unscaled, mu_hi, out=N_lo)
    cp.subtract(y_min_unscaled, mu_lo, out=N_hi)

    # J bounds = 1/sigma without divide-by-zero warnings.
    cp.equal(sig_hi, 0.0, out=mask0)
    J_lo[...] = inv_pad
    J_lo[~mask0] = 1.0 / sig_hi[~mask0]
    cp.equal(sig_lo, 0.0, out=mask1)
    J_hi[...] = inv_pad
    J_hi[~mask1] = 1.0 / sig_lo[~mask1]

    # Z = N * J.
    _interval_product_bounds(N_lo, N_hi, J_lo, J_hi, cp, Z_lo, Z_hi, tmp)

    # Phi(Z), stored in J buffers.
    _norm_cdf(Z_lo, cp, out=J_lo, tmp=tmp)
    _norm_cdf(Z_hi, cp, out=J_hi, tmp=tmp)

    # Track where [Z_lo, Z_hi] crosses zero (needed for phi upper bound).
    cp.less_equal(Z_lo, 0.0, out=mask0)
    cp.greater_equal(Z_hi, 0.0, out=mask1)
    cp.logical_and(mask0, mask1, out=mask0)

    # phi(Z), stored in Z buffers.
    _norm_pdf(Z_lo, cp, out=Z_lo)
    _norm_pdf(Z_hi, cp, out=Z_hi)
    cp.maximum(Z_lo, Z_hi, out=tmp)
    cp.minimum(Z_lo, Z_hi, out=Z_lo)
    Z_hi[...] = tmp
    Z_hi[mask0] = _INV_SQRT2PI

    # U = N * Phi, stored in EI buffers for now.
    EI_lo = cp.empty(n, dtype=cp.float64)
    EI_hi = cp.empty(n, dtype=cp.float64)
    _interval_product_bounds(N_lo, N_hi, J_lo, J_hi, cp, EI_lo, EI_hi, tmp)

    # V = sigma * phi, stored in N buffers.
    _interval_product_bounds(sig_lo, sig_hi, Z_lo, Z_hi, cp, N_lo, N_hi, tmp)

    # EI = U + V
    cp.add(EI_lo, N_lo, out=EI_lo)
    cp.add(EI_hi, N_hi, out=EI_hi)
    cp.maximum(EI_lo, 0.0, out=EI_lo)
    cp.maximum(EI_hi, 0.0, out=EI_hi)

    # If sigma interval hits zero exactly, enforce the same EI conventions as before.
    cp.equal(sig_hi, 0.0, out=mask0)
    EI_hi[mask0] = 0.0
    cp.equal(sig_lo, 0.0, out=mask1)
    cp.logical_or(mask0, mask1, out=mask1)
    EI_lo[mask1] = 0.0

    return EI_lo, EI_hi

def _interval_product_bounds(a_lo, a_hi, b_lo, b_hi, xp, out_lo, out_hi, tmp):
    """Compute interval bounds for elementwise products [a_lo, a_hi] * [b_lo, b_hi]."""
    xp.multiply(a_lo, b_lo, out=out_lo)
    out_hi[...] = out_lo

    xp.multiply(a_lo, b_hi, out=tmp)
    xp.minimum(out_lo, tmp, out=out_lo)
    xp.maximum(out_hi, tmp, out=out_hi)

    xp.multiply(a_hi, b_lo, out=tmp)
    xp.minimum(out_lo, tmp, out=out_lo)
    xp.maximum(out_hi, tmp, out=out_hi)

    xp.multiply(a_hi, b_hi, out=tmp)
    xp.minimum(out_lo, tmp, out=out_lo)
    xp.maximum(out_hi, tmp, out=out_hi)

# Normal CDF/PDF using erf approximation for cupynumeric, since it doesn't have scipy.stats.norm.
def _erf_approx(x, xp, *, out=None):
    """
    Approximate erf(x) using Abramowitz & Stegun 7.1.26, which has absolute error < 1.5e-7.
    With module selection and output reuse.
    """
    x = xp.asarray(x, dtype=xp.float64)
    if out is None:
        out = xp.empty_like(x)
    
    # Coefficients
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    sign = xp.sign(x)

    # Buffers for intermediate computations
    ax = xp.empty_like(x)
    t = xp.empty_like(x)
    poly = xp.empty_like(x)

    # t = 1 / (1 + p * |x|)
    xp.abs(x, out=ax)
    xp.multiply(ax, p, out=t)
    xp.add(t, 1.0, out=t)
    xp.divide(1.0, t, out=t)

    # poly = a5*t^5 + a4*t^4 + a3*t^3 + a2*t^2 + a1*t
    xp.multiply(t, a5, out=poly)
    xp.add(poly, a4, out=poly)
    xp.multiply(poly, t, out=poly)
    xp.add(poly, a3, out=poly)
    xp.multiply(poly, t, out=poly)
    xp.add(poly, a2, out=poly)
    xp.multiply(poly, t, out=poly)
    xp.add(poly, a1, out=poly)
    xp.multiply(poly, t, out=poly)

    # exp(-x^2) = exp(-ax), where ax = x^2
    xp.multiply(ax, ax, out=ax)
    xp.multiply(ax, -1.0, out=ax)
    xp.exp(ax, out=ax)

    # erf = sign * (1 - poly * exp(-x^2))
    xp.multiply(poly, ax, out=poly)
    xp.subtract(1.0, poly, out=out)
    xp.multiply(out, sign, out=out)

    return out

def _norm_cdf(z,xp, *, out=None, tmp=None):
    z = xp.asarray(z, dtype=xp.float64)
    if out is None:
        out = xp.empty_like(z)
    if tmp is None:
        tmp = xp.empty_like(z)

    # CDF(z) = 0.5 * (1 + erf(z / sqrt(2)))
    xp.divide(z, _SQRT2, out=tmp)
    _erf_approx(tmp, xp, out=tmp)
    xp.add(tmp, 1.0, out=tmp)
    xp.multiply(tmp, 0.5, out=out)
    return out

def _norm_pdf(z,xp, *, out=None):
    z = xp.asarray(z, dtype=xp.float64)
    if out is None:
        out = xp.empty_like(z)

    # PDF(z) = (1 / sqrt(2*pi)) * exp(-0.5 * z^2)
    xp.multiply(z, z, out=out)
    xp.multiply(out, -0.5, out=out)
    xp.exp(out, out=out)
    xp.multiply(out, _INV_SQRT2PI, out=out)
    return out