from __future__ import annotations

from importlib import import_module

import numpy as np

from tamubo.utils import BackendName, resolve_backend

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

    Parameters
    ----------
    bounds_L, bounds_U : np.ndarray or cupynumeric.ndarray
        Lower/upper bounds for n boxes. Accepts shape (n*d,),
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
        if bounds_L.ndim == 2:
            bounds_L = bounds_L.ravel()
        if bounds_U.ndim == 2:
            bounds_U = bounds_U.ravel()
        if bounds_L.size != n * d or bounds_U.size != n * d:
            raise ValueError("bounds_L/R must have size n*d.")
        if xi.size != d:
            raise ValueError("xi must have size d.")

    # Serial computation for numpy (more efficient for small n).
    if xp is np:
        K_lo = []
        K_hi = []
        # For each box, compute the kernel bounds to xi
        for i in range(n):
            # Extract the bounds for the i-th box
            bounds_L_i = bounds_L[i * d : (i + 1) * d] # (d,) 
            bounds_U_i = bounds_U[i * d : (i + 1) * d] # (d,)

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
        xi_ext = xp.tile(xi, n)

        d_min = xp.maximum(xp.maximum(bounds_L - xi_ext, xi_ext - bounds_U), 0)
        d_max = xp.maximum(xp.abs(bounds_L - xi_ext), xp.abs(xi_ext - bounds_U))

        D_min = xp.linalg.norm(d_min.reshape(n, d), axis=1)
        D_max = xp.linalg.norm(d_max.reshape(n, d), axis=1)

        K_lo = sigma_f_2 * xp.exp(-1 / (2 * length_scale ** 2) * xp.power(D_max, 2))
        K_hi = sigma_f_2 * xp.exp(-1 / (2 * length_scale ** 2) * xp.power(D_min, 2))

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
        alphaK_lo = xp.zeros((n, N))
        alphaK_hi = xp.zeros((n, N))

        alpha_mask_pos = alpha >= 0

        # when alpha >= 0: hi→hi, lo→lo
        alphaK_hi[:, alpha_mask_pos] = K_hi[:, alpha_mask_pos] * alpha[alpha_mask_pos]
        alphaK_lo[:, alpha_mask_pos] = K_lo[:, alpha_mask_pos] * alpha[alpha_mask_pos]

        # when alpha < 0: hi→lo, lo→hi
        alphaK_lo[:, ~alpha_mask_pos] = K_hi[:, ~alpha_mask_pos] * alpha[~alpha_mask_pos]
        alphaK_hi[:, ~alpha_mask_pos] = K_lo[:, ~alpha_mask_pos] * alpha[~alpha_mask_pos]

        # sum each row (normalized scale)
        mu_hi = alphaK_hi.sum(axis=1)  # shape (n,)
        mu_lo = alphaK_lo.sum(axis=1)  # shape (n,)

        # unnormalize to original target scale
        mu_hi = y_train_mean + y_train_std * mu_hi
        mu_lo = y_train_mean + y_train_std * mu_lo

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
    # Initialize v_lo and v_hi to store the forward solve results for all boxes at once.
    v_lo = cp.zeros((n, N))
    v_hi = cp.zeros((n, N))

    # Transpose L for indexing purpose
    LT = L.T

    # Forward solve bounds: L v = k
    for j in range(N):
        S_lo = 0
        S_hi = 0
        # S_j = Σ_{i<j} LT_{ij} v_i, with each term interval-bounded
        for i in range(j):
            LTij = LT[i, j]
            LTv_lo = LTij * v_lo[:, i]  # (n,)
            LTV_hi = LTij * v_hi[:, i]  # (n,)
            Si_lo = cp.minimum(LTv_lo, LTV_hi)  # (n,)
            Si_hi = cp.maximum(LTv_lo, LTV_hi)  # (n,)
            S_lo = S_lo + Si_lo  # (n,)
            S_hi = S_hi + Si_hi  # (n,)

        # N_j = k_j - S_j
        N_lo = K_lo[:, j] - S_hi
        N_hi = K_hi[:, j] - S_lo

        # v_j = N_j/LT{jj}
        LTjj = LT[j, j]
        v_lo[:, j] = N_lo / LTjj
        v_hi[:, j] = N_hi / LTjj

    # Q = vTv
    v2_lo = v_lo * v_lo  # (n, N)
    v2_hi = v_hi * v_hi  # (n, N)
    flag_v2_0 = (v_lo < 0) & (v_hi > 0)  # (n, N)
    Q_lo = cp.sum(cp.where(flag_v2_0, 0, cp.minimum(v2_lo, v2_hi)), axis=1)  # (n,)
    Q_hi = cp.sum(cp.maximum(v2_lo, v2_hi), axis=1)  # (n,)

    # var = sigma_f_2 - Q
    var_lo = cp.maximum(0, sigma_f_2 - Q_hi)
    var_hi = cp.maximum(0, sigma_f_2 - Q_lo)

    # sig = sqrt(var)
    sig_lo = cp.sqrt(var_lo)
    sig_hi = cp.sqrt(var_hi)

    # unnormalize to original target scale
    sig_lo = sig_lo * y_train_std
    sig_hi = sig_hi * y_train_std

    return (sig_lo, sig_hi)


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
        return _ei_bounds_numpy(mu_lo, mu_hi, sig_lo, sig_hi, n, y_min_unscaled)
    else:
        # Vectorized computation for cupynumeric (more efficient for large n).
        return _ei_bounds_cupynumeric(mu_lo, mu_hi, sig_lo, sig_hi, y_min_unscaled)
    
def _ei_bounds_numpy(mu_lo, mu_hi, sig_lo, sig_hi, n, y_min_unscaled): #FIX
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
            sig_lo_i = 1e-5  # small positive number to avoid division by zero
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
            phi_hi = norm.pdf(0)
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

def _ei_bounds_cupynumeric(mu_lo, mu_hi, sig_lo, sig_hi, y_min_unscaled):
    import cupynumeric as cp
    # N bounds
    N_lo = y_min_unscaled - mu_hi  # (n,)
    N_hi = y_min_unscaled - mu_lo  # (n,)

    # Handle sigma == 0 cases
    mask_sig_lo_0 = sig_lo == 0.0  # (n,)
    mask_ei_0 = sig_hi == 0.0  # (n,)
    mask_ei_lo_0 = mask_sig_lo_0 & ~mask_ei_0  # (n,)

    pad = 1e-5
    sig_lo_safe = cp.where(mask_sig_lo_0, pad, sig_lo)  # (n,)
    sig_hi_safe = cp.where(mask_ei_0, pad, sig_hi)  # (n,)

    # J = 1/sigma bounds
    J_lo = 1.0 / sig_hi_safe  # (n,)
    J_hi = 1.0 / sig_lo_safe  # (n,)

    # Z = N * J
    Z_lo = cp.minimum(cp.minimum(N_lo * J_lo, N_lo * J_hi), cp.minimum(N_hi * J_lo, N_hi * J_hi)) # (n,)
    Z_hi = cp.maximum(cp.maximum(N_lo * J_lo, N_lo * J_hi), cp.maximum(N_hi * J_lo, N_hi * J_hi)) # (n,)

    # Phi bounds (monotone)
    Phi_lo = _norm_cdf(Z_lo, cp)  # (n,)
    Phi_hi = _norm_cdf(Z_hi, cp)  # (n,)

    # phi bounds (unimodal, symmetric)
    norm_pdf_Z_lo = _norm_pdf(Z_lo, cp)  # (n,)
    norm_pdf_Z_hi = _norm_pdf(Z_hi, cp)  # (n,)
    phi_lo = cp.minimum(norm_pdf_Z_lo, norm_pdf_Z_hi)  # (n,)
    mask_phi_hi = (Z_lo <= 0.0) & (Z_hi >= 0.0)  # (n,)
    phi_hi = cp.where(mask_phi_hi, _norm_pdf(0, cp), cp.maximum(norm_pdf_Z_lo, norm_pdf_Z_hi))  # (n,)

    # U = N * Phi, V = sigma * phi
    U_lo = cp.minimum(cp.minimum(N_lo * Phi_lo, N_lo * Phi_hi), cp.minimum(N_hi * Phi_lo, N_hi * Phi_hi)) # (n,)
    U_hi = cp.maximum(cp.maximum(N_lo * Phi_lo, N_lo * Phi_hi), cp.maximum(N_hi * Phi_lo, N_hi * Phi_hi)) # (n,)    
    V_lo = cp.minimum(cp.minimum(sig_lo * phi_lo, sig_lo * phi_hi), cp.minimum(sig_hi * phi_lo, sig_hi * phi_hi))  # (n,)
    V_hi = cp.maximum(cp.maximum(sig_lo * phi_lo, sig_lo * phi_hi), cp.maximum(sig_hi * phi_lo, sig_hi * phi_hi))  # (n,)

    # EI = U + V
    EI_lo = U_lo + V_lo  # (n,)
    EI_hi = U_hi + V_hi  # (n,)

    # If sigma was exactly zero, EI is zero
    EI_lo = cp.where((mask_ei_lo_0 | mask_ei_0), 0.0, cp.maximum(EI_lo, 0))  # (n,)
    EI_hi = cp.where(mask_ei_0, 0.0, cp.maximum(EI_hi, 0))  # (n,)

    return (EI_lo, EI_hi)

# Normal CDF/PDF using erf approximation for cupynumeric, since it doesn't have scipy.stats.norm.
def _erf_approx(x,xp):
    """
    Approximate erf(x) using Abramowitz & Stegun 7.1.26.
    Max error ~1.5e-7.
    """
    # Coefficients
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign = xp.sign(x)
    ax = xp.abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * xp.exp(-ax * ax)
    return sign * y
def _norm_cdf(z,xp):
    sqrt2 = xp.sqrt(2.0)
    return 0.5 * (1.0 + _erf_approx(z / sqrt2, xp))
def _norm_pdf(z,xp):
    inv_sqrt2pi = 1.0 / xp.sqrt(2.0 * xp.pi)
    return inv_sqrt2pi * xp.exp(-0.5 * z * z)