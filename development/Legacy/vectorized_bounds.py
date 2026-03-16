from __future__ import annotations

import cupynumeric as cp


# Define rbf_k_bounds
def rbf_k_bounds(bounds_L, bounds_U, xi, n, d, sigma_f_2, length_scale, validation=True):
    """
    Compute lower/upper bounds of the RBF kernel between xi and each hyperbox.

    Parameters
    ----------
    bounds_L, bounds_U : cupynumeric.ndarray
        Lower/upper bounds for n boxes. Accepts shape (n*d,) or (n, d),
        with box coordinates stored consecutively by dimension.
    xi : cupynumeric.ndarray
        Query point in R^d with shape (d,).
    n : int
        Number of boxes.
    d : int
        Dimension of the design space.
    sigma_f_2 : float
        Kernel variance from the trained GP.
    length_scale : float
        RBF length scale from the trained GP.
    validation: default=True
        Validate dimensions of inputs. Default: True.

    Returns
    -------
    (K_lo, K_hi) : tuple[cupynumeric.ndarray, cupynumeric.ndarray]
        Lower and upper kernel bounds for each box, each with shape (n,N).
    """
    if validation:
        if bounds_L.ndim == 2:
            bounds_L = bounds_L.ravel()
        if bounds_U.ndim == 2:
            bounds_U = bounds_U.ravel()
        if bounds_L.size != n * d or bounds_U.size != n * d:
            raise ValueError("bounds_L/R must have size n*d.")
        if xi.size != d:
            raise ValueError("xi must have size d.")

    xi_ext = cp.tile(xi, n)

    d_min = cp.maximum(cp.maximum(bounds_L - xi_ext, xi_ext - bounds_U), 0)
    d_max = cp.maximum(cp.abs(bounds_L - xi_ext), cp.abs(xi_ext - bounds_U))

    D_min = cp.linalg.norm(d_min.reshape(n, d), axis=1)
    D_max = cp.linalg.norm(d_max.reshape(n, d), axis=1)

    K_lo = sigma_f_2 * cp.exp(-1 / (2 * length_scale ** 2) * cp.power(D_max, 2))
    K_hi = sigma_f_2 * cp.exp(-1 / (2 * length_scale ** 2) * cp.power(D_min, 2))

    return (K_lo, K_hi)


def mu_bounds(alpha, K_lo, K_hi, n, N, y_train_mean=0, y_train_std=1, validate=True):
    """
    Compute lower/upper bounds on the GP posterior mean per box, in original scale.

    Parameters
    ----------
    alpha : cupynumeric.ndarray
        GP dual coefficients (typically gp.alpha_), shape (N,).
    K_lo, K_hi : cupynumeric.ndarray
        Lower/upper kernel bounds between each box and each training point,
        each with shape (n, N).
    n : int
        Number of boxes.
    N : int
        Number of training points.
    y_train_mean : float, default=0
        Training target mean used by the GP (normalize_y=True).
    y_train_std : float, default=1
        Training target std used by the GP (normalize_y=True).
    validate : bool, default=True
        If True, validate shapes and sizes.

    Returns
    -------
    (mu_lo, mu_hi) : tuple[cupynumeric.ndarray, cupynumeric.ndarray]
        Lower and upper bounds on the mean for each box, each with shape (n,).
    """
    if validate:
        if alpha.size != N:
            raise ValueError("alpha must have size N.")
        if K_lo.shape != (n, N) or K_hi.shape != (n, N):
            raise ValueError("K_lo and K_hi must have shape (n, N).")

    alphaK_lo = cp.zeros((n, N))
    alphaK_hi = cp.zeros((n, N))

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


def sigma_bounds(K_lo, K_hi, L, n, N, sigma_f_2, y_train_std=1, validate=True):
    """
    Vectorized sigma bounds for all boxes at once.

    Parameters
    ----------
    K_lo, K_hi : cupynumeric.ndarray
        Kernel bounds per box vs training points, shape (n, N).
    L : cupynumeric.ndarray
        Cholesky factor (N, N) of K + σ_n^2 I (lower triangular).
    n : int
        Number of boxes.
    N : int
        Number of training points.
    sigma_f_2 : float
        Kernel variance σ_f^2.
    y_train_std : float, default=1
        Training target std used by the GP (normalize_y=True).
    validate : bool, default=True
        If True, validate shapes and sizes.

    Returns
    -------
    (sig_lo, sig_hi) : tuple[cupynumeric.ndarray, cupynumeric.ndarray]
        Lower/upper sigma bounds per box, each shape (n,).
    """
    if validate:
        if K_lo.shape != K_hi.shape:
            raise ValueError("K_lo and K_hi must have the same shape.")
        if L.shape[0] != L.shape[1]:
            raise ValueError("L must be square.")
        if K_lo.shape[1] != L.shape[0]:
            raise ValueError("K_lo/K_hi second dim must match L size.")

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


def ei_bounds(mu_lo, mu_hi, sig_lo, sig_hi, y_min, y_train_mean=0, y_train_std=1, validate=True):
    """
    Vectorized IA bounds for EI across all boxes:
      EI = N * Phi(Z) + sigma * phi(Z), where N = f_min - mu, Z = N / sigma.

    Parameters
    ----------
    mu_lo, mu_hi : cupynumeric.ndarray
        Mean bounds per box, shape (n,).
    sig_lo, sig_hi : cupynumeric.ndarray
        Sigma bounds per box, shape (n,), sig_lo >= 0.
    y_min : float
        Minimum of normalized training targets (as in model.y_train_).
    y_train_mean, y_train_std : float, optional
        Normalization parameters used by GP (normalize_y=True).
    validate : bool, optional
        If True, validate shapes/sizes.

    Returns
    -------
    (ei_lo, ei_hi) : tuple[cupynumeric.ndarray, cupynumeric.ndarray]
        EI bounds per box, shape (n,).
    """
    if validate:
        if mu_lo.shape != mu_hi.shape:
            raise ValueError("mu_lo and mu_hi must have the same shape.")
        if sig_lo.shape != sig_hi.shape:
            raise ValueError("sig_lo and sig_hi must have the same shape.")
        if mu_lo.shape != sig_lo.shape:
            raise ValueError("mu and sigma bounds must have the same shape.")

    # f_min in original scale
    f_min = y_train_std * y_min + y_train_mean

    # N bounds
    N_lo = f_min - mu_hi  # (n,)
    N_hi = f_min - mu_lo  # (n,)

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

    # interval product helper
    def prod_bounds(a_lo, a_hi, b_lo, b_hi):
        p1 = a_lo * b_lo
        p2 = a_lo * b_hi
        p3 = a_hi * b_lo
        p4 = a_hi * b_hi
        lo = cp.minimum(cp.minimum(p1, p2), cp.minimum(p3, p4))
        hi = cp.maximum(cp.maximum(p1, p2), cp.maximum(p3, p4))
        return lo, hi

    # Z = N * J
    Z_lo, Z_hi = prod_bounds(N_lo, N_hi, J_lo, J_hi)  # both (n,)

    # Normal CDF/PDF using erf
    sqrt2 = cp.sqrt(2.0)
    inv_sqrt2pi = 1.0 / cp.sqrt(2.0 * cp.pi)

    def erf_approx(x):
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

        sign = cp.sign(x)
        ax = cp.abs(x)
        t = 1.0 / (1.0 + p * ax)
        y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * cp.exp(-ax * ax)
        return sign * y

    def norm_cdf(z):
        return 0.5 * (1.0 + erf_approx(z / sqrt2))

    def norm_pdf(z):
        return inv_sqrt2pi * cp.exp(-0.5 * z * z)

    # Phi bounds (monotone)
    Phi_lo = norm_cdf(Z_lo)  # (n,)
    Phi_hi = norm_cdf(Z_hi)  # (n,)

    # phi bounds (unimodal, symmetric)
    norm_pdf_Z_lo = norm_pdf(Z_lo)  # (n,)
    norm_pdf_Z_hi = norm_pdf(Z_hi)  # (n,)
    phi_lo = cp.minimum(norm_pdf_Z_lo, norm_pdf_Z_hi)  # (n,)
    mask_phi_hi = (Z_lo <= 0.0) & (Z_hi >= 0.0)  # (n,)
    phi_hi = cp.where(mask_phi_hi, norm_pdf(0), cp.maximum(norm_pdf_Z_lo, norm_pdf_Z_hi))  # (n,)

    # U = N * Phi, V = sigma * phi
    U_lo, U_hi = prod_bounds(N_lo, N_hi, Phi_lo, Phi_hi)  # both (n,)
    V_lo, V_hi = prod_bounds(sig_lo, sig_hi, phi_lo, phi_hi)  # both (n,)

    # EI = U + V
    EI_lo = U_lo + V_lo  # (n,)
    EI_hi = U_hi + V_hi  # (n,)

    # If sigma was exactly zero, EI is zero
    EI_lo = cp.where((mask_ei_lo_0 | mask_ei_0), 0.0, cp.maximum(EI_lo, 0))  # (n,)
    EI_hi = cp.where(mask_ei_0, 0.0, cp.maximum(EI_hi, 0))  # (n,)

    return (EI_lo, EI_hi)
