from __future__ import annotations

import cupynumeric as cp


def rbf_k_bounds(bounds_l, bounds_u, xi, n, d, sigma_f_2, length_scale, validation=True):
    """Vectorized RBF kernel bounds between xi and each box."""

    if validation:
        if bounds_l.ndim == 2:
            bounds_l = bounds_l.ravel()
        if bounds_u.ndim == 2:
            bounds_u = bounds_u.ravel()
        if bounds_l.size != n * d or bounds_u.size != n * d:
            raise ValueError("bounds_l/bounds_u must have size n*d")
        if xi.size != d:
            raise ValueError("xi must have size d")

    xi_ext = cp.tile(xi, n)
    d_min = cp.maximum(cp.maximum(bounds_l - xi_ext, xi_ext - bounds_u), 0)
    d_max = cp.maximum(cp.abs(bounds_l - xi_ext), cp.abs(xi_ext - bounds_u))

    d_min_norm = cp.linalg.norm(d_min.reshape(n, d), axis=1)
    d_max_norm = cp.linalg.norm(d_max.reshape(n, d), axis=1)

    k_lo = sigma_f_2 * cp.exp(-(d_max_norm ** 2) / (2 * length_scale ** 2))
    k_hi = sigma_f_2 * cp.exp(-(d_min_norm ** 2) / (2 * length_scale ** 2))
    return k_lo, k_hi


def mu_bounds(alpha, k_lo, k_hi, n, n_train, y_train_mean=0, y_train_std=1, validate=True):
    """Vectorized posterior-mean bounds per box."""

    if validate:
        if alpha.size != n_train:
            raise ValueError("alpha must have size n_train")
        if k_lo.shape != (n, n_train) or k_hi.shape != (n, n_train):
            raise ValueError("k_lo and k_hi must have shape (n, n_train)")

    alpha_k_lo = cp.zeros((n, n_train))
    alpha_k_hi = cp.zeros((n, n_train))

    alpha_mask_pos = alpha >= 0
    alpha_k_hi[:, alpha_mask_pos] = k_hi[:, alpha_mask_pos] * alpha[alpha_mask_pos]
    alpha_k_lo[:, alpha_mask_pos] = k_lo[:, alpha_mask_pos] * alpha[alpha_mask_pos]

    alpha_k_lo[:, ~alpha_mask_pos] = k_hi[:, ~alpha_mask_pos] * alpha[~alpha_mask_pos]
    alpha_k_hi[:, ~alpha_mask_pos] = k_lo[:, ~alpha_mask_pos] * alpha[~alpha_mask_pos]

    mu_hi = alpha_k_hi.sum(axis=1)
    mu_lo = alpha_k_lo.sum(axis=1)

    mu_hi = y_train_mean + y_train_std * mu_hi
    mu_lo = y_train_mean + y_train_std * mu_lo
    return mu_lo, mu_hi


def sigma_bounds(k_lo, k_hi, chol_l, n, n_train, sigma_f_2, y_train_std=1, validate=True):
    """Vectorized posterior-standard-deviation bounds per box."""

    if validate:
        if k_lo.shape != k_hi.shape:
            raise ValueError("k_lo and k_hi must have same shape")
        if chol_l.shape[0] != chol_l.shape[1]:
            raise ValueError("chol_l must be square")
        if k_lo.shape[1] != chol_l.shape[0]:
            raise ValueError("k_lo/k_hi second dim must match chol_l size")

    v_lo = cp.zeros((n, n_train))
    v_hi = cp.zeros((n, n_train))

    lt = chol_l.T
    for j in range(n_train):
        s_lo = 0
        s_hi = 0
        for i in range(j):
            lt_ij = lt[i, j]
            lt_v_lo = lt_ij * v_lo[:, i]
            lt_v_hi = lt_ij * v_hi[:, i]
            s_lo = s_lo + cp.minimum(lt_v_lo, lt_v_hi)
            s_hi = s_hi + cp.maximum(lt_v_lo, lt_v_hi)

        n_lo = k_lo[:, j] - s_hi
        n_hi = k_hi[:, j] - s_lo

        lt_jj = lt[j, j]
        v_lo[:, j] = n_lo / lt_jj
        v_hi[:, j] = n_hi / lt_jj

    v2_lo = v_lo * v_lo
    v2_hi = v_hi * v_hi
    crosses_zero = (v_lo < 0) & (v_hi > 0)

    q_lo = cp.sum(cp.where(crosses_zero, 0, cp.minimum(v2_lo, v2_hi)), axis=1)
    q_hi = cp.sum(cp.maximum(v2_lo, v2_hi), axis=1)

    var_lo = cp.maximum(0, sigma_f_2 - q_hi)
    var_hi = cp.maximum(0, sigma_f_2 - q_lo)

    sig_lo = cp.sqrt(var_lo) * y_train_std
    sig_hi = cp.sqrt(var_hi) * y_train_std
    return sig_lo, sig_hi


def ei_bounds(mu_lo, mu_hi, sig_lo, sig_hi, y_min, y_train_mean=0, y_train_std=1, validate=True):
    """Vectorized EI interval bounds for all boxes."""

    if validate:
        if mu_lo.shape != mu_hi.shape:
            raise ValueError("mu_lo and mu_hi must have the same shape")
        if sig_lo.shape != sig_hi.shape:
            raise ValueError("sig_lo and sig_hi must have the same shape")
        if mu_lo.shape != sig_lo.shape:
            raise ValueError("mu and sigma bounds must have the same shape")

    f_min = y_train_std * y_min + y_train_mean

    n_lo = f_min - mu_hi
    n_hi = f_min - mu_lo

    mask_sig_lo_0 = sig_lo == 0.0
    mask_ei_0 = sig_hi == 0.0
    mask_ei_lo_0 = mask_sig_lo_0 & ~mask_ei_0

    pad = 1e-5
    sig_lo_safe = cp.where(mask_sig_lo_0, pad, sig_lo)
    sig_hi_safe = cp.where(mask_ei_0, pad, sig_hi)

    j_lo = 1.0 / sig_hi_safe
    j_hi = 1.0 / sig_lo_safe

    def prod_bounds(a_lo, a_hi, b_lo, b_hi):
        p1 = a_lo * b_lo
        p2 = a_lo * b_hi
        p3 = a_hi * b_lo
        p4 = a_hi * b_hi
        lo = cp.minimum(cp.minimum(p1, p2), cp.minimum(p3, p4))
        hi = cp.maximum(cp.maximum(p1, p2), cp.maximum(p3, p4))
        return lo, hi

    z_lo, z_hi = prod_bounds(n_lo, n_hi, j_lo, j_hi)

    sqrt2 = cp.sqrt(2.0)
    inv_sqrt2pi = 1.0 / cp.sqrt(2.0 * cp.pi)

    def erf_approx(x):
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

    phi_lo = norm_cdf(z_lo)
    phi_hi = norm_cdf(z_hi)

    pdf_z_lo = norm_pdf(z_lo)
    pdf_z_hi = norm_pdf(z_hi)
    psi_lo = cp.minimum(pdf_z_lo, pdf_z_hi)
    crosses_origin = (z_lo <= 0.0) & (z_hi >= 0.0)
    psi_hi = cp.where(crosses_origin, norm_pdf(0), cp.maximum(pdf_z_lo, pdf_z_hi))

    u_lo, u_hi = prod_bounds(n_lo, n_hi, phi_lo, phi_hi)
    v_lo, v_hi = prod_bounds(sig_lo, sig_hi, psi_lo, psi_hi)

    ei_lo = u_lo + v_lo
    ei_hi = u_hi + v_hi

    ei_lo = cp.where((mask_ei_lo_0 | mask_ei_0), 0.0, cp.maximum(ei_lo, 0))
    ei_hi = cp.where(mask_ei_0, 0.0, cp.maximum(ei_hi, 0))
    return ei_lo, ei_hi
