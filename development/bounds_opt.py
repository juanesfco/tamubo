from __future__ import annotations

import math

import cupynumeric as cp

# Constants reused across helper calls.
_SQRT2 = math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _interval_product_bounds(a_lo, a_hi, b_lo, b_hi, out_lo, out_hi, tmp):
    """Compute interval bounds for elementwise products [a_lo, a_hi] * [b_lo, b_hi]."""
    cp.multiply(a_lo, b_lo, out=out_lo)
    out_hi[...] = out_lo

    cp.multiply(a_lo, b_hi, out=tmp)
    cp.minimum(out_lo, tmp, out=out_lo)
    cp.maximum(out_hi, tmp, out=out_hi)

    cp.multiply(a_hi, b_lo, out=tmp)
    cp.minimum(out_lo, tmp, out=out_lo)
    cp.maximum(out_hi, tmp, out=out_hi)

    cp.multiply(a_hi, b_hi, out=tmp)
    cp.minimum(out_lo, tmp, out=out_lo)
    cp.maximum(out_hi, tmp, out=out_hi)


def rbf_k_bounds(bounds_L, bounds_U, xi, n, d, sigma_f_2, length_scale, *, validation=True):
    """Memory-optimized cuPyNumeric implementation for RBF kernel interval bounds."""
    bounds_L = cp.asarray(bounds_L, dtype=cp.float64)
    bounds_U = cp.asarray(bounds_U, dtype=cp.float64)
    xi = cp.asarray(xi, dtype=cp.float64)

    if bounds_L.ndim == 2:
        bounds_L = bounds_L.ravel()
    if bounds_U.ndim == 2:
        bounds_U = bounds_U.ravel()

    if validation:
        if bounds_L.size != n * d or bounds_U.size != n * d:
            raise ValueError("bounds_L and bounds_U must have size n*d.")
        if xi.size != d:
            raise ValueError("xi must have size d.")

    bounds_L_2d = bounds_L.reshape(n, d)
    bounds_U_2d = bounds_U.reshape(n, d)

    diff_lo = cp.empty((n, d), dtype=cp.float64)
    diff_hi = cp.empty((n, d), dtype=cp.float64)
    d_min = cp.empty((n, d), dtype=cp.float64)
    d_max = cp.empty((n, d), dtype=cp.float64)

    # diff_lo = bounds_L - xi, diff_hi = xi - bounds_U
    cp.subtract(bounds_L_2d, xi, out=diff_lo)
    cp.subtract(xi, bounds_U_2d, out=diff_hi)

    # d_min = max(max(diff_lo, diff_hi), 0)
    cp.maximum(diff_lo, diff_hi, out=d_min)
    cp.maximum(d_min, 0.0, out=d_min)

    # d_max = max(abs(diff_lo), abs(diff_hi))
    cp.abs(diff_lo, out=diff_lo)
    cp.abs(diff_hi, out=diff_hi)
    cp.maximum(diff_lo, diff_hi, out=d_max)

    # We only need squared norms for RBF exponent.
    cp.multiply(d_min, d_min, out=d_min)
    cp.multiply(d_max, d_max, out=d_max)
    D2_min = cp.sum(d_min, axis=1)
    D2_max = cp.sum(d_max, axis=1)

    coef = -0.5 / (length_scale * length_scale)

    K_lo = cp.empty_like(D2_max)
    K_hi = cp.empty_like(D2_min)

    cp.multiply(D2_max, coef, out=K_lo)
    cp.multiply(D2_min, coef, out=K_hi)
    cp.exp(K_lo, out=K_lo)
    cp.exp(K_hi, out=K_hi)
    cp.multiply(K_lo, sigma_f_2, out=K_lo)
    cp.multiply(K_hi, sigma_f_2, out=K_hi)

    return K_lo, K_hi


def mu_bounds(
    alpha,
    K_lo,
    K_hi,
    n,
    N,
    *,
    y_train_mean=0.0,
    y_train_std=1.0,
    validation=True,
):
    """Memory-optimized cuPyNumeric implementation for posterior mean interval bounds."""
    alpha = cp.asarray(alpha, dtype=cp.float64)
    K_lo = cp.asarray(K_lo, dtype=cp.float64)
    K_hi = cp.asarray(K_hi, dtype=cp.float64)

    if validation:
        if alpha.size != N:
            raise ValueError("alpha must have size N.")
        if K_lo.shape != (n, N) or K_hi.shape != (n, N):
            raise ValueError("K_lo and K_hi must have shape (n, N).")

    # Split alpha into positive/negative parts to avoid (n, N) intermediates.
    alpha_pos = cp.empty_like(alpha)
    alpha_neg = cp.empty_like(alpha)
    cp.maximum(alpha, 0.0, out=alpha_pos)
    cp.minimum(alpha, 0.0, out=alpha_neg)

    mu_lo = K_lo @ alpha_pos
    tmp = K_hi @ alpha_neg
    cp.add(mu_lo, tmp, out=mu_lo)

    mu_hi = K_hi @ alpha_pos
    tmp = K_lo @ alpha_neg
    cp.add(mu_hi, tmp, out=mu_hi)

    cp.multiply(mu_lo, y_train_std, out=mu_lo)
    cp.add(mu_lo, y_train_mean, out=mu_lo)
    cp.multiply(mu_hi, y_train_std, out=mu_hi)
    cp.add(mu_hi, y_train_mean, out=mu_hi)

    return mu_lo, mu_hi


def sigma_bounds(K_lo, K_hi, L, n, N, sigma_f_2, *, y_train_std=1.0, validation=True):
    """Memory-optimized cuPyNumeric implementation for posterior sigma interval bounds."""
    K_lo = cp.asarray(K_lo, dtype=cp.float64)
    K_hi = cp.asarray(K_hi, dtype=cp.float64)
    L = cp.asarray(L, dtype=cp.float64)

    if validation:
        if K_lo.shape != (n, N) or K_hi.shape != (n, N):
            raise ValueError("K_lo and K_hi must have shape (n, N).")
        if L.shape != (N, N):
            raise ValueError("L must have shape (N, N).")

    # Store recurrence columns; avoid extra (n, N) arrays for v^2 and masks.
    v_lo = cp.empty((n, N), dtype=cp.float64)
    v_hi = cp.empty((n, N), dtype=cp.float64)

    Q_lo = cp.zeros(n, dtype=cp.float64)
    Q_hi = cp.zeros(n, dtype=cp.float64)

    S_lo = cp.empty(n, dtype=cp.float64)
    S_hi = cp.empty(n, dtype=cp.float64)
    prod_a = cp.empty(n, dtype=cp.float64)
    prod_b = cp.empty(n, dtype=cp.float64)
    prod_lo = cp.empty(n, dtype=cp.float64)
    prod_hi = cp.empty(n, dtype=cp.float64)
    num_lo = cp.empty(n, dtype=cp.float64)
    num_hi = cp.empty(n, dtype=cp.float64)

    sq_lo = cp.empty(n, dtype=cp.float64)
    sq_hi = cp.empty(n, dtype=cp.float64)
    sq_min = cp.empty(n, dtype=cp.float64)
    cross_a = cp.empty(n, dtype=bool)
    cross_b = cp.empty(n, dtype=bool)

    for j in range(N):
        S_lo[...] = 0.0
        S_hi[...] = 0.0

        for i in range(j):
            Lji = L[j, i]

            cp.multiply(v_lo[:, i], Lji, out=prod_a)
            cp.multiply(v_hi[:, i], Lji, out=prod_b)
            cp.minimum(prod_a, prod_b, out=prod_lo)
            cp.maximum(prod_a, prod_b, out=prod_hi)
            cp.add(S_lo, prod_lo, out=S_lo)
            cp.add(S_hi, prod_hi, out=S_hi)

        cp.subtract(K_lo[:, j], S_hi, out=num_lo)
        cp.subtract(K_hi[:, j], S_lo, out=num_hi)

        Ljj = L[j, j]
        cp.divide(num_lo, Ljj, out=v_lo[:, j])
        cp.divide(num_hi, Ljj, out=v_hi[:, j])

        # Incrementally accumulate Q bounds to avoid v2 matrices.
        cp.multiply(v_lo[:, j], v_lo[:, j], out=sq_lo)
        cp.multiply(v_hi[:, j], v_hi[:, j], out=sq_hi)

        cp.maximum(sq_lo, sq_hi, out=prod_hi)
        cp.add(Q_hi, prod_hi, out=Q_hi)

        cp.minimum(sq_lo, sq_hi, out=sq_min)
        cp.less(v_lo[:, j], 0.0, out=cross_a)
        cp.greater(v_hi[:, j], 0.0, out=cross_b)
        cp.logical_and(cross_a, cross_b, out=cross_a)
        sq_min[cross_a] = 0.0
        cp.add(Q_lo, sq_min, out=Q_lo)

    var_lo = cp.empty_like(Q_hi)
    var_hi = cp.empty_like(Q_lo)
    cp.subtract(sigma_f_2, Q_hi, out=var_lo)
    cp.subtract(sigma_f_2, Q_lo, out=var_hi)
    cp.maximum(var_lo, 0.0, out=var_lo)
    cp.maximum(var_hi, 0.0, out=var_hi)

    sig_lo = cp.empty_like(var_lo)
    sig_hi = cp.empty_like(var_hi)
    cp.sqrt(var_lo, out=sig_lo)
    cp.sqrt(var_hi, out=sig_hi)
    cp.multiply(sig_lo, y_train_std, out=sig_lo)
    cp.multiply(sig_hi, y_train_std, out=sig_hi)

    return sig_lo, sig_hi


def _erf_approx(x, *, out=None):
    """Approximate erf(x) using Abramowitz & Stegun 7.1.26, with output reuse."""
    x = cp.asarray(x, dtype=cp.float64)
    if out is None:
        out = cp.empty_like(x)

    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign = cp.sign(x)
    ax = cp.empty_like(x)
    t = cp.empty_like(x)
    poly = cp.empty_like(x)

    cp.abs(x, out=ax)
    cp.multiply(ax, p, out=t)
    cp.add(t, 1.0, out=t)
    cp.divide(1.0, t, out=t)

    cp.multiply(t, a5, out=poly)
    cp.add(poly, a4, out=poly)
    cp.multiply(poly, t, out=poly)
    cp.add(poly, a3, out=poly)
    cp.multiply(poly, t, out=poly)
    cp.add(poly, a2, out=poly)
    cp.multiply(poly, t, out=poly)
    cp.add(poly, a1, out=poly)
    cp.multiply(poly, t, out=poly)

    cp.multiply(ax, ax, out=ax)
    cp.multiply(ax, -1.0, out=ax)
    cp.exp(ax, out=ax)

    cp.multiply(poly, ax, out=poly)
    cp.subtract(1.0, poly, out=out)
    cp.multiply(out, sign, out=out)

    return out


def _norm_cdf(z, *, out=None, tmp=None):
    """Normal CDF using erf approximation with output reuse."""
    z = cp.asarray(z, dtype=cp.float64)
    if out is None:
        out = cp.empty_like(z)
    if tmp is None:
        tmp = cp.empty_like(z)

    cp.divide(z, _SQRT2, out=tmp)
    _erf_approx(tmp, out=tmp)
    cp.add(tmp, 1.0, out=tmp)
    cp.multiply(tmp, 0.5, out=out)
    return out


def _norm_pdf(z, *, out=None):
    """Normal PDF with output reuse."""
    z = cp.asarray(z, dtype=cp.float64)
    if out is None:
        out = cp.empty_like(z)

    cp.multiply(z, z, out=out)
    cp.multiply(out, -0.5, out=out)
    cp.exp(out, out=out)
    cp.multiply(out, _INV_SQRT2PI, out=out)
    return out


def ei_bounds(mu_lo, mu_hi, sig_lo, sig_hi, n, y_min_unscaled, *, validation=True, pad=1e-5):
    """Memory-optimized cuPyNumeric implementation for EI interval bounds."""
    mu_lo = cp.asarray(mu_lo, dtype=cp.float64)
    mu_hi = cp.asarray(mu_hi, dtype=cp.float64)
    sig_lo = cp.asarray(sig_lo, dtype=cp.float64)
    sig_hi = cp.asarray(sig_hi, dtype=cp.float64)

    if validation:
        if mu_lo.shape != (n,) or mu_hi.shape != (n,):
            raise ValueError("mu_lo and mu_hi must have shape (n,).")
        if sig_lo.shape != (n,) or sig_hi.shape != (n,):
            raise ValueError("sig_lo and sig_hi must have shape (n,).")

    N_lo = cp.empty(n, dtype=cp.float64)
    N_hi = cp.empty(n, dtype=cp.float64)
    cp.subtract(y_min_unscaled, mu_hi, out=N_lo)
    cp.subtract(y_min_unscaled, mu_lo, out=N_hi)

    mask_sig_lo_0 = sig_lo == 0.0
    mask_ei_0 = sig_hi == 0.0

    sig_lo_safe = cp.empty_like(sig_lo)
    sig_hi_safe = cp.empty_like(sig_hi)
    sig_lo_safe[...] = sig_lo
    sig_hi_safe[...] = sig_hi
    sig_lo_safe[mask_sig_lo_0] = pad
    sig_hi_safe[mask_ei_0] = pad

    J_lo = cp.empty_like(sig_hi_safe)
    J_hi = cp.empty_like(sig_lo_safe)
    cp.divide(1.0, sig_hi_safe, out=J_lo)
    cp.divide(1.0, sig_lo_safe, out=J_hi)

    Z_lo = cp.empty(n, dtype=cp.float64)
    Z_hi = cp.empty(n, dtype=cp.float64)
    tmp = cp.empty(n, dtype=cp.float64)
    _interval_product_bounds(N_lo, N_hi, J_lo, J_hi, Z_lo, Z_hi, tmp)

    Phi_lo = cp.empty(n, dtype=cp.float64)
    Phi_hi = cp.empty(n, dtype=cp.float64)
    _norm_cdf(Z_lo, out=Phi_lo, tmp=tmp)
    _norm_cdf(Z_hi, out=Phi_hi, tmp=tmp)

    # phi bounds from interval [Z_lo, Z_hi]
    phi_lo = cp.empty(n, dtype=cp.float64)
    phi_hi = cp.empty(n, dtype=cp.float64)
    _norm_pdf(Z_lo, out=phi_lo)
    _norm_pdf(Z_hi, out=phi_hi)

    cp.maximum(phi_lo, phi_hi, out=tmp)
    cp.minimum(phi_lo, phi_hi, out=phi_lo)
    phi_hi[...] = tmp

    mask_phi_hi = cp.empty(n, dtype=bool)
    cross_tmp = cp.empty(n, dtype=bool)
    cp.less_equal(Z_lo, 0.0, out=mask_phi_hi)
    cp.greater_equal(Z_hi, 0.0, out=cross_tmp)
    cp.logical_and(mask_phi_hi, cross_tmp, out=mask_phi_hi)
    phi_hi[mask_phi_hi] = _INV_SQRT2PI

    U_lo = cp.empty(n, dtype=cp.float64)
    U_hi = cp.empty(n, dtype=cp.float64)
    _interval_product_bounds(N_lo, N_hi, Phi_lo, Phi_hi, U_lo, U_hi, tmp)

    V_lo = cp.empty(n, dtype=cp.float64)
    V_hi = cp.empty(n, dtype=cp.float64)
    _interval_product_bounds(sig_lo, sig_hi, phi_lo, phi_hi, V_lo, V_hi, tmp)

    EI_lo = U_lo
    EI_hi = U_hi
    cp.add(EI_lo, V_lo, out=EI_lo)
    cp.add(EI_hi, V_hi, out=EI_hi)
    cp.maximum(EI_lo, 0.0, out=EI_lo)
    cp.maximum(EI_hi, 0.0, out=EI_hi)

    cp.logical_or(mask_sig_lo_0, mask_ei_0, out=mask_sig_lo_0)
    EI_lo[mask_sig_lo_0] = 0.0
    EI_hi[mask_ei_0] = 0.0

    return EI_lo, EI_hi
