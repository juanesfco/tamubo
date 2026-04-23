from __future__ import annotations

import math

from tamubo.utils import BackendName

from ._cupynumeric import cp, require_cupynumeric_backend

_SQRT2 = math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


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
    """Compute interval bounds for the RBF kernel over ``n`` boxes."""
    require_cupynumeric_backend(backend)

    bounds_L = cp.asarray(bounds_L, dtype=cp.float64)
    bounds_U = cp.asarray(bounds_U, dtype=cp.float64)
    xi = cp.asarray(xi, dtype=cp.float64)
    length_scale = cp.asarray(length_scale, dtype=cp.float64)

    if validation:
        if bounds_L.shape != (n, d) or bounds_U.shape != (n, d):
            raise ValueError("bounds_L and bounds_U must have shape (n, d).")
        if xi.size != d:
            raise ValueError("xi must have size d.")

    diff_lo = (bounds_L - xi) / length_scale
    diff_hi = (xi - bounds_U) / length_scale

    d_min = cp.maximum(cp.maximum(diff_lo, diff_hi), 0.0)
    d_max = cp.maximum(cp.abs(diff_lo), cp.abs(diff_hi))

    K_lo = sigma_f_2 * cp.exp(-0.5 * cp.sum(d_max * d_max, axis=1))
    K_hi = sigma_f_2 * cp.exp(-0.5 * cp.sum(d_min * d_min, axis=1))
    return K_lo, K_hi


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
    """Compute interval bounds for the GP posterior mean."""
    require_cupynumeric_backend(backend)

    alpha = cp.asarray(alpha, dtype=cp.float64)
    K_lo = cp.asarray(K_lo, dtype=cp.float64)
    K_hi = cp.asarray(K_hi, dtype=cp.float64)

    if validation:
        if alpha.shape != (N,):
            raise ValueError("alpha must have shape (N,).")
        if K_lo.shape != (n, N) or K_hi.shape != (n, N):
            raise ValueError("K_lo and K_hi must have shape (n, N).")

    alpha_pos = cp.maximum(alpha, 0.0)
    alpha_neg = cp.minimum(alpha, 0.0)

    mu_lo = K_lo @ alpha_pos + K_hi @ alpha_neg
    mu_hi = K_hi @ alpha_pos + K_lo @ alpha_neg

    if scaled_output:
        return mu_lo, mu_hi

    return y_train_mean + y_train_std * mu_lo, y_train_mean + y_train_std * mu_hi


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
) -> tuple:
    """Compute interval bounds for the GP posterior standard deviation."""
    require_cupynumeric_backend(backend)

    K_lo = cp.asarray(K_lo, dtype=cp.float64)
    K_hi = cp.asarray(K_hi, dtype=cp.float64)
    L = cp.asarray(L, dtype=cp.float64)

    if validation:
        if K_lo.shape != (n, N) or K_hi.shape != (n, N):
            raise ValueError("K_lo and K_hi must have shape (n, N).")
        if L.shape != (N, N):
            raise ValueError("L must have shape (N, N).")

    v_lo = cp.empty_like(K_lo)
    v_hi = cp.empty_like(K_hi)

    q_hi = cp.zeros((n,), dtype=cp.float64)
    q_lo = cp.zeros((n,), dtype=cp.float64)

    sum_lo = cp.empty((n,), dtype=cp.float64)
    sum_hi = cp.empty((n,), dtype=cp.float64)
    sq_lo = cp.empty((n,), dtype=cp.float64)
    sq_hi = cp.empty((n,), dtype=cp.float64)
    cross_zero = cp.empty((n,), dtype=bool)

    for j in range(N):
        sum_lo[...] = 0.0
        sum_hi[...] = 0.0

        for i in range(j):
            Lji = float(L[j, i])
            if Lji >= 0.0:
                sum_lo += Lji * v_lo[:, i]
                sum_hi += Lji * v_hi[:, i]
            else:
                sum_lo += Lji * v_hi[:, i]
                sum_hi += Lji * v_lo[:, i]

        diag = float(L[j, j])
        v_lo[:, j] = (K_lo[:, j] - sum_hi) / diag
        v_hi[:, j] = (K_hi[:, j] - sum_lo) / diag

        sq_lo[...] = v_lo[:, j] * v_lo[:, j]
        sq_hi[...] = v_hi[:, j] * v_hi[:, j]

        q_hi += cp.maximum(sq_lo, sq_hi)
        sq_lo[...] = cp.minimum(sq_lo, sq_hi)

        cross_zero[...] = cp.logical_and(v_lo[:, j] < 0.0, v_hi[:, j] > 0.0)
        sq_lo[cross_zero] = 0.0
        q_lo += sq_lo

    var_lo = cp.maximum(sigma_f_2 - q_hi, 1e-12)
    var_hi = cp.maximum(sigma_f_2 - q_lo, 1e-12)
    sig_lo = cp.sqrt(var_lo)
    sig_hi = cp.sqrt(var_hi)

    if scaled_output:
        return sig_lo, sig_hi

    return y_train_std * sig_lo, y_train_std * sig_hi


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
) -> tuple:
    """Compute interval bounds for expected improvement."""
    require_cupynumeric_backend(backend)

    mu_lo = cp.asarray(mu_lo, dtype=cp.float64)
    mu_hi = cp.asarray(mu_hi, dtype=cp.float64)
    sig_lo = cp.asarray(sig_lo, dtype=cp.float64)
    sig_hi = cp.asarray(sig_hi, dtype=cp.float64)

    if validation:
        if mu_lo.shape != (n,) or mu_hi.shape != (n,):
            raise ValueError("mu_lo and mu_hi must have shape (n,).")
        if sig_lo.shape != (n,) or sig_hi.shape != (n,):
            raise ValueError("sig_lo and sig_hi must have shape (n,).")

    N_lo = y_min - mu_hi
    N_hi = y_min - mu_lo

    inv_pad = 1.0 / pad
    J_lo = cp.full((n,), inv_pad, dtype=cp.float64)
    J_hi = cp.full((n,), inv_pad, dtype=cp.float64)

    nonzero_hi = sig_hi != 0.0
    nonzero_lo = sig_lo != 0.0
    J_lo[nonzero_hi] = 1.0 / sig_hi[nonzero_hi]
    J_hi[nonzero_lo] = 1.0 / sig_lo[nonzero_lo]

    Z_lo = cp.empty((n,), dtype=cp.float64)
    Z_hi = cp.empty((n,), dtype=cp.float64)
    tmp = cp.empty((n,), dtype=cp.float64)
    _interval_product_bounds(N_lo, N_hi, J_lo, J_hi, Z_lo, Z_hi, tmp)

    Phi_lo = _norm_cdf(Z_lo)
    Phi_hi = _norm_cdf(Z_hi)

    pdf_lo = _norm_pdf(Z_lo)
    pdf_hi = _norm_pdf(Z_hi)
    phi_lo = cp.minimum(pdf_lo, pdf_hi)
    phi_hi = cp.maximum(pdf_lo, pdf_hi)
    phi_hi[cp.logical_and(Z_lo <= 0.0, Z_hi >= 0.0)] = _INV_SQRT2PI

    U_lo = cp.empty((n,), dtype=cp.float64)
    U_hi = cp.empty((n,), dtype=cp.float64)
    V_lo = cp.empty((n,), dtype=cp.float64)
    V_hi = cp.empty((n,), dtype=cp.float64)

    _interval_product_bounds(N_lo, N_hi, Phi_lo, Phi_hi, U_lo, U_hi, tmp)
    _interval_product_bounds(sig_lo, sig_hi, phi_lo, phi_hi, V_lo, V_hi, tmp)

    EI_lo = cp.maximum(U_lo + V_lo, 0.0)
    EI_hi = cp.maximum(U_hi + V_hi, 0.0)

    sigma_hi_zero = sig_hi == 0.0
    sigma_lo_zero = sig_lo == 0.0
    EI_hi[sigma_hi_zero] = 0.0
    EI_lo[cp.logical_or(sigma_hi_zero, sigma_lo_zero)] = 0.0
    return EI_lo, EI_hi


def _interval_product_bounds(a_lo, a_hi, b_lo, b_hi, out_lo, out_hi, tmp) -> None:
    out_lo[...] = a_lo * b_lo
    out_hi[...] = out_lo

    tmp[...] = a_lo * b_hi
    out_lo[...] = cp.minimum(out_lo, tmp)
    out_hi[...] = cp.maximum(out_hi, tmp)

    tmp[...] = a_hi * b_lo
    out_lo[...] = cp.minimum(out_lo, tmp)
    out_hi[...] = cp.maximum(out_hi, tmp)

    tmp[...] = a_hi * b_hi
    out_lo[...] = cp.minimum(out_lo, tmp)
    out_hi[...] = cp.maximum(out_hi, tmp)


def _erf_approx(x, *, out=None):
    """Abramowitz-Stegun 7.1.26 approximation with max error below 1.5e-7."""
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
    ax = cp.abs(x)
    t = 1.0 / (1.0 + p * ax)
    poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
    out[...] = sign * (1.0 - poly * cp.exp(-(ax * ax)))
    return out


def _norm_cdf(z):
    z = cp.asarray(z, dtype=cp.float64)
    return 0.5 * (1.0 + _erf_approx(z / _SQRT2))


def _norm_pdf(z):
    z = cp.asarray(z, dtype=cp.float64)
    return _INV_SQRT2PI * cp.exp(-0.5 * z * z)
