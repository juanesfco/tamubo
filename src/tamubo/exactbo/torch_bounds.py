from __future__ import annotations

import math

import torch

_SQRT2 = math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)

__all__ = [
    "expected_improvement_torch",
    "ei_bounds_torch",
    "mu_bounds_torch",
    "rbf_k_bounds_torch",
    "sigma_bounds_torch",
]


def rbf_k_bounds_torch(
    bounds_L: torch.Tensor,
    bounds_U: torch.Tensor,
    xi: torch.Tensor,
    sigma_f_squared: float,
    length_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute interval bounds for an RBF kernel over boxes."""
    diff_lo = (bounds_L - xi) / length_scale
    diff_hi = (xi - bounds_U) / length_scale

    d_min = torch.clamp(torch.maximum(diff_lo, diff_hi), min=0.0)
    d_max = torch.maximum(diff_lo.abs(), diff_hi.abs())

    K_lo = sigma_f_squared * torch.exp(-0.5 * (d_max * d_max).sum(dim=-1))
    K_hi = sigma_f_squared * torch.exp(-0.5 * (d_min * d_min).sum(dim=-1))
    return K_lo, K_hi


def mu_bounds_torch(
    alpha: torch.Tensor,
    K_lo: torch.Tensor,
    K_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute interval bounds for the GP posterior mean."""
    alpha_pos = torch.clamp(alpha, min=0.0)
    alpha_neg = torch.clamp(alpha, max=0.0)
    mu_lo = K_lo @ alpha_pos + K_hi @ alpha_neg
    mu_hi = K_hi @ alpha_pos + K_lo @ alpha_neg
    return mu_lo, mu_hi


def sigma_bounds_torch(
    K_lo: torch.Tensor,
    K_hi: torch.Tensor,
    L: torch.Tensor,
    sigma_f_squared: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute interval bounds for the GP posterior standard deviation."""
    n_boxes, n_train = K_lo.shape
    v_lo = torch.empty_like(K_lo)
    v_hi = torch.empty_like(K_hi)

    q_hi = torch.zeros(n_boxes, dtype=K_lo.dtype, device=K_lo.device)
    q_lo = torch.zeros(n_boxes, dtype=K_lo.dtype, device=K_lo.device)

    sum_lo = torch.empty_like(q_hi)
    sum_hi = torch.empty_like(q_hi)

    for j in range(n_train):
        sum_lo.zero_()
        sum_hi.zero_()

        for i in range(j):
            Lji = float(L[j, i].item())
            if Lji >= 0.0:
                sum_lo = sum_lo + Lji * v_lo[:, i]
                sum_hi = sum_hi + Lji * v_hi[:, i]
            else:
                sum_lo = sum_lo + Lji * v_hi[:, i]
                sum_hi = sum_hi + Lji * v_lo[:, i]

        diag = float(L[j, j].item())
        v_lo[:, j] = (K_lo[:, j] - sum_hi) / diag
        v_hi[:, j] = (K_hi[:, j] - sum_lo) / diag

        sq_lo = v_lo[:, j].square()
        sq_hi = v_hi[:, j].square()
        q_hi = q_hi + torch.maximum(sq_lo, sq_hi)

        term_lo = torch.minimum(sq_lo, sq_hi)
        term_lo = torch.where(
            torch.logical_and(v_lo[:, j] < 0.0, v_hi[:, j] > 0.0),
            torch.zeros_like(term_lo),
            term_lo,
        )
        q_lo = q_lo + term_lo

    var_lo = torch.clamp(sigma_f_squared - q_hi, min=1e-12)
    var_hi = torch.clamp(sigma_f_squared - q_lo, min=1e-12)
    return torch.sqrt(var_lo), torch.sqrt(var_hi)


def expected_improvement_torch(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    best_f: float,
) -> torch.Tensor:
    """Analytic expected improvement for minimization."""
    z = (best_f - mu) / sigma
    return (best_f - mu) * _norm_cdf(z) + sigma * _norm_pdf(z)


def ei_bounds_torch(
    mu_lo: torch.Tensor,
    mu_hi: torch.Tensor,
    sig_lo: torch.Tensor,
    sig_hi: torch.Tensor,
    best_f: float,
    *,
    pad: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute interval bounds for expected improvement."""
    N_lo = best_f - mu_hi
    N_hi = best_f - mu_lo

    inv_pad = 1.0 / pad
    J_lo = torch.full_like(sig_hi, inv_pad)
    J_hi = torch.full_like(sig_lo, inv_pad)

    nonzero_hi = sig_hi != 0.0
    nonzero_lo = sig_lo != 0.0
    J_lo = torch.where(nonzero_hi, 1.0 / sig_hi, J_lo)
    J_hi = torch.where(nonzero_lo, 1.0 / sig_lo, J_hi)

    Z_lo, Z_hi = _interval_product_bounds(N_lo, N_hi, J_lo, J_hi)
    Phi_lo = _norm_cdf(Z_lo)
    Phi_hi = _norm_cdf(Z_hi)

    pdf_lo = _norm_pdf(Z_lo)
    pdf_hi = _norm_pdf(Z_hi)
    phi_lo = torch.minimum(pdf_lo, pdf_hi)
    phi_hi = torch.maximum(pdf_lo, pdf_hi)
    phi_hi = torch.where(
        torch.logical_and(Z_lo <= 0.0, Z_hi >= 0.0),
        torch.full_like(phi_hi, _INV_SQRT2PI),
        phi_hi,
    )

    U_lo, U_hi = _interval_product_bounds(N_lo, N_hi, Phi_lo, Phi_hi)
    V_lo, V_hi = _interval_product_bounds(sig_lo, sig_hi, phi_lo, phi_hi)

    EI_lo = torch.clamp(U_lo + V_lo, min=0.0)
    EI_hi = torch.clamp(U_hi + V_hi, min=0.0)

    sigma_hi_zero = sig_hi == 0.0
    sigma_lo_zero = sig_lo == 0.0
    EI_hi = torch.where(sigma_hi_zero, torch.zeros_like(EI_hi), EI_hi)
    EI_lo = torch.where(
        torch.logical_or(sigma_hi_zero, sigma_lo_zero),
        torch.zeros_like(EI_lo),
        EI_lo,
    )
    return EI_lo, EI_hi


def _interval_product_bounds(
    a_lo: torch.Tensor,
    a_hi: torch.Tensor,
    b_lo: torch.Tensor,
    b_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    p0 = a_lo * b_lo
    p1 = a_lo * b_hi
    p2 = a_hi * b_lo
    p3 = a_hi * b_hi
    stacked = torch.stack((p0, p1, p2, p3), dim=0)
    return stacked.min(dim=0).values, stacked.max(dim=0).values


def _norm_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / _SQRT2))


def _norm_pdf(z: torch.Tensor) -> torch.Tensor:
    return _INV_SQRT2PI * torch.exp(-0.5 * z.square())
