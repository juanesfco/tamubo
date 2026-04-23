from __future__ import annotations

from dataclasses import dataclass
import math
import time as pytime

import torch

from .torch_bounds import (
    ei_bounds_torch,
    expected_improvement_torch,
    mu_bounds_torch,
    rbf_k_bounds_torch,
    sigma_bounds_torch,
)

__all__ = [
    "ExactBOTorchPartitionResult",
    "TorchGPState",
    "exactbo_torch_partitioning",
    "extract_torch_gp_state",
]


@dataclass(frozen=True)
class TorchGPState:
    X_train: torch.Tensor
    alpha: torch.Tensor
    L: torch.Tensor
    length_scale: torch.Tensor
    sigma_f_squared: float
    sigma_n_squared: float
    best_f_scaled: float
    y_train_mean: float
    y_train_std: float


@dataclass(frozen=True)
class ExactBOTorchPartitionResult:
    candidate: torch.Tensor
    ei_value_scaled: torch.Tensor
    log: dict | None = None


def extract_torch_gp_state(model, best_f) -> TorchGPState:
    """Extract the ExactBO state from a fitted BoTorch single-output GP."""
    if not hasattr(model, "train_inputs") or len(model.train_inputs) != 1:
        raise TypeError("ExactBO torch partitioning expects a single-input BoTorch GP model.")
    if not hasattr(model, "covar_module") or not hasattr(model, "likelihood"):
        raise TypeError("ExactBO torch partitioning expects a BoTorch GP model with covariance and likelihood modules.")

    train_X = model.train_inputs[0].detach()
    train_Y = model.train_targets.detach()
    if train_Y.ndim == 2 and train_Y.shape[-1] == 1:
        train_Y = train_Y.squeeze(-1)
    if train_Y.ndim != 1:
        raise TypeError("ExactBO torch partitioning supports only single-output GP models.")

    covar_module = model.covar_module
    base_kernel = getattr(covar_module, "base_kernel", None)
    if base_kernel is None or not hasattr(base_kernel, "lengthscale"):
        raise TypeError("ExactBO torch partitioning currently requires a ScaleKernel(RBFKernel)-style covariance module.")
    if not hasattr(covar_module, "outputscale"):
        raise TypeError("ExactBO torch partitioning requires a covariance module with outputscale.")
    if not hasattr(model.likelihood, "noise"):
        raise TypeError("ExactBO torch partitioning requires a Gaussian likelihood with scalar noise.")

    dtype = train_X.dtype
    device = train_X.device
    length_scale = base_kernel.lengthscale.detach().reshape(-1).to(dtype=dtype, device=device)
    sigma_f_squared = float(covar_module.outputscale.detach().reshape(-1)[0].item())
    sigma_n_squared = float(model.likelihood.noise.detach().reshape(-1)[0].item())

    y_train_mean = 0.0
    y_train_std = 1.0
    best_f_scaled = float(best_f)
    outcome_transform = getattr(model, "outcome_transform", None)
    if outcome_transform is not None and hasattr(outcome_transform, "means") and hasattr(outcome_transform, "stdvs"):
        means = outcome_transform.means.detach().reshape(-1).to(dtype=dtype, device=device)
        stdvs = outcome_transform.stdvs.detach().reshape(-1).to(dtype=dtype, device=device)
        if means.numel() != 1 or stdvs.numel() != 1:
            raise TypeError("ExactBO torch partitioning supports only single-output standardized outcome transforms.")
        y_train_mean = float(means[0].item())
        y_train_std = float(stdvs[0].item())
        best_tensor = torch.as_tensor(best_f, dtype=dtype, device=device)
        best_f_scaled = float(((best_tensor - means[0]) / stdvs[0]).item())

    K = _rbf_kernel(train_X, train_X, length_scale, sigma_f_squared)
    diag_idx = torch.arange(train_X.shape[0], device=device)
    K = K.clone()
    K[diag_idx, diag_idx] += sigma_n_squared + 1e-10
    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(train_Y.unsqueeze(-1), L).squeeze(-1)

    return TorchGPState(
        X_train=train_X.to(dtype=dtype, device=device),
        alpha=alpha.to(dtype=dtype, device=device),
        L=L.to(dtype=dtype, device=device),
        length_scale=length_scale,
        sigma_f_squared=sigma_f_squared,
        sigma_n_squared=sigma_n_squared,
        best_f_scaled=best_f_scaled,
        y_train_mean=y_train_mean,
        y_train_std=y_train_std,
    )


def exactbo_torch_partitioning(
    model,
    bounds: torch.Tensor,
    epsilon_X,
    epsilon_ei: float,
    best_f,
    max_partitions: int,
    *,
    validation: bool = True,
    verbose: bool = False,
    logMask: bool = False,
) -> ExactBOTorchPartitionResult:
    """Run one ExactBO partitioning step on a fitted BoTorch GP model."""
    state = extract_torch_gp_state(model, best_f)
    lower, upper = _normalize_bounds(bounds, state.X_train)
    dim = lower.shape[-1]

    partitions = int(max_partitions)
    if partitions <= 0:
        raise ValueError(f"max_partitions must be > 0, got {partitions}")

    epsilon_X = _normalize_epsilon_torch(epsilon_X, dim, reference=state.X_train)
    bounds_L = lower.unsqueeze(0)
    bounds_U = upper.unsqueeze(0)
    domain_width = upper - lower

    lhs_points_per_box = int(2**dim)
    lhs_unit_design = _centered_latin_hypercube_unit_torch(
        lhs_points_per_box,
        dim,
        dtype=state.X_train.dtype,
        device=state.X_train.device,
    )

    stride = 2 * dim + 1
    partition = 0
    n_total = 1
    n_target_start = 1
    idx_best_global = 0
    idx_best_global_next = 0
    best_x = None
    best_ei_scaled = 0.0

    log = {} if logMask else None
    start_time = pytime.perf_counter() if logMask else None

    while partition < partitions:
        if partition > 0:
            n_target_start = n_target * stride
            idx_best_global_start = idx_best_global_next * stride
            preserved_analyze_idx = torch.arange(
                idx_best_global_start,
                idx_best_global_start + stride,
                device=state.X_train.device,
            )
        else:
            preserved_analyze_idx = torch.tensor([0], device=state.X_train.device)

        bounds_L_target = bounds_L[:n_target_start]
        bounds_U_target = bounds_U[:n_target_start]

        if verbose:
            print(
                f"Partition {partition}/{partitions - 1}, "
                f"Boxes: {n_total}, Target boxes to analyze: {n_target_start}."
            )

        ei_hi = _compute_ei_upper_bounds_torch(bounds_L_target, bounds_U_target, state)
        idx_max_ei_hi = _scalar_int(torch.argmax(ei_hi))
        max_ei_hi = _scalar_float(ei_hi[idx_max_ei_hi])

        analyze_box_mask = ei_hi >= (max_ei_hi - epsilon_ei)
        analyze_box_mask[preserved_analyze_idx] = True
        analyze_local_idx = torch.where(analyze_box_mask)[0]
        n_analyze = _scalar_int(analyze_local_idx.shape[0])

        analyze_best_points, ei_analyze = _sample_boxes_best_ei_torch(
            bounds_L_target[analyze_local_idx],
            bounds_U_target[analyze_local_idx],
            state,
            lhs_unit_design,
        )
        idx_ei_max_analyze = _scalar_int(torch.argmax(ei_analyze))
        ei_max_analyze = _scalar_float(ei_analyze[idx_ei_max_analyze])
        idx_ei_max_analyze_local = _scalar_int(analyze_local_idx[idx_ei_max_analyze])
        best_x_analyze = analyze_best_points[idx_ei_max_analyze]
        best_x = best_x_analyze
        best_ei_scaled = ei_max_analyze

        w_max_ei_analyzed = bounds_U_target[idx_ei_max_analyze_local] - bounds_L_target[idx_ei_max_analyze_local]
        active_boxes_mask = ei_hi > (ei_max_analyze + epsilon_ei)
        n_active = _scalar_int(active_boxes_mask.sum())

        if n_active == 0 and _scalar_bool(torch.all(w_max_ei_analyzed < epsilon_X)):
            return _build_partition_result(best_x_analyze, ei_max_analyze, log, start_time, state)

        active_boxes_mask[idx_ei_max_analyze_local] = True
        active_local_idx = torch.where(active_boxes_mask)[0]
        n_active = _scalar_int(active_local_idx.shape[0])

        active_best_points, ei_active = _sample_boxes_best_ei_torch(
            bounds_L_target[active_local_idx],
            bounds_U_target[active_local_idx],
            state,
            lhs_unit_design,
        )
        idx_best = _scalar_int(torch.argmax(ei_active))
        ei_max_active = _scalar_float(ei_active[idx_best])
        idx_best_local = _scalar_int(active_local_idx[idx_best])
        best_x_active = active_best_points[idx_best]
        best_x = best_x_active
        best_ei_scaled = ei_max_active

        w_max_ei_active = bounds_U_target[idx_best_local] - bounds_L_target[idx_best_local]
        target_boxes_mask = ei_hi > (ei_max_active + epsilon_ei)
        n_target = _scalar_int(target_boxes_mask.sum())

        if n_target == 0 and _scalar_bool(torch.all(w_max_ei_active < epsilon_X)):
            return _build_partition_result(best_x_active, ei_max_active, log, start_time, state)

        idx_best_global = idx_best_local
        target_boxes_mask[idx_best_global] = True
        n_target = _scalar_int(target_boxes_mask.sum())
        idx_best_global_next = _scalar_int(target_boxes_mask[:idx_best_global].sum())

        if verbose:
            target_width = (bounds_U_target[target_boxes_mask] - bounds_L_target[target_boxes_mask]).amax(dim=0)
            print(
                f"  Analyzed: {n_analyze}, Active: {n_active}, Target: {n_target}, "
                f"Max EI_hi: {max_ei_hi:.6f}, Max EI Active: {ei_max_active:.6f}, "
                f"Max Width: {target_width}."
            )

        partition += 1
        n_total += n_target * (2 * dim)

        if partition < partitions:
            bounds_L, bounds_U = _split_boxes_torch(
                bounds_L_target,
                bounds_U_target,
                target_boxes_mask,
                domain_width,
            )

    return _build_partition_result(best_x, best_ei_scaled, log, start_time, state)


def _normalize_bounds(bounds: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bounds_t = torch.as_tensor(bounds, dtype=reference.dtype, device=reference.device)
    if bounds_t.ndim != 2:
        raise ValueError(f"bounds must be 2D, got shape {tuple(bounds_t.shape)}")

    if bounds_t.shape[0] == 2:
        lower = bounds_t[0]
        upper = bounds_t[1]
    elif bounds_t.shape[1] == 2:
        lower = bounds_t[:, 0]
        upper = bounds_t[:, 1]
    else:
        raise ValueError("bounds must have shape (2, d) or (d, 2).")

    if lower.shape != upper.shape:
        raise ValueError("lower and upper bounds must have matching shapes.")
    return lower, upper


def _normalize_epsilon_torch(epsilon, dim: int, *, reference: torch.Tensor) -> torch.Tensor:
    eps = torch.as_tensor(epsilon, dtype=reference.dtype, device=reference.device)
    if eps.ndim == 0:
        return torch.full((dim,), float(eps.item()), dtype=reference.dtype, device=reference.device)
    if tuple(eps.shape) == (dim,):
        return eps
    raise ValueError(f"epsilon_X must be scalar or shape ({dim},), got {tuple(eps.shape)}")


def _centered_latin_hypercube_unit_torch(
    n_points: int,
    dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    centers = (torch.arange(n_points, dtype=dtype, device=device) + 0.5) / float(n_points)
    perm_ids = torch.arange(n_points, dtype=torch.long, device=device)
    lhs = torch.empty((n_points, dim), dtype=dtype, device=device)

    for j in range(dim):
        step = 2 * j + 1
        while math.gcd(step, n_points) != 1:
            step += 2
        lhs[:, j] = centers[(perm_ids * step + j) % n_points]

    return lhs


def _rbf_kernel(
    X: torch.Tensor,
    Y: torch.Tensor,
    length_scale: torch.Tensor,
    sigma_f_squared: float,
) -> torch.Tensor:
    X_scaled = X / length_scale
    Y_scaled = Y / length_scale
    X_sq = (X_scaled * X_scaled).sum(dim=1, keepdim=True)
    Y_sq = (Y_scaled * Y_scaled).sum(dim=1, keepdim=True).transpose(-2, -1)
    sq_dists = torch.clamp(X_sq + Y_sq - 2.0 * (X_scaled @ Y_scaled.transpose(-2, -1)), min=0.0)
    return sigma_f_squared * torch.exp(-0.5 * sq_dists)


def _predict_standardized_posterior(points: torch.Tensor, state: TorchGPState) -> tuple[torch.Tensor, torch.Tensor]:
    K_trans = _rbf_kernel(points, state.X_train, state.length_scale, state.sigma_f_squared)
    mu = K_trans @ state.alpha
    V = torch.linalg.solve_triangular(state.L, K_trans.transpose(-2, -1), upper=False)
    var = state.sigma_f_squared + state.sigma_n_squared - (V * V).sum(dim=0)
    std = torch.sqrt(torch.clamp(var, min=0.0))
    return mu, std


def _compute_ei_upper_bounds_torch(
    bounds_L_target: torch.Tensor,
    bounds_U_target: torch.Tensor,
    state: TorchGPState,
) -> torch.Tensor:
    n_target = bounds_L_target.shape[0]
    n_train = state.alpha.shape[0]

    K_lo = torch.empty((n_target, n_train), dtype=bounds_L_target.dtype, device=bounds_L_target.device)
    K_hi = torch.empty_like(K_lo)
    for i in range(n_train):
        K_lo[:, i], K_hi[:, i] = rbf_k_bounds_torch(
            bounds_L_target,
            bounds_U_target,
            state.X_train[i],
            state.sigma_f_squared,
            state.length_scale,
        )

    mu_lo, mu_hi = mu_bounds_torch(state.alpha, K_lo, K_hi)
    sig_lo, sig_hi = sigma_bounds_torch(K_lo, K_hi, state.L, state.sigma_f_squared)
    _, ei_hi = ei_bounds_torch(mu_lo, mu_hi, sig_lo, sig_hi, state.best_f_scaled)
    return ei_hi


def _sample_boxes_best_ei_torch(
    boxes_L: torch.Tensor,
    boxes_U: torch.Tensor,
    state: TorchGPState,
    lhs_unit_design: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes_L.shape[0] == 0:
        return (
            torch.empty((0, boxes_L.shape[-1]), dtype=boxes_L.dtype, device=boxes_L.device),
            torch.empty((0,), dtype=boxes_L.dtype, device=boxes_L.device),
        )

    widths = boxes_U - boxes_L
    sampled_points = boxes_L[:, None, :] + lhs_unit_design[None, :, :] * widths[:, None, :]
    flat_points = sampled_points.reshape(-1, boxes_L.shape[-1])

    mu, sigma = _predict_standardized_posterior(flat_points, state)
    mu = mu.reshape(boxes_L.shape[0], lhs_unit_design.shape[0])
    sigma = sigma.reshape(boxes_L.shape[0], lhs_unit_design.shape[0])

    sigma_latent = torch.sqrt(torch.clamp(sigma.square() - state.sigma_n_squared, min=1e-12))
    ei = expected_improvement_torch(mu, sigma_latent, state.best_f_scaled)
    best_idx = torch.argmax(ei, dim=1, keepdim=True)
    best_ei = torch.take_along_dim(ei, best_idx, dim=1).reshape(-1)
    best_points = torch.take_along_dim(
        sampled_points,
        best_idx.unsqueeze(-1).expand(-1, 1, boxes_L.shape[-1]),
        dim=1,
    ).reshape(boxes_L.shape[0], boxes_L.shape[-1])
    return best_points, best_ei


def _split_boxes_torch(
    bounds_L: torch.Tensor,
    bounds_U: torch.Tensor,
    active_boxes_mask: torch.Tensor,
    domain_width: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    active_bounds_L = bounds_L[active_boxes_mask]
    active_bounds_U = bounds_U[active_boxes_mask]
    n_active, dim = active_bounds_L.shape

    stride = 2 * dim + 1
    split_bounds_L = active_bounds_L[:, None, :].repeat(1, stride, 1)
    split_bounds_U = active_bounds_U[:, None, :].repeat(1, stride, 1)

    active_width = active_bounds_U - active_bounds_L
    split_order = torch.argsort(active_width / domain_width, dim=1, descending=True)
    dim_ids = torch.arange(dim, device=bounds_L.device)[None, :]

    for rank in range(dim):
        cols = split_order[:, rank : rank + 1]
        third_width = torch.take_along_dim(active_width, cols, dim=1) / 3.0
        lower_third = torch.take_along_dim(active_bounds_L, cols, dim=1) + third_width
        upper_third = torch.take_along_dim(active_bounds_U, cols, dim=1) - third_width
        dim_mask = cols == dim_ids

        lower_row = 2 * rank
        upper_row = lower_row + 1
        split_bounds_U[:, lower_row, :] = torch.where(
            dim_mask,
            lower_third,
            split_bounds_U[:, lower_row, :],
        )
        split_bounds_L[:, upper_row, :] = torch.where(
            dim_mask,
            upper_third,
            split_bounds_L[:, upper_row, :],
        )

        if upper_row + 1 < stride:
            tail_mask = dim_mask[:, None, :]
            split_bounds_L[:, upper_row + 1 :, :] = torch.where(
                tail_mask,
                lower_third[:, None, :],
                split_bounds_L[:, upper_row + 1 :, :],
            )
            split_bounds_U[:, upper_row + 1 :, :] = torch.where(
                tail_mask,
                upper_third[:, None, :],
                split_bounds_U[:, upper_row + 1 :, :],
            )

    return split_bounds_L.reshape(n_active * stride, dim), split_bounds_U.reshape(n_active * stride, dim)


def _build_partition_result(
    best_x: torch.Tensor,
    ei_best_scaled: float,
    log: dict | None,
    start_time: float | None,
    state: TorchGPState,
) -> ExactBOTorchPartitionResult:
    if log is not None and start_time is not None:
        log["time"] = pytime.perf_counter() - start_time
        log["ei_max"] = float(ei_best_scaled * state.y_train_std)
        log["ei_max_scaled"] = float(ei_best_scaled)

    return ExactBOTorchPartitionResult(
        candidate=best_x.detach().clone(),
        ei_value_scaled=torch.as_tensor(
            ei_best_scaled,
            dtype=best_x.dtype,
            device=best_x.device,
        ),
        log=log,
    )


def _scalar_int(value) -> int:
    return int(value.item())


def _scalar_float(value) -> float:
    return float(value.item())


def _scalar_bool(value) -> bool:
    return bool(value.item())
