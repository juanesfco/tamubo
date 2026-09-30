from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable
import math
import time as pytime

import numpy as np

from tamubo.utils import (
    BOResult,
    BackendName,
    _evaluate_objective,
    _from_unit_cube,
    _init_log,
    _normalize_problem_to_unit_cube,
    get_array_module as _array_module,
    resolve_backend,
    to_numpy,
)
from tamubo.acquisition_functions import expected_improvement, log_expected_improvement
from tamubo.gpugp.posterior import gp_posterior
from .bounds import rbf_k_bounds, mu_bounds, sigma_bounds, sigma_bound_factors, ei_bounds
from .partition import split_boxes

def _normalize_epsilon(epsilon: np.ndarray | float, dim: int) -> np.ndarray:
    """Normalize epsilon to a per-dimension array."""
    eps = np.asarray(epsilon, dtype=float)
    if eps.ndim == 0:
        return np.full((dim,), float(eps), dtype=float)
    if eps.shape == (dim,):
        return eps
    raise ValueError(f"epsilon must be scalar or shape ({dim},), got {eps.shape}")

def _get_timer(xp):
    """Return a zero-argument clock in seconds that waits for pending GPU work."""
    if xp is np:
        return pytime.perf_counter
    stream = xp.cuda.get_current_stream()

    def cupy_now() -> float:
        # Kernels launch asynchronously; sync so elapsed time covers them.
        stream.synchronize()
        return pytime.perf_counter()

    return cupy_now


@dataclass
class _GPState:
    """Trained-GP arrays used by the per-box work, resident on one device."""

    device: int | None  # None for numpy (host)
    X_train: Any
    alpha: Any
    L: Any
    L_inv: Any  # L^{-1}, for the sigma bounds
    K_inv: Any  # (K + sigma_n^2 I)^{-1}, for the AutoBound variance bounds
    length_scale: Any
    unit_design: Any


def _resolve_devices(xp, n_gpus: int | None) -> list[int | None]:
    """Device ids to spread per-box work over; the current device comes first."""
    if xp is np:
        return [None]
    available = xp.cuda.runtime.getDeviceCount()
    n = available if n_gpus is None else int(n_gpus)
    if not 1 <= n <= available:
        raise ValueError(f"n_gpus must be between 1 and {available} (visible GPUs), got {n_gpus}.")
    main = xp.cuda.Device().id
    return [main] + [i for i in range(available) if i != main][: n - 1]


def _build_gp_state(
    xp, device: int | None, X: np.ndarray, gp, unit_design: np.ndarray, L_inv: np.ndarray
) -> _GPState:
    """Copy the trained GP's arrays to `device` (built from host arrays, no peer copies)."""
    params = gp.kernel_.get_params()
    with (xp.cuda.Device(device) if device is not None else nullcontext()):
        return _GPState(
            device=device,
            X_train=xp.asarray(X, dtype=xp.float64),
            alpha=xp.asarray(gp.alpha_, dtype=xp.float64).reshape(-1),
            L=xp.asarray(gp.L_, dtype=xp.float64),
            L_inv=xp.asarray(L_inv, dtype=xp.float64),
            K_inv=xp.asarray(L_inv.T @ L_inv, dtype=xp.float64),
            length_scale=xp.asarray(params["k1__k2__length_scale"], dtype=xp.float64),
            unit_design=xp.asarray(unit_design, dtype=xp.float64),
        )


# Least GPU work (boxes x N^2 x points per box) worth giving an extra GPU. Each
# extra GPU costs a few ms of host time per call (thread, peer copies and its own
# Python kernel launches, serialized by the GIL); ~1e10 units is ~70 ms of H200
# time, so smaller calls stay on fewer GPUs.
_MIN_WORK_PER_GPU = 1e10

# One persistent worker thread per extra GPU. cuBLAS/cuSOLVER handles are per
# thread and allocate device memory outside cupy's pool, so they must be created
# once, up front: created lazily in a fresh thread, they can fail
# (CUSOLVER_STATUS_INTERNAL_ERROR) once the pool has cached the device's memory.
_GPU_WORKERS: dict[int, ThreadPoolExecutor] = {}


def _create_library_handles(xp, device: int) -> None:
    xp.cuda.Device(device).use()
    xp.cuda.device.get_cublas_handle()
    xp.cuda.device.get_cusolver_handle()


def _gpu_worker(xp, device: int) -> ThreadPoolExecutor:
    worker = _GPU_WORKERS.get(device)
    if worker is None:
        worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"exactbo-gpu{device}")
        worker.submit(_create_library_handles, xp, device).result()
        _GPU_WORKERS[device] = worker
    return worker


def _prepare_devices(xp, states: list[_GPState]) -> None:
    """Create library handles for every device before large allocations begin."""
    if xp is np:
        return
    with xp.cuda.Device(states[0].device):
        xp.cuda.device.get_cublas_handle()
        xp.cuda.device.get_cusolver_handle()
    for state in states[1:]:
        _gpu_worker(xp, state.device)


def _run_sharded(xp, fn, states: list[_GPState], boxes_L, boxes_U, work_per_box: float) -> tuple:
    """
    Run ``fn(state, boxes_L, boxes_U) -> tuple of per-box arrays`` with the
    boxes split row-wise into contiguous shards over as many devices as the
    call's work justifies, and return the outputs concatenated in the original
    box order on the first device.

    The first device's shard runs in the calling thread and every other shard
    in that device's persistent worker thread, so host-side syncs inside ``fn``
    on one device do not stall the others.
    """
    n = int(boxes_L.shape[0])
    n_devices = min(len(states), n, int(n * work_per_box // _MIN_WORK_PER_GPU))
    if n_devices <= 1:
        return fn(states[0], boxes_L, boxes_U)
    states = states[:n_devices]

    main = states[0].device
    edges = np.linspace(0, n, len(states) + 1).astype(np.int64)
    # Shards are read from the main device by the workers; finish its pending work first.
    xp.cuda.Device(main).synchronize()

    def work(k: int) -> tuple:
        state = states[k]
        start, end = int(edges[k]), int(edges[k + 1])
        with xp.cuda.Device(state.device):
            # ndarray.copy() places the copy on the current device, even across devices.
            out = fn(state, boxes_L[start:end].copy(), boxes_U[start:end].copy())
            xp.cuda.Device(state.device).synchronize()
        return out

    futures = [_gpu_worker(xp, states[k].device).submit(work, k) for k in range(1, len(states))]
    shard_outputs = [work(0)] + [future.result() for future in futures]

    gathered = tuple(
        xp.concatenate([out[i].copy() for out in shard_outputs])
        for i in range(len(shard_outputs[0]))
    )
    # The peer copies read worker memory; keep it alive until they complete.
    xp.cuda.Device(main).synchronize()
    return gathered


def _centered_latin_hypercube_unit(n_points: int, dim: int) -> np.ndarray:
    """
    Return a deterministic centered Latin hypercube in ``[0, 1]^dim``.

    The construction is deterministic so ExactBO remains reproducible while
    still spreading the ``2**d`` intra-box probes across each box interior.
    """
    if n_points <= 0:
        raise ValueError(f"n_points must be positive, got {n_points}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    centers = (np.arange(n_points, dtype=np.float64) + 0.5) / float(n_points)
    perm_ids = np.arange(n_points, dtype=np.int64)
    lhs = np.empty((n_points, dim), dtype=np.float64)

    for j in range(dim):
        step = 2 * j + 1
        while math.gcd(step, n_points) != 1:
            step += 2
        lhs[:, j] = centers[(perm_ids * step + j) % n_points]

    return lhs


def exactbo(
    X0: np.ndarray,
    bounds: np.ndarray,
    epsilon_X: np.ndarray | float,
    epsilon_ei: float,
    gp,
    f: Callable[[np.ndarray], np.ndarray],
    max_iters: int,
    max_partitions: int,
    *,
    backend: BackendName = "auto",
    n_gpus: int | None = None,
    box_sampling: str = "lhs",
    acquisition: str = "logei",
    bound_method: str = "autobound",
    autobound_degree: int = 2,
    autobound_batch_size: int = 8192,
    predict_batch_size: int | None = None,
    bounds_batch_size: int | None = None,
    max_target_boxes: int | None = None,
    validation: bool = True,
    verbose: bool = False,
    logMask: bool = False,
    normalize_to_unit_cube: bool = False,
) -> BOResult:
    """
    Run ExactBO with backend selection and CPU fallback.

    Parameters
    ----------
    X0 : ndarray, shape (N0, d)
        Initial evaluated points.
    bounds : ndarray, shape (d, 2)
        Search-space bounds, [lower, upper] per dim.
    epsilon_X : float or ndarray, shape (d,)
        Partition termination threshold(s) for the input space.
    epsilon_ei : float
        Pruning tolerance: a box is discarded once its acquisition upper bound
        is within ``epsilon_ei`` of the best sampled value. With
        ``acquisition="logei"`` it is a tolerance on log EI (relative: 0.1
        keeps boxes that could beat the best EI by more than ~10.5%); with
        ``"ei"`` it is absolute, in the GP's standardized target space.
    gp : sklearn-like regressor
        Surrogate model with .fit/.predict plus sklearn GP attributes.
    f : callable
        Objective function.
    max_iters : int
        Outer BO iterations.
    max_partitions : int
        Max partition loops per BO iteration.
    backend : {"auto", "numpy", "cupy"}, default="auto"
        Execution backend. ``"auto"`` uses cupy when a GPU is visible.
    n_gpus : int, optional
        cupy only: number of visible GPUs the per-box work is split across
        (single process). None uses all visible GPUs.
    box_sampling : {"lhs", "center"}, default="lhs"
        Where EI is sampled inside each analyzed/active box: ``"lhs"`` at 2**d
        centered Latin-hypercube points, ``"center"`` only at the box center
        (2**d times fewer posterior evaluations and less sampling memory).
    acquisition : {"logei", "ei"}, default="logei"
        Score sampled points and bound boxes with the numerically stable log EI
        (finite even where EI underflows to 0) or with plain EI. Sets the
        meaning of ``epsilon_ei``.
    bound_method : {"autobound", "interval"}, default="autobound"
        ``"interval"`` bounds each kernel entry separately (cheap, loose for
        large boxes). ``"autobound"`` also bounds the posterior mean and
        variance as whole functions of x with AutoBound Taylor enclosures
        (requires ``jax`` and ``autobound``) and keeps the tighter of the two.
    autobound_degree : int, default=2
        Taylor degree for ``bound_method="autobound"``.
    autobound_batch_size : int, default=8192
        Boxes per AutoBound call (per GPU); memory is ~125 kB per box at
        N=32, d=10, degree 2.
    predict_batch_size : int, optional
        Max number of query points per GP posterior prediction call during
        partitioning (per GPU). If None, an automatic memory-aware value is used.
    bounds_batch_size : int, optional
        Max number of target boxes processed per bounds chunk during
        partitioning. If None, an automatic memory-aware value is used.
    max_target_boxes : int, optional
        Hard cap for the number of target boxes kept per partition.
        If None, no cap is applied.
    validation : bool, default=True
        Run additional checks for validation purposes (not optimized).
    verbose : bool, default=False
        Print loop-level progress.
    logMask : bool, default=False
        Enable logging of intermediate results.
    normalize_to_unit_cube : bool, default=False
        If True, optimize internally on [0, 1]^d and evaluate the objective after
        mapping candidates back to the original finite bounds.

    Returns
    -------
    BOResult
        Final design points, objective values, backend resolution info and log.
    """
    
    X = np.asarray(X0, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)

    if validation:
        if X.ndim != 2:
            raise ValueError(f"X0 must be 2D with shape (N0, d), got {X.shape}")
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"bounds must have shape (d, 2), got {bounds.shape}")

    dim = bounds.shape[0]
    if validation:
        if X.shape[1] != dim:
            raise ValueError(
                f"X0 second dimension ({X.shape[1]}) must match bounds dimension ({dim})"
            )

    objective = f
    physical_bounds = None
    if normalize_to_unit_cube:
        X, bounds, objective, physical_bounds = _normalize_problem_to_unit_cube(
            X,
            bounds,
            f,
            validation=validation,
        )

    epsilon_X = _normalize_epsilon(epsilon_X, dim)
    backend_info = resolve_backend(backend)

    log = _init_log(logMask)

    for iteration in range(max_iters):
        X_display = (
            _from_unit_cube(X, physical_bounds, validation=False)
            if physical_bounds is not None
            else X
        )
        if verbose:
            print(f"Iteration {iteration + 1}/{max_iters}")
        # Evaluate function at current data points
        y = _evaluate_objective(objective, X)  # (N,)
        if verbose:
            print(f"Current training data: \nX: {X_display}, \ny: {y}")
        if logMask:
            log[f"i{iteration}"] = {"X": X_display.copy(), "y": y.copy()}

        # Fit Gaussian Process
        gp.fit(X, y)
        if verbose:
            print(f"GP kernel after fitting: {gp.kernel_}")

        # Run partitioning to find next point and evaluate it
        partitioning_result = exactbo_partitioning(
            X,
            bounds,
            epsilon_X,
            epsilon_ei,
            gp,
            max_partitions,
            backend=backend_info.selected,
            n_gpus=n_gpus,
            box_sampling=box_sampling,
            acquisition=acquisition,
            bound_method=bound_method,
            autobound_degree=autobound_degree,
            autobound_batch_size=autobound_batch_size,
            predict_batch_size=predict_batch_size,
            bounds_batch_size=bounds_batch_size,
            max_target_boxes=max_target_boxes,
            validation=validation,
            verbose=verbose,
            logMask=logMask,
        )
        Xn = np.asarray(partitioning_result.X, dtype=np.float64).ravel()  # (d,)
        yn = _evaluate_objective(objective, Xn)  # (1,)
        Xn_display = (
            _from_unit_cube(Xn, physical_bounds, validation=False)
            if physical_bounds is not None
            else Xn
        )
        if verbose:
            print(f"Evaluated new point: {Xn_display} -> {yn}")
        if logMask:
            log[f"i{iteration}"].update(partitioning_result.log)
            log[f"i{iteration}"].update({"Xn": Xn_display.copy(), "yn": yn.copy()})

        # Update data
        X = np.vstack((X, Xn))  # (N+1,d)
        y = np.hstack((y, yn))  # (N+1,)


    X_result = (
        _from_unit_cube(X, physical_bounds, validation=False)
        if physical_bounds is not None
        else X
    )
    return BOResult(X_result, y, backend_info, log)


def exactbo_partitioning(
    X: np.ndarray,
    bounds: np.ndarray,
    epsilon_X: np.ndarray,
    epsilon_ei: float,
    gp,
    max_partitions: int,
    *,
    backend: BackendName = "auto",
    n_gpus: int | None = None,
    box_sampling: str = "lhs",
    acquisition: str = "logei",
    bound_method: str = "autobound",
    autobound_degree: int = 2,
    autobound_batch_size: int = 8192,
    predict_batch_size: int | None = None,
    bounds_batch_size: int | None = None,
    max_target_boxes: int | None = None,
    validation: bool = True,
    verbose: bool = False,
    logMask: bool = False,
) -> BOResult:
    """
    Run ExactBO Partitioning with backend selection.

    Parameters
    ----------
    X : ndarray, shape (N, d)
        Evaluated points.
    bounds : ndarray, shape (d, 2)
        Search-space bounds, [lower, upper] per dim.
    epsilon_X : float or ndarray, shape (d,)
        Partition termination threshold(s) for the input space.
    epsilon_ei : float
        Pruning tolerance on the acquisition values: log EI for
        ``acquisition="logei"``, EI in the GP's standardized target space for
        ``"ei"``.
    gp : sklearn-like regressor
        Surrogate model with .fit/.predict plus sklearn GP attributes.
    iteration : int
        Current BO iteration.
    max_partitions : int
        Max partition loops per BO iteration.
    backend : {"auto", "numpy", "cupy"}, default="auto"
        Backend used for array ops.
    n_gpus : int, optional
        cupy only: number of visible GPUs the per-box EI-bound and sampling
        work is split across. None uses all visible GPUs.
    box_sampling : {"lhs", "center"}, default="lhs"
        Sample EI at 2**d centered Latin-hypercube points per box, or only at
        the box center.
    acquisition : {"logei", "ei"}, default="logei"
        Score sampled points and bound boxes with the numerically stable log EI
        (finite even where EI underflows to 0) or with plain EI. Sets the
        meaning of ``epsilon_ei``.
    bound_method : {"autobound", "interval"}, default="autobound"
        ``"interval"`` bounds each kernel entry separately (cheap, loose for
        large boxes). ``"autobound"`` also bounds the posterior mean and
        variance as whole functions of x with AutoBound Taylor enclosures
        (requires ``jax`` and ``autobound``) and keeps the tighter of the two.
    autobound_degree : int, default=2
        Taylor degree for ``bound_method="autobound"``.
    autobound_batch_size : int, default=8192
        Boxes per AutoBound call (per GPU); memory is ~125 kB per box at
        N=32, d=10, degree 2.
    predict_batch_size : int, optional
        Max number of query points per GP posterior prediction call (per GPU).
        If None, an automatic memory-aware value is used.
    bounds_batch_size : int, optional
        Max number of target boxes processed per bounds chunk (per GPU).
        If None, an automatic memory-aware value is used.
    max_target_boxes : int, optional
        Hard cap for the number of target boxes kept per partition.
        If None, no cap is applied.
    validation : bool, default=True
        Run additional checks for validation purposes (not optimized).
    verbose : bool, default=False
        Print loop-level progress.
    logMask : bool, default=False
        Enable logging of intermediate results.

    Returns
    -------
    BOResult
        Next design point, backend resolution info and log.
    """
    xp = _array_module(backend)

    # Initialize boxes on the main (current) device
    ## One row per box (initially one box)
    ## One column per dimension
    bounds_L = xp.asarray(bounds[np.newaxis, :, 0], dtype=xp.float64)  # (n,d)
    bounds_U = xp.asarray(bounds[np.newaxis, :, 1], dtype=xp.float64)  # (n,d)

    # GP hyperparameters (host scalars)
    gp_kernel_params = gp.kernel_.get_params()
    sigma_f_2 = gp_kernel_params["k1__k1__constant_value"]
    sigma_n_2 = gp_kernel_params["k2__noise_level"]
    y_train_std = float(np.asarray(gp._y_train_std, dtype=np.float64).ravel()[0])
    y_train_mean = float(np.asarray(gp._y_train_mean, dtype=np.float64).ravel()[0])
    y_min_scaled = float(np.min(gp.y_train_))

    # Partition parameters
    N = X.shape[0]  # Number of data points
    d = bounds.shape[0]  # Number of dimensions
    if predict_batch_size is None:
        predict_batch_size = max(1, int((128 * 1024**2) // max(16 * N, 16)))
    else:
        predict_batch_size = int(predict_batch_size)
        if predict_batch_size <= 0:
            raise ValueError("predict_batch_size must be a positive integer.")
    if bounds_batch_size is None:
        bounds_batch_size = max(1, int((192 * 1024**2) // max(16 * N, 16)))
    else:
        bounds_batch_size = int(bounds_batch_size)
        if bounds_batch_size <= 0:
            raise ValueError("bounds_batch_size must be a positive integer.")
    if max_target_boxes is not None:
        max_target_boxes = int(max_target_boxes)
        if max_target_boxes <= 0:
            raise ValueError("max_target_boxes must be a positive integer.")
    stride = 2 * d + 1
    # Points (in box-relative [0, 1]^d coordinates) where EI is sampled in each box.
    if box_sampling == "lhs":
        unit_design = _centered_latin_hypercube_unit(int(2**d), d)
    elif box_sampling == "center":
        unit_design = np.full((1, d), 0.5)
    else:
        raise ValueError(f"box_sampling must be 'lhs' or 'center', got {box_sampling!r}.")
    points_per_box = unit_design.shape[0]
    if acquisition not in ("logei", "ei"):
        raise ValueError(f"acquisition must be 'logei' or 'ei', got {acquisition!r}.")
    use_log = acquisition == "logei"
    if bound_method == "autobound":
        from .autobound_bounds import taylor_mu_q_bounds
    elif bound_method != "interval":
        raise ValueError(f"bound_method must be 'autobound' or 'interval', got {bound_method!r}.")
    # GP-only factors of the sigma bounds, computed once on the host.
    L_inv, lambda_max = sigma_bound_factors(gp.L_)
    # One copy of the trained GP per device; per-box work is sharded across them.
    states = [
        _build_gp_state(xp, device, X, gp, unit_design, L_inv)
        for device in _resolve_devices(xp, n_gpus)
    ]
    _prepare_devices(xp, states)
    # Per-box GPU work estimates that decide how many devices a call uses:
    # interval EI bounds do a few N^2 ops per box (sigma GEMMs); AutoBound's
    # degree-2 enclosure carries N*d^2 coefficients through ~100s of ops;
    # sampling evaluates the posterior (~N^2 each) at points_per_box points per box.
    bounds_work_per_box = 3.0 * N * N + (600.0 * N * d * d if bound_method == "autobound" else 0.0)
    sample_work_per_box = float(points_per_box) * N * N
    w = bounds_U[0] - bounds_L[0]  # Bounds with per dimension (d,)
    epsilon_X = xp.asarray(epsilon_X, dtype=xp.float64)
    partition = 0
    w_max = w.copy()
    target_boxes_mask = xp.ones((1,), dtype=bool)
    n_target_start = 1
    idx_best_global = 0
    n_total = 1
    # Best sampled point over all partitions (the incumbent). Children are sampled
    # at new points, so a partition's best can be below an earlier one; pruning
    # against the incumbent is valid (it is an attained value) and never weaker.
    best_x_incumbent = None
    best_score_incumbent = -math.inf

    # Initialize log
    log = _init_log(logMask)
    if logMask:
        log["partitions"] = []

    def _finalize_result(best_x, score_max: float) -> BOResult:
        best_x_result = to_numpy(best_x).astype(np.float64, copy=False)
        if logMask:
            t1 = now()
            ei_max_scaled = math.exp(score_max) if use_log else score_max
            log["time"] = t1 - t0
            log["ei_max"] = float(ei_max_scaled * y_train_std)
            log["ei_max_scaled"] = float(ei_max_scaled)
            if use_log:
                log["log_ei_max_scaled"] = float(score_max)
        return BOResult(X=best_x_result, log=log)

    def _predict_with_std(state, points):
        points = xp.asarray(points, dtype=xp.float64)
        n_points = int(points.shape[0])
        if n_points <= predict_batch_size:
            if xp is np:
                mu_chunk, sigma_chunk = gp.predict(np.asarray(points), return_std=True)
                mu_chunk = (mu_chunk - y_train_mean) / y_train_std
                sigma_chunk = sigma_chunk / y_train_std
            else:
                mu_chunk, sigma_chunk = gp_posterior(
                    points,
                    X_train=state.X_train,
                    alpha=state.alpha,
                    L=state.L,
                    length_scale=state.length_scale,
                    sigma_f_squared=sigma_f_2,
                    sigma_n_squared=sigma_n_2,
                    y_train_mean=y_train_mean,
                    y_train_std=y_train_std,
                    scaled_output=True,
                    return_std=True,
                    backend=backend,
                    validation=False,
                )
            return (
                xp.asarray(mu_chunk, dtype=xp.float64),
                xp.asarray(sigma_chunk, dtype=xp.float64),
            )

        mu = xp.empty((n_points,), dtype=xp.float64)
        sigma = xp.empty((n_points,), dtype=xp.float64)
        for start in range(0, n_points, predict_batch_size):
            end = min(start + predict_batch_size, n_points)
            chunk = points[start:end]
            if xp is np:
                mu_chunk, sigma_chunk = gp.predict(np.asarray(chunk), return_std=True)
                mu_chunk = (mu_chunk - y_train_mean) / y_train_std
                sigma_chunk = sigma_chunk / y_train_std
            else:
                mu_chunk, sigma_chunk = gp_posterior(
                    chunk,
                    X_train=state.X_train,
                    alpha=state.alpha,
                    L=state.L,
                    length_scale=state.length_scale,
                    sigma_f_squared=sigma_f_2,
                    sigma_n_squared=sigma_n_2,
                    y_train_mean=y_train_mean,
                    y_train_std=y_train_std,
                    scaled_output=True,
                    return_std=True,
                    backend=backend,
                    validation=False,
                )
            mu[start:end] = xp.asarray(mu_chunk, dtype=xp.float64)
            sigma[start:end] = xp.asarray(sigma_chunk, dtype=xp.float64)
        return mu, sigma

    def _ei_hi_bounds_chunked(state, bounds_L_target, bounds_U_target):
        n_target = int(bounds_L_target.shape[0])
        ei_hi = xp.empty((n_target,), dtype=xp.float64)
        for start in range(0, n_target, bounds_batch_size):
            end = min(start + bounds_batch_size, n_target)
            chunk_n = end - start
            chunk_L = bounds_L_target[start:end]
            chunk_U = bounds_U_target[start:end]

            K_lo = xp.empty((chunk_n, N), dtype=xp.float64)
            K_hi = xp.empty((chunk_n, N), dtype=xp.float64)
            for i in range(N):
                xi = state.X_train[i]
                K_lo[:, i], K_hi[:, i] = rbf_k_bounds(
                    chunk_L,
                    chunk_U,
                    xi,
                    chunk_n,
                    d,
                    sigma_f_2,
                    state.length_scale,
                    backend=backend,
                    validation=validation,
                )

            mu_lo, mu_hi = mu_bounds(
                state.alpha,
                K_lo,
                K_hi,
                chunk_n,
                N,
                y_train_mean=y_train_mean,
                y_train_std=y_train_std,
                scaled_output=True,
                backend=backend,
                validation=validation,
            )
            q_bounds = None
            if bound_method == "autobound":
                # Whole-function Taylor bounds; keep the tighter of both methods.
                ab_mu_lo, ab_mu_hi, ab_q_lo, ab_q_hi = taylor_mu_q_bounds(
                    chunk_L,
                    chunk_U,
                    state.X_train,
                    state.alpha,
                    state.K_inv,
                    state.length_scale,
                    sigma_f_2,
                    xp=xp,
                    degree=autobound_degree,
                    batch_size=autobound_batch_size,
                )
                xp.maximum(mu_lo, ab_mu_lo, out=mu_lo)
                xp.minimum(mu_hi, ab_mu_hi, out=mu_hi)
                q_bounds = (ab_q_lo, ab_q_hi)
                del ab_mu_lo, ab_mu_hi
            sig_lo, sig_hi = sigma_bounds(
                K_lo,
                K_hi,
                state.L,
                chunk_n,
                N,
                sigma_f_2,
                y_train_std=y_train_std,
                scaled_output=True,
                backend=backend,
                validation=validation,
                L_inv=state.L_inv,
                lambda_max=lambda_max,
                q_bounds=q_bounds,
            )
            _, ei_hi_chunk = ei_bounds(
                mu_lo,
                mu_hi,
                sig_lo,
                sig_hi,
                chunk_n,
                y_min_scaled,
                backend=backend,
                validation=validation,
                log=use_log,
            )
            ei_hi[start:end] = ei_hi_chunk
            del K_lo, K_hi, mu_lo, mu_hi, sig_lo, sig_hi, ei_hi_chunk, q_bounds
        return (ei_hi,)

    def _sampled_box_best_ei(state, boxes_L, boxes_U):
        n_boxes = int(boxes_L.shape[0])
        if n_boxes == 0:
            return (
                xp.empty((0, d), dtype=xp.float64),
                xp.empty((0,), dtype=xp.float64),
            )

        best_points = xp.empty((n_boxes, d), dtype=xp.float64)
        best_ei = xp.empty((n_boxes,), dtype=xp.float64)
        boxes_per_chunk = max(1, predict_batch_size // points_per_box)

        for start in range(0, n_boxes, boxes_per_chunk):
            if verbose:
                print(f"    Sampling points for boxes {start} to {min(start + boxes_per_chunk, n_boxes) - 1}...")
            end = min(start + boxes_per_chunk, n_boxes)
            chunk_n = end - start
            chunk_L = boxes_L[start:end]
            chunk_U = boxes_U[start:end]
            chunk_width = chunk_U - chunk_L

            sampled_points = (
                chunk_L[:, xp.newaxis, :]
                + state.unit_design[xp.newaxis, :, :] * chunk_width[:, xp.newaxis, :]
            )  # (chunk_n, points_per_box, d)
            flat_points = sampled_points.reshape((chunk_n * points_per_box, d))

            mu_chunk, sigma_chunk = _predict_with_std(state, flat_points)
            mu_chunk = xp.asarray(mu_chunk, dtype=xp.float64).reshape(
                (chunk_n, points_per_box)
            )
            sigma_chunk = xp.asarray(sigma_chunk, dtype=xp.float64).reshape(
                (chunk_n, points_per_box)
            )
            sigma_chunk_lat = xp.sqrt(
                xp.clip(sigma_chunk**2 - sigma_n_2, 1e-12, None)
            )
            acquisition_fn = log_expected_improvement if use_log else expected_improvement
            ei_chunk = acquisition_fn(
                mu_chunk,
                sigma_chunk_lat,
                y_min_scaled,
                backend=backend,
            )  # (chunk_n, points_per_box)

            best_idx = xp.argmax(ei_chunk, axis=1).reshape((chunk_n, 1))
            best_ei[start:end] = xp.take_along_axis(
                ei_chunk,
                best_idx,
                axis=1,
            ).reshape((chunk_n,))
            gather_idx = xp.broadcast_to(
                best_idx[:, :, xp.newaxis],
                (chunk_n, 1, d),
            )
            best_points[start:end] = xp.take_along_axis(
                sampled_points,
                gather_idx,
                axis=1,
            ).reshape((chunk_n, d))

            del (
                chunk_L,
                chunk_U,
                chunk_width,
                sampled_points,
                flat_points,
                mu_chunk,
                sigma_chunk,
                sigma_chunk_lat,
                ei_chunk,
                best_idx,
                gather_idx,
            )

        return best_points, best_ei

    def _topk_primary_secondary(primary, secondary, k: int):
        """
        Return indices of the top-k entries using `primary` as the main score and
        `secondary` as a tie-breaker, without sorting the full input.
        """
        m = int(primary.shape[0])
        if k >= m:
            return xp.arange(m, dtype=xp.int64)

        shortlist = min(m, max(2 * k, k + 4096))
        if hasattr(xp, "argpartition") and shortlist < m:
            shortlist_idx = xp.argpartition(primary, m - shortlist)[m - shortlist:]
        else:
            shortlist_idx = xp.argsort(primary)[-shortlist:]

        short_primary = primary[shortlist_idx]
        short_secondary = secondary[shortlist_idx]
        s = int(shortlist_idx.shape[0])

        rank_secondary = xp.empty((s,), dtype=xp.int64)
        order_secondary = xp.argsort(short_secondary)
        rank_secondary[order_secondary] = xp.arange(s, dtype=xp.int64)

        rank_primary = xp.empty((s,), dtype=xp.int64)
        order_primary = xp.argsort(short_primary)
        rank_primary[order_primary] = xp.arange(s, dtype=xp.int64)

        combined = rank_primary * (s + 1) + rank_secondary
        keep_order = xp.argsort(combined)[-k:]
        keep_idx = shortlist_idx[keep_order]
        del short_primary, short_secondary, rank_secondary, order_secondary, rank_primary, order_primary, combined, keep_order
        return keep_idx
    
    if logMask:
        now = _get_timer(xp)
        t0 = now()
    
    while partition < max_partitions:
        if verbose:
            print(
                f"Partition {partition}/{max_partitions-1},"
            )
        # Total number of boxes
        n = bounds_L.shape[0]
        
        # For the first partition, we analyze just the original box. For subsequent partitions, 
        # we only focus on the new target boxes resulting from the previous partition.
        if partition > 0:
            # Starting target box count (new 2d+1 boxes per each of the n_targets boxes from the previous partition)
            n_target_start = n_target * stride
            # Starting target box mask (only analyze the new target boxes from the previous partition)
            target_boxes_mask = xp.zeros((n,), dtype=bool)
            target_boxes_mask[:n_target_start] = True
            # All children of the previously best target box are contiguous in the
            # split output, so preserve that whole child block for analysis.
            idx_best_global_start = int(idx_best_global_next * stride)
            preserved_analyze_idx = xp.arange(
                idx_best_global_start,
                idx_best_global_start + stride,
                dtype=xp.int64,
            )
        else:
            preserved_analyze_idx = xp.asarray([0], dtype=xp.int64)
        
        # Target boxes are always at the start of the arrays after each split.
        # Use slicing (views) to avoid advanced-index copies of large arrays.
        bounds_L_target = bounds_L[:n_target_start]
        bounds_U_target = bounds_U[:n_target_start]

        if verbose:
            print(f"  Start target boxes: {n}, to analyze: {bounds_L_target.shape[0]}, Best global box index: {idx_best_global}.")
        # Compute EI upper bounds in chunks to cap peak GPU memory, sharded across GPUs.
        (ei_hi,) = _run_sharded(
            xp, _ei_hi_bounds_chunked, states, bounds_L_target, bounds_U_target, bounds_work_per_box
        )
        if verbose:
            print(f"  Computed EI upper bounds for {ei_hi.shape[0]} target boxes.")

        # Find the box with the highest upper EI bound
        idx_max_ei_hi = int(xp.argmax(ei_hi))
        #if verbose:
        #    print(f"  Max EI_hi at box index {idx_max_ei_hi}.")
        max_ei_hi = float(ei_hi[idx_max_ei_hi])

        # Analyze boxes where the upper EI bound is within epsilon_ei of the maximum upper EI bound
        analyze_box_mask = ei_hi >= (max_ei_hi - epsilon_ei)  # (n_target_start,)
        # Also analyze the full child block of the previously best target box.
        analyze_box_mask[preserved_analyze_idx] = True
        analyze_local_idx = xp.where(analyze_box_mask)[0]  # (n_analyze,)
        n_analyze = int(analyze_local_idx.shape[0])
        # Sample EI within each analyzed box (per box_sampling) and retain
        # the best sampled EI per box in standardized target space.
        analyze_best_points, ei_analyze = _run_sharded(
            xp,
            _sampled_box_best_ei,
            states,
            bounds_L_target[analyze_local_idx],
            bounds_U_target[analyze_local_idx],
            sample_work_per_box,
        )  # ((n_analyze, d), (n_analyze,))
        if verbose:
            print(f"  Analyzed {ei_analyze.shape[0]} boxes with EI within {epsilon_ei} of max EI_hi.")
        # Find the box with the highest analyzed EI
        idx_ei_max_analyze = int(xp.argmax(ei_analyze))
        ei_max_analyze = float(ei_analyze[idx_ei_max_analyze])
        idx_ei_max_analyze_local = int(analyze_local_idx[idx_ei_max_analyze])
        # Find best point among the analyzed boxes and the width of the box with the highest analyzed EI
        best_x_analyze = xp.array(analyze_best_points[idx_ei_max_analyze], dtype=xp.float64)
        w_max_ei_analyzed = (
            bounds_U_target[idx_ei_max_analyze_local] - bounds_L_target[idx_ei_max_analyze_local]
        )  # (d,)
        del analyze_best_points, ei_analyze
        if ei_max_analyze > best_score_incumbent:
            best_x_incumbent, best_score_incumbent = best_x_analyze, ei_max_analyze

        # Active boxes are the ones where ei_hi is higher than the incumbent plus epsilon_ei,
        active_boxes_mask = ei_hi > (best_score_incumbent + epsilon_ei)  # (n_target_start,)
        n_active = int(xp.sum(active_boxes_mask))
        
        # No active boxes and max EI box is smaller than epsilon_X, return the best point found
        if n_active == 0 and xp.all(w_max_ei_analyzed < epsilon_X):
            if verbose:
                print(
                    f"  Boxes: {n_total}, Analyzed: {n_analyze}, Active: 0,\n"
                    f"  Max EI_hi: {max_ei_hi:.6f}, Max EI Analyzed: {ei_max_analyze:.6f},\n"
                    f"  Max EI Analyzed Box Width: {w_max_ei_analyzed}, Terminating partitioning."
                )
            # Uncomment this for 2D animations
            #if logMask:
            #    log[f"p{partition}"] = {
            #        "bounds_L": np.asarray(bounds_L),
            #        "bounds_U": np.asarray(bounds_U),
            #        "target_boxes_mask": np.zeros((n,), dtype=bool),
            #    }
            return _finalize_result(best_x_incumbent, best_score_incumbent)
        else:
            # Ensure the box with the highest analyzed EI is also active.
            active_boxes_mask[idx_ei_max_analyze_local] = True
            n_active = int(xp.sum(active_boxes_mask))

        # Check the active boxes.
        active_local_idx = xp.where(active_boxes_mask)[0] # (n_active,)
        # Sample EI within each active box (per box_sampling) and retain
        # the best sampled EI per box.
        if verbose:
            print(f"  Active boxes with EI_hi > max EI_analyze + epsilon_ei: {n_active}.")
        active_best_points, ei_active = _run_sharded(
            xp,
            _sampled_box_best_ei,
            states,
            bounds_L_target[active_local_idx],
            bounds_U_target[active_local_idx],
            sample_work_per_box,
        )  # ((n_active, d), (n_active,))
        if verbose:
            print(f"  Sampled best points for {ei_active.shape[0]} active boxes with EI_hi > max EI_analyze + epsilon_ei.")
        # Find the box with the highest EI among the active boxes
        idx_best = int(xp.argmax(ei_active))
        ei_max_active = float(ei_active[idx_best])
        idx_best_local = int(active_local_idx[idx_best])
        # Find best point among the active boxes and the width of the box with the highest active EI
        best_x_active = xp.array(active_best_points[idx_best], dtype=xp.float64)
        w_max_ei_active = bounds_U_target[idx_best_local] - bounds_L_target[idx_best_local]  # (d,)
        del active_best_points
        if ei_max_active > best_score_incumbent:
            best_x_incumbent, best_score_incumbent = best_x_active, ei_max_active

        # Target boxes are the ones where ei_hi is more than epsilon_ei plus the incumbent.
        target_boxes_mask[:n_target_start] = ei_hi > (best_score_incumbent + epsilon_ei)
        n_target = int(xp.sum(target_boxes_mask))
        if logMask:
            log["partitions"].append({
                "n_boxes": int(n_target_start),
                "n_analyze": n_analyze,
                "n_active": n_active,
                "n_target": n_target,
                "max_ei_hi": max_ei_hi,
                "best": best_score_incumbent,
                "time": now() - t0,
            })

        # No target boxes and max EI box is smaller than epsilon_X, return the best point found
        if n_target == 0 and xp.all(w_max_ei_active < epsilon_X):
            if verbose:
                print(
                    f"  Boxes: {n_total}, Analyzed: {n_analyze}, Active: {n_active}, Target: 0,\n"
                    f"  Max EI_hi: {max_ei_hi:.6f}, Max EI Analyzed: {ei_max_analyze:.6f}, Max EI Active: {ei_max_active:.6f},\n"
                    f"  Max EI Active Box Width: {w_max_ei_active}, Terminating partitioning."
                )
            # Uncomment this for 2D animations
            #if logMask:
            #    log[f"p{partition}"] = {
            #        "bounds_L": np.asarray(bounds_L),
            #        "bounds_U": np.asarray(bounds_U),
            #        "target_boxes_mask": np.zeros((n,), dtype=bool),
            #    }
            return _finalize_result(best_x_incumbent, best_score_incumbent)
        else:
            # Ensure the box with the highest active EI is also a target box.
            idx_best_global = int(idx_best_local)
            target_boxes_mask[idx_best_global] = True
            n_target = int(xp.sum(target_boxes_mask))

        # Optional approximation guard to avoid combinatorial target growth.
        if max_target_boxes is not None and n_target > max_target_boxes:
            #if verbose:
            #    print(
            #        f"  Pruning target boxes from {n_target} to {max_target_boxes} to limit combinatorial growth."
            #    )
            keep = min(max_target_boxes, n_target)
            # Target boxes are a subset of active boxes, so use active ordering
            # directly and avoid the expensive searchsorted/multi-sort path.
            target_in_active_mask = ei_hi[active_local_idx] > (best_score_incumbent + epsilon_ei)
            target_in_active_mask[idx_best] = True
            target_local_idx = active_local_idx[target_in_active_mask]
            target_ei_active = ei_active[target_in_active_mask]
            target_ei_hi = ei_hi[target_local_idx]

            keep_pos = _topk_primary_secondary(target_ei_active, target_ei_hi, keep)
            keep_local = target_local_idx[keep_pos]
            target_boxes_mask[:n_target_start] = False
            target_boxes_mask[keep_local] = True
            target_boxes_mask[idx_best_global] = True
            n_target = int(xp.sum(target_boxes_mask))
            if verbose:
                print(
                    f"  Pruned target boxes from {int(target_local_idx.shape[0])} to {n_target} "
                    f"(max_target_boxes={max_target_boxes}, score=ei_active+ei_hi tie-break)."
                )
            del target_in_active_mask, target_local_idx, target_ei_active, target_ei_hi, keep_pos, keep_local

        # Calculate position of the best point in next partition
        idx_best_global_next = int(xp.sum(target_boxes_mask[:idx_best_global]))
        del ei_hi, ei_active, analyze_box_mask, active_boxes_mask, analyze_local_idx, active_local_idx
        
        # Update maximum width of active boxes
        w_max = xp.max(bounds_U[target_boxes_mask] - bounds_L[target_boxes_mask], axis=0)

        if verbose:
            print(
                f"  Boxes: {n_total}, Analyzed: {n_analyze}, Active: {n_active}, Target: {n_target},\n"
                f"  Max EI_hi: {max_ei_hi:.6f}, Max EI Analyzed: {ei_max_analyze:.6f}, Max EI Active: {ei_max_active:.6f},\n" 
                f"  Incumbent: {best_score_incumbent:.6f}, Max Width: {w_max}."
            )

        # Uncomment this for 2D animations
        #if logMask:
        #    log[f"p{partition}"] = {
        #        "bounds_L": np.asarray(bounds_L),
        #        "bounds_U": np.asarray(bounds_U),
        #        "target_boxes_mask": np.asarray(target_boxes_mask),
        #    }
        
        # Update partition count
        partition += 1

        # Calculate number of boxes for next iteration
        n_total = n_total + n_target * (2 * d)

        # Split active boxes (don't if its the last partition)
        if partition < max_partitions:
            bounds_L, bounds_U = split_boxes(
                bounds_L,
                bounds_U,
                target_boxes_mask,
                w,
                n,
                d,
                keep_inactive=False,
                backend=backend,
                validation=validation,
            )
    return _finalize_result(best_x_incumbent, best_score_incumbent)
