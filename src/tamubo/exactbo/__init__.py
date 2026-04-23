"""
exactBO: exact Bayesian Optimization utilities.

Exports:
- Bounds functions: rbf_k_bounds, mu_bounds, sigma_bounds, ei_bounds
- Main functions: exactbo, exactbo_partitioning
- Partitioning utility: split_boxes
- BoTorch optimizer adapter: optimize_acqf_exactbo
- BoTorch workflow: run_botorch_exactbo_ei
- 2D Plotting helpers: plot_f, plot_log, plot_opt
"""
try:
    from .plot2D import plot_f, plot_log, plot_opt
except ImportError:
    plot_f = None
    plot_log = None
    plot_opt = None

try:
    from .bounds import rbf_k_bounds, mu_bounds, sigma_bounds, ei_bounds
    from .run import exactbo, exactbo_partitioning
    from .partition import split_boxes
except ImportError:
    rbf_k_bounds = None
    mu_bounds = None
    sigma_bounds = None
    ei_bounds = None
    exactbo = None
    exactbo_partitioning = None
    split_boxes = None

try:
    from .torch_bounds import (
        expected_improvement_torch,
        ei_bounds_torch,
        mu_bounds_torch,
        rbf_k_bounds_torch,
        sigma_bounds_torch,
    )
    from .torch_partition import exactbo_torch_partitioning, extract_torch_gp_state
    from .botorch import optimize_acqf_exactbo
    from .torch_run import run_botorch_exactbo_ei
except ImportError:
    expected_improvement_torch = None
    ei_bounds_torch = None
    mu_bounds_torch = None
    rbf_k_bounds_torch = None
    sigma_bounds_torch = None
    exactbo_torch_partitioning = None
    extract_torch_gp_state = None
    optimize_acqf_exactbo = None
    run_botorch_exactbo_ei = None

__all__ = [
    "rbf_k_bounds", "mu_bounds", "sigma_bounds", "ei_bounds",
    "exactbo", "exactbo_partitioning",
    "split_boxes",
    "rbf_k_bounds_torch", "mu_bounds_torch", "sigma_bounds_torch", "ei_bounds_torch",
    "expected_improvement_torch",
    "extract_torch_gp_state", "exactbo_torch_partitioning",
    "optimize_acqf_exactbo", "run_botorch_exactbo_ei",
    "plot_f", "plot_log", "plot_opt"
]
