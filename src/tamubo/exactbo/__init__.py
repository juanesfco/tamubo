"""
exactBO: minimal exact Bayesian Optimization scaffold.

Exports:
- Box, Boxes, split_box, hypermask
- expected_improvement
- rbf_k_bounds, mu_bounds, sigma_bounds, ei_bounds_from_mu_sigma, ei_bounds
- Bounds, prod_bound_scalar, add_bounds, sub_bounds, prod_bounds, square_bounds, sqrt_bounds, forward_solve_bounds
- PartitionMaxEISearch
- ExactBOLoop, BOResult
- run_exactbo, ExactBORunResult
- BackendInfo, has_cupynumeric, resolve_backend
- plot_iterations, plot_iterations_2d, plot_function_2d, plot_partitions_2d, plot_ploop_2d
"""
from .partition import Box, Boxes, split_box, hypermask
from .ei import expected_improvement
from .bounds import rbf_k_bounds, mu_bounds, sigma_bounds, ei_bounds_from_mu_sigma, ei_bounds
from .interval_arithmetics import Bounds, prod_bound_scalar, add_bounds, sub_bounds, prod_bounds, square_bounds, sqrt_bounds, forward_solve_bounds
from .loop_partition import PartitionMaxEISearch
from .loop_exactbo import ExactBOLoop, BOResult
from .backend import BackendInfo, has_cupynumeric, resolve_backend
from .run import ExactBORunResult, run_exactbo
from .plots import plot_iterations, plot_iterations_2d, plot_function_2d, plot_partitions_2d, plot_ploop_2d

__all__ = [
    "Box",
    "Boxes",
    "split_box",
    "hypermask",
    "expected_improvement",
    "rbf_k_bounds",
    "mu_bounds",
    "sigma_bounds",
    "ei_bounds_from_mu_sigma",
    "ei_bounds",
    "Bounds",
    "prod_bound_scalar",
    "add_bounds",
    "sub_bounds",
    "prod_bounds",
    "square_bounds",
    "sqrt_bounds",
    "forward_solve_bounds",
    "PartitionMaxEISearch",
    "ExactBOLoop",
    "BOResult",
    "BackendInfo",
    "has_cupynumeric",
    "resolve_backend",
    "ExactBORunResult",
    "run_exactbo",
    "plot_iterations",
    "plot_iterations_2d",
    "plot_function_2d",
    "plot_partitions_2d",
    "plot_ploop_2d",
]
