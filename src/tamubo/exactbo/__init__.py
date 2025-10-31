"""
exactBO: minimal exact Bayesian Optimization scaffold.

Exports:
- Box, split_box, hypermask
- expected_improvement
- ei_bounds_from_mu_sigma_intervals, EIBounds
- PartitionMaxEISearch
- ExactBOLoop, BOResult
"""
from .partition import Box, split_box, hypermask
from .ei import expected_improvement
from .bounds import mu_bounds, sigma_bounds, ei_bounds_from_mu_sigma, ei_bounds
from .interval_arithmetics import Bounds, prod_bound_scalar, add_bounds, sub_bounds, prod_bounds, square_bounds, sqrt_bounds, forward_solve_bounds
from .loop_partition import PartitionMaxEISearch
from .loop_exactbo import ExactBOLoop, BOResult

__all__ = [
    "Box",
    "split_box",
    "hypermask",
    "expected_improvement",
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
]
