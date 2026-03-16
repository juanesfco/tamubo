"""
exactBO: minimal exact Bayesian Optimization scaffold.

Exports:
- Bounds functions: rbf_k_bounds, mu_bounds, sigma_bounds, ei_bounds
- Main functions: exactbo, exactbo_partitioning
- Partitioning utility: split_boxes
- 2D Plotting helpers: plot_f, plot_log, plot_opt
"""
from .bounds import rbf_k_bounds, mu_bounds, sigma_bounds, ei_bounds
from .run import exactbo, exactbo_partitioning
from .partition import split_boxes
from .plot2D import plot_f, plot_log, plot_opt

__all__ = [
    "rbf_k_bounds", "mu_bounds", "sigma_bounds", "ei_bounds",
    "exactbo", "exactbo_partitioning",
    "split_boxes",
    "plot_f", "plot_log", "plot_opt"
]
