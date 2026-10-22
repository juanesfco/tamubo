"""Optional, application-independent integration boundaries."""
from .gp import (
    FrozenExactGPState,
    FrozenIndependentMultiOutputGPState,
    freeze_gpytorch_exact_gp,
    freeze_botorch_single_task_gp,
    freeze_sklearn_gaussian_process_regressor,
    frozen_gp_posterior,
    frozen_gp_posterior_mean,
    frozen_gp_posterior_variance,
    frozen_gp_posterior_covariance,
)
__all__ = [
    "FrozenExactGPState", "FrozenIndependentMultiOutputGPState",
    "freeze_gpytorch_exact_gp", "freeze_botorch_single_task_gp",
    "freeze_sklearn_gaussian_process_regressor",
    "frozen_gp_posterior", "frozen_gp_posterior_mean", "frozen_gp_posterior_variance",
    "frozen_gp_posterior_covariance",
]
