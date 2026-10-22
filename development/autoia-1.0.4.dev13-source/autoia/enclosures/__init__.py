"""AutoIA scalar enclosure representations."""

from .runtime import (
    interval_intersection,
    safe_interval_representative,
    RangePolicyContext,
    record_range_policy_event,
    active_range_policy,
    ApproximationDiagnosticContext,
    active_approximation_diagnostics,
    new_approximation_source_mask,
    record_approximation_recombination,
)
from .polynomial import (
    ParameterizedEnclosure,
    PolynomialEnclosure,
    contains_pe,
    extract_affine_array,
    extract_polynomial_array,
)
from .interval import IntervalEnclosure
from .generator import (
    SharedGeneratorPolynomialEnclosure, generator_matrix_from_values,
    extract_generator_polynomial_array,
)
from .piecewise import PiecewiseEnclosure
from .value import (
    ValidatedValue,
    same_domain,
    contains_validated,
    first_validated,
    unwrap_for_polynomial,
    unwrap_for_generator,
    unwrap_for_interval,
    all_active_representations,
)

__all__ = [
    "ParameterizedEnclosure",
    "PolynomialEnclosure",
    "IntervalEnclosure",
    "SharedGeneratorPolynomialEnclosure",
    "generator_matrix_from_values",
    "extract_generator_polynomial_array",
    "PiecewiseEnclosure",
    "ValidatedValue",
    "contains_pe",
    "extract_affine_array",
    "extract_polynomial_array",
    "contains_validated",
    "first_validated",
    "unwrap_for_polynomial",
    "unwrap_for_generator",
    "unwrap_for_interval",
    "all_active_representations",
    "same_domain",
    "interval_intersection",
    "safe_interval_representative",
    "RangePolicyContext",
    "record_range_policy_event",
    "active_range_policy",
    "ApproximationDiagnosticContext",
    "active_approximation_diagnostics",
    "new_approximation_source_mask",
    "record_approximation_recombination",
    "ValidatedPredicate", "BranchDecisionRequired", "InfeasiblePath",
    "BranchRuntime", "BranchExecutionContext", "active_branch_runtime",
    "DomainRequirement", "DomainProofResult", "DomainRefinementRequired",
    "DomainTraceEntry", "DomainRuntime", "DomainExecutionContext",
    "active_domain_runtime", "prove_domain", "require_domain",
    "elementary_requirement", "real_power_requirement",
]

from .branching import (
    ValidatedPredicate, BranchDecisionRequired, InfeasiblePath,
    BranchRuntime, BranchExecutionContext, active_branch_runtime,
)

from .domain import (
    DomainRequirement, DomainProofResult, DomainRefinementRequired,
    DomainTraceEntry, DomainRuntime, DomainExecutionContext,
    active_domain_runtime, prove_domain, require_domain,
    elementary_requirement, real_power_requirement,
)
