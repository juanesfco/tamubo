"""Validated semantic primitives."""

from .linear_solve import (
    build_solution_polynomials,
    residual_coefficient_intervals,
    abs_bound_interval_vector_poly,
    range_interval_coefficient_poly,
    left_multiply_interval_vector_polynomial,
    range_interval_vector_polynomial,
    predictor_abs_bounds,
    refine_componentwise_correction,
    SolveContext,
    interval_semantic_solve,
    GenericSolveContext,
)

__all__ = [
    "build_solution_polynomials",
    "residual_coefficient_intervals",
    "abs_bound_interval_vector_poly",
    "range_interval_coefficient_poly",
    "left_multiply_interval_vector_polynomial",
    "range_interval_vector_polynomial",
    "predictor_abs_bounds",
    "refine_componentwise_correction",
    "SolveContext",
    "interval_semantic_solve",
    "GenericSolveContext",
    "ImplicitDual",
    "ImplicitSolveContext",
    "ImplicitSolveExecutionContext",
    "CertifiedLinprogResult",
    "LinearProgramContext",
    "LinearProgramExecutionContext",
    "CertifiedRootScalarResult",
    "prepare_model_for_dispatch",
    "prepare_dispatch_model",
    "PreparedDispatchModel",
    "execute_model_with_dispatch",
    "SPDMatrixCertificate",
    "certify_spd_matrix",
    "validated_slogdet",
    "validated_logdet",
    "validated_det",
    "validated_quad_solve",
]

from .dispatch import (
    prepare_model_for_dispatch, prepare_dispatch_model, PreparedDispatchModel,
    execute_model_with_dispatch,
)

from .implicit_solve import (
    ImplicitDual, ImplicitSolveContext, ImplicitSolveExecutionContext,
    CertifiedRootScalarResult,
)

from .spd_matrix import (
    SPDMatrixCertificate, certify_spd_matrix, validated_slogdet,
    validated_logdet, validated_det, validated_quad_solve,
)

from .linear_program import (
    CertifiedLinprogResult, LinearProgramContext, LinearProgramExecutionContext,
)
