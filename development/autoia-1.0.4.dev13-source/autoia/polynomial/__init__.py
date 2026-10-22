"""Internal polynomial layer for AutoIA.

The names exported here are implementation-layer APIs used by the current
reference core.  They are not yet frozen as part of the AutoIA 1.0 public API.
"""

from .algebra import (
    zero_alpha, basis_alpha, total_degree, multi_indices_exact_degree,
    clean_poly, monomial_magnitude_upper, monomial_range_symmetric,
    point_power_interval, variable_power_range, monomial_range_with_fixed,
    bound_coefficient_error_poly, rigorous_poly_add_point,
    rigorous_poly_mul_truncated, build_reciprocal_predictor_point,
)
from .ranging import (
    range_termwise_scalar, range_polynomial_with_fixed,
    derivative_range_termwise, range_monotonicity_scalar, all_indices_leq,
    power_coeff_unit_box_interval, range_bernstein_scalar,
    power_coeff_reduced_unit_box_interval, range_bernstein_with_fixed_scalar,
    estimate_monotonicity_reduced_bernstein_cost,
    range_monotonicity_reduced_bernstein_scalar, range_polynomial,
    _degree_vector_and_bernstein_cost,
)

from .packed import (
    PolynomialRangePlan, PackedPointPolynomialFamily, PackedIntervalPolynomialFamily,
    compile_polynomial_range_plan, pack_point_polynomial_family,
    pack_interval_polynomial_family, monomial_range_endpoints,
    interval_objects_from_endpoints, range_packed_point_termwise,
    range_packed_interval_termwise, derivative_factor_endpoints,
    range_packed_point_derivatives, monotonicity_signs_from_derivative_endpoints,
    range_packed_point_monotonicity, dense_point_coefficient_tensor,
    bernstein_axis_transform_endpoints, packed_bernstein_coefficients,
    range_packed_point_bernstein, range_packed_point_reduced_bernstein,
    range_point_polynomial_family,
    interval_coefficient_representatives, range_interval_polynomial_family,
)

__all__ = [name for name in globals() if not name.startswith("_")]

from .recenter import rigorous_recenter_point_polynomial
