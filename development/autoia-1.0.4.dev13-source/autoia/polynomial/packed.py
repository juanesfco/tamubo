"""Numeric lowered forms for rigorous polynomial ranging.

The semantic AutoIA polynomial representation remains a sparse mapping
``multi_index -> coefficient``.  This module introduces a proof-kernel boundary
representation intended for repeated and batched range calculations.  Structural
information is compiled once into :class:`PolynomialRangePlan`; coefficient
families are packed into float64 structure-of-arrays buffers with the term axis
last so trusted pairwise reductions can process many independent polynomials at
once.

Only the physical centered variables ``q`` participate in these polynomial
buffers.  Auxiliary shared-error generators ``epsilon`` remain in their separate
generator representation and are deliberately absent from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb
from typing import Iterable, Mapping

import numpy as np

from ..trusted import (
    Interval, integer_interval,
    rigorous_point_interval_sum_axis_last,
    rigorous_interval_product_sum_axis_last,
    rigorous_interval_subtract_point_arrays,
)
from .algebra import (
    monomial_range_symmetric, monomial_range_with_fixed, point_power_interval,
)
from .ranging import range_polynomial


@dataclass(frozen=True)
class PolynomialRangePlan:
    """Immutable structural plan shared by a family of sparse polynomials.

    Parameters
    ----------
    support:
        Canonically ordered tuple of physical-variable multi-indices.
    exponents:
        Integer array with shape ``(T, m)``. ``T`` is the number of retained
        monomials and ``m`` is the number of physical uncertain inputs.
    degree_vector:
        Maximum exponent in each physical coordinate.
    even_monomial:
        Boolean vector identifying monomials whose exponents are all even.
    derivative_support:
        Canonical union of all first-derivative physical monomials.
    derivative_term_index:
        Integer array with shape ``(m, T)`` mapping each active original term
        and physical derivative direction to ``derivative_support``; ``-1``
        marks an inactive derivative contribution.
    derivative_multipliers:
        Integer array with shape ``(m, T)`` containing the exact exponent
        multiplier for each first-derivative contribution.
    dense_shape:
        Full tensor-product power-basis shape ``(d_1+1, ..., d_m+1)`` where
        ``d_i`` is the maximum retained degree in physical coordinate ``i``.
    dense_term_index:
        Flat C-order destination index of each sparse support term inside the
        dense tensor-product coefficient buffer.

    The plan contains no coefficient values, box widths, auxiliary generators,
    or mutable certification state.  It is therefore safe to reuse whenever the
    physical polynomial support is unchanged.
    """

    support: tuple[tuple[int, ...], ...]
    exponents: np.ndarray
    degree_vector: tuple[int, ...]
    even_monomial: np.ndarray
    derivative_support: tuple[tuple[int, ...], ...]
    derivative_term_index: np.ndarray
    derivative_multipliers: np.ndarray
    dense_shape: tuple[int, ...]
    dense_term_index: np.ndarray

    @property
    def m(self) -> int:
        return len(self.degree_vector)

    @property
    def term_count(self) -> int:
        return len(self.support)


@dataclass(frozen=True)
class PackedPointPolynomialFamily:
    """Point-coefficient polynomial family with final axis equal to terms."""

    plan: PolynomialRangePlan
    coefficients: np.ndarray
    family_shape: tuple[int, ...]


@dataclass(frozen=True)
class PackedIntervalPolynomialFamily:
    """Interval-coefficient polynomial family stored as endpoint arrays."""

    plan: PolynomialRangePlan
    lower: np.ndarray
    upper: np.ndarray
    family_shape: tuple[int, ...]


def _canonical_support(support: Iterable[tuple[int, ...]], m: int):
    m = int(m)
    if m < 0:
        raise ValueError("Polynomial dimension m must be nonnegative.")
    canonical = set()
    for alpha in support:
        alpha = tuple(int(v) for v in alpha)
        if len(alpha) != m:
            raise ValueError("Every polynomial multi-index must have length m.")
        if any(v < 0 for v in alpha):
            raise ValueError("Polynomial exponents must be nonnegative integers.")
        canonical.add(alpha)
    return tuple(sorted(canonical))


@lru_cache(maxsize=512)
def _compile_polynomial_range_plan_cached(
    support: tuple[tuple[int, ...], ...], m: int
) -> PolynomialRangePlan:
    exponents = np.asarray(support, dtype=np.int64)
    if len(support) == 0:
        exponents = np.empty((0, int(m)), dtype=np.int64)
        degrees = (0,) * int(m)
        even = np.empty(0, dtype=bool)
    else:
        exponents = exponents.reshape(len(support), int(m))
        degrees = tuple(int(v) for v in np.max(exponents, axis=0))
        even = np.all((exponents % 2) == 0, axis=1)
    derivative_support_set = set()
    for alpha in support:
        for i, k in enumerate(alpha):
            if int(k) <= 0:
                continue
            beta = list(alpha)
            beta[i] -= 1
            derivative_support_set.add(tuple(beta))
    derivative_support = tuple(sorted(derivative_support_set))
    derivative_lookup = {beta: j for j, beta in enumerate(derivative_support)}
    derivative_term_index = np.full((int(m), len(support)), -1, dtype=np.int64)
    derivative_multipliers = np.zeros((int(m), len(support)), dtype=np.int64)
    for t, alpha in enumerate(support):
        for i, k in enumerate(alpha):
            k = int(k)
            if k <= 0:
                continue
            beta = list(alpha)
            beta[i] -= 1
            beta = tuple(beta)
            derivative_term_index[i, t] = int(derivative_lookup[beta])
            derivative_multipliers[i, t] = k

    exponents.setflags(write=False)
    even.setflags(write=False)
    dense_shape = tuple(int(d) + 1 for d in degrees)
    if len(support) == 0:
        dense_term_index = np.empty(0, dtype=np.int64)
    elif int(m) == 0:
        dense_term_index = np.zeros(len(support), dtype=np.int64)
    else:
        dense_term_index = np.ravel_multi_index(
            np.asarray(support, dtype=np.int64).T, dense_shape, order="C"
        ).astype(np.int64, copy=False)

    derivative_term_index.setflags(write=False)
    derivative_multipliers.setflags(write=False)
    dense_term_index.setflags(write=False)
    return PolynomialRangePlan(
        support=support,
        exponents=exponents,
        degree_vector=degrees,
        even_monomial=even,
        derivative_support=derivative_support,
        derivative_term_index=derivative_term_index,
        derivative_multipliers=derivative_multipliers,
        dense_shape=dense_shape,
        dense_term_index=dense_term_index,
    )


def compile_polynomial_range_plan(
    support: Iterable[tuple[int, ...]], m: int
) -> PolynomialRangePlan:
    """Compile or reuse an immutable structural polynomial range plan."""
    canonical = _canonical_support(support, m)
    return _compile_polynomial_range_plan_cached(canonical, int(m))


def _common_family_shape(coeff_arrays: Mapping, support) -> tuple[int, ...]:
    shape = None
    for alpha in support:
        if alpha not in coeff_arrays:
            continue
        arr = np.asarray(coeff_arrays[alpha], dtype=object)
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ValueError("All coefficient arrays in a polynomial family must match in shape.")
    return tuple() if shape is None else tuple(shape)


def pack_point_polynomial_family(
    coeff_arrays: Mapping[tuple[int, ...], object],
    m: int,
    *,
    plan: PolynomialRangePlan | None = None,
) -> PackedPointPolynomialFamily:
    """Lower a sparse point-coefficient polynomial family to float64 buffers.

    ``coeff_arrays[alpha]`` may be a scalar or an arbitrary same-shaped array.
    The returned coefficient buffer has shape ``family_shape + (T,)``.  A future
    many-box axis is therefore just another leading family axis; the kernel does
    not need a separate one-box implementation.
    """
    if plan is None:
        plan = compile_polynomial_range_plan(coeff_arrays.keys(), m)
    elif plan.m != int(m):
        raise ValueError("Range plan dimension does not match m.")

    family_shape = _common_family_shape(coeff_arrays, plan.support)
    coeff = np.zeros(family_shape + (plan.term_count,), dtype=np.float64)
    for t, alpha in enumerate(plan.support):
        if alpha not in coeff_arrays:
            continue
        arr = np.asarray(coeff_arrays[alpha], dtype=np.float64)
        if arr.shape != family_shape:
            raise ValueError("All coefficient arrays in a polynomial family must match in shape.")
        if np.any(~np.isfinite(arr)):
            raise FloatingPointError("Point polynomial coefficients must be finite binary64 values.")
        coeff[..., t] = arr
    return PackedPointPolynomialFamily(plan, coeff, family_shape)


def pack_interval_polynomial_family(
    coeff_arrays: Mapping[tuple[int, ...], object],
    m: int,
    *,
    plan: PolynomialRangePlan | None = None,
) -> PackedIntervalPolynomialFamily:
    """Lower a sparse interval-coefficient family into endpoint buffers."""
    if plan is None:
        plan = compile_polynomial_range_plan(coeff_arrays.keys(), m)
    elif plan.m != int(m):
        raise ValueError("Range plan dimension does not match m.")

    family_shape = _common_family_shape(coeff_arrays, plan.support)
    lower = np.zeros(family_shape + (plan.term_count,), dtype=np.float64)
    upper = np.zeros_like(lower)

    for t, alpha in enumerate(plan.support):
        if alpha not in coeff_arrays:
            continue
        arr = np.asarray(coeff_arrays[alpha], dtype=object)
        if arr.shape != family_shape:
            raise ValueError("All coefficient arrays in a polynomial family must match in shape.")
        for idx in np.ndindex(family_shape):
            value = arr[idx]
            if not isinstance(value, Interval):
                raise TypeError("Interval polynomial families require Interval coefficients.")
            lower[idx + (t,)] = value.lo
            upper[idx + (t,)] = value.hi
        if family_shape == tuple():
            value = arr.item()
            if not isinstance(value, Interval):
                raise TypeError("Interval polynomial families require Interval coefficients.")
            lower[t] = value.lo
            upper[t] = value.hi

    return PackedIntervalPolynomialFamily(plan, lower, upper, family_shape)


def monomial_range_endpoints(plan: PolynomialRangePlan, h, backend):
    """Compute one rigorous physical-box monomial range per plan term."""
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    if h.size != plan.m:
        raise ValueError("Half-width vector length does not match polynomial dimension.")
    if np.any(~np.isfinite(h)) or np.any(h < 0.0):
        raise ValueError("Half-widths must be finite and nonnegative.")
    lo = np.empty(plan.term_count, dtype=np.float64)
    hi = np.empty(plan.term_count, dtype=np.float64)
    for t, alpha in enumerate(plan.support):
        value = monomial_range_symmetric(alpha, h, backend)
        lo[t] = value.lo
        hi[t] = value.hi
    return lo, hi


def interval_objects_from_endpoints(lo, hi):
    """Materialize endpoint buffers only at an object/API boundary."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if lo.shape != hi.shape:
        raise ValueError("Endpoint arrays must have the same shape.")
    out = np.empty(lo.shape, dtype=object)
    for idx in np.ndindex(lo.shape):
        out[idx] = Interval(float(lo[idx]), float(hi[idx]))
    if lo.shape == tuple():
        return Interval(float(lo), float(hi))
    return out


def range_packed_point_termwise(
    packed: PackedPointPolynomialFamily, h, backend, *, monomial_endpoints=None
):
    """Termwise range every point polynomial in a packed family."""
    if monomial_endpoints is None:
        monomial_endpoints = monomial_range_endpoints(packed.plan, h, backend)
    mlo, mhi = monomial_endpoints
    return rigorous_point_interval_sum_axis_last(
        packed.coefficients,
        np.asarray(mlo, dtype=np.float64),
        np.asarray(mhi, dtype=np.float64),
        backend,
    )


def range_packed_interval_termwise(
    packed: PackedIntervalPolynomialFamily, h, backend, *, monomial_endpoints=None
):
    """Natural termwise range every interval polynomial in a packed family."""
    if monomial_endpoints is None:
        monomial_endpoints = monomial_range_endpoints(packed.plan, h, backend)
    mlo, mhi = monomial_endpoints
    return rigorous_interval_product_sum_axis_last(
        packed.lower,
        packed.upper,
        np.asarray(mlo, dtype=np.float64),
        np.asarray(mhi, dtype=np.float64),
        backend,
    )



def derivative_factor_endpoints(plan: PolynomialRangePlan, h, backend):
    """Compile rigorous derivative monomial factors for one physical box.

    For original term ``t`` with multi-index ``alpha`` and physical coordinate
    ``i``, the derivative contribution is

        c_t * alpha_i * q**(alpha - e_i).

    This function returns rigorous endpoint arrays for the factor
    ``alpha_i * q**(alpha-e_i)`` with shape ``(m, T)``.  Inactive derivative
    slots are exact zero.  The expensive monomial range for a derivative
    multi-index is computed once from the union support compiled into the plan.
    """
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    if h.size != plan.m:
        raise ValueError("Half-width vector length does not match polynomial dimension.")
    if np.any(~np.isfinite(h)) or np.any(h < 0.0):
        raise ValueError("Half-widths must be finite and nonnegative.")

    support_lo = np.empty(len(plan.derivative_support), dtype=np.float64)
    support_hi = np.empty(len(plan.derivative_support), dtype=np.float64)
    for j, beta in enumerate(plan.derivative_support):
        value = monomial_range_symmetric(beta, h, backend)
        support_lo[j] = value.lo
        support_hi[j] = value.hi

    lo = np.zeros((plan.m, plan.term_count), dtype=np.float64)
    hi = np.zeros_like(lo)
    for i in range(plan.m):
        for t in range(plan.term_count):
            j = int(plan.derivative_term_index[i, t])
            if j < 0:
                continue
            k = int(plan.derivative_multipliers[i, t])
            value = backend.mul(
                integer_interval(k),
                Interval(float(support_lo[j]), float(support_hi[j])),
            )
            lo[i, t] = value.lo
            hi[i, t] = value.hi
    return lo, hi


def range_packed_point_derivatives(
    packed: PackedPointPolynomialFamily, h, backend, *, derivative_factors=None
):
    """Range all first physical partial derivatives of a packed family.

    If the polynomial family has leading shape ``F``, the returned endpoint
    arrays have shape ``F + (m,)``.  The final axis indexes derivatives with
    respect to the physical centered coordinates ``q_i``.  Auxiliary shared
    error generators are not differentiation variables.
    """
    if derivative_factors is None:
        derivative_factors = derivative_factor_endpoints(packed.plan, h, backend)
    flo, fhi = derivative_factors
    flo = np.asarray(flo, dtype=np.float64)
    fhi = np.asarray(fhi, dtype=np.float64)
    expected = (packed.plan.m, packed.plan.term_count)
    if flo.shape != expected or fhi.shape != expected:
        raise ValueError("Derivative factor endpoint shape does not match range plan.")

    coefficients = np.expand_dims(packed.coefficients, axis=-2)
    return rigorous_point_interval_sum_axis_last(
        coefficients, flo, fhi, backend
    )


def monotonicity_signs_from_derivative_endpoints(derivative_lo, derivative_hi):
    """Classify rigorous derivative intervals using the historical sign rule.

    ``+1`` means the derivative interval is nonnegative, ``-1`` means it is
    nonpositive, and ``0`` means its sign is unresolved.  The exact-zero
    interval is classified ``+1`` because the scalar Version-0.11 rule checks
    nonnegativity first.
    """
    derivative_lo, derivative_hi = np.broadcast_arrays(
        np.asarray(derivative_lo, dtype=np.float64),
        np.asarray(derivative_hi, dtype=np.float64),
    )
    if np.any(derivative_lo > derivative_hi):
        raise ValueError("Invalid derivative interval endpoints.")
    signs = np.zeros(derivative_lo.shape, dtype=np.int8)
    nonnegative = derivative_lo >= 0.0
    nonpositive = (~nonnegative) & (derivative_hi <= 0.0)
    signs[nonnegative] = 1
    signs[nonpositive] = -1
    return signs


def _fixed_monomial_endpoints_for_sign_patterns(plan, h, signs, backend):
    """Build lower/upper extremum monomial factors for rigorous sign patterns."""
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    signs = np.asarray(signs, dtype=np.int8)
    if signs.shape[-1:] != (plan.m,):
        raise ValueError("Monotonicity sign array must end with the physical dimension m.")
    family_shape = signs.shape[:-1]
    if plan.m == 0:
        base_lo = np.ones(family_shape + (plan.term_count,), dtype=np.float64)
        base_hi = np.ones_like(base_lo)
        return base_lo, base_hi, base_lo.copy(), base_hi.copy(), np.empty((0, 0), dtype=np.int8)

    flat = signs.reshape((-1, plan.m))
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    U = unique.shape[0]
    lower_lo = np.empty((U, plan.term_count), dtype=np.float64)
    lower_hi = np.empty_like(lower_lo)
    upper_lo = np.empty_like(lower_lo)
    upper_hi = np.empty_like(lower_lo)

    for u, pattern in enumerate(unique):
        lower_fixed = {}
        upper_fixed = {}
        for i, sign in enumerate(pattern):
            if int(sign) > 0:
                lower_fixed[i] = -float(h[i])
                upper_fixed[i] = +float(h[i])
            elif int(sign) < 0:
                lower_fixed[i] = +float(h[i])
                upper_fixed[i] = -float(h[i])

        for t, alpha in enumerate(plan.support):
            lv = monomial_range_with_fixed(alpha, h, lower_fixed, backend)
            uv = monomial_range_with_fixed(alpha, h, upper_fixed, backend)
            lower_lo[u, t] = lv.lo
            lower_hi[u, t] = lv.hi
            upper_lo[u, t] = uv.lo
            upper_hi[u, t] = uv.hi

    gather_shape = family_shape + (plan.term_count,)
    lower_lo = lower_lo[inverse].reshape(gather_shape)
    lower_hi = lower_hi[inverse].reshape(gather_shape)
    upper_lo = upper_lo[inverse].reshape(gather_shape)
    upper_hi = upper_hi[inverse].reshape(gather_shape)
    return lower_lo, lower_hi, upper_lo, upper_hi, unique


def range_packed_point_monotonicity(
    packed: PackedPointPolynomialFamily,
    h,
    backend,
    *,
    derivative_endpoints=None,
):
    """Rigorous batched analogue of ``range_monotonicity_scalar``.

    First derivatives are ranged together.  Their rigorous signs determine
    which physical coordinates may be fixed at opposite endpoints for the
    lower and upper extremum problems.  Polynomials sharing a sign pattern also
    share the proof-critical fixed-monomial factor table.
    """
    if derivative_endpoints is None:
        derivative_endpoints = range_packed_point_derivatives(packed, h, backend)
    derivative_lo, derivative_hi = derivative_endpoints
    signs = monotonicity_signs_from_derivative_endpoints(derivative_lo, derivative_hi)

    lower_mlo, lower_mhi, upper_mlo, upper_mhi, unique_sign_patterns = (
        _fixed_monomial_endpoints_for_sign_patterns(
            packed.plan, h, signs, backend
        )
    )
    lower_lo, lower_hi = rigorous_point_interval_sum_axis_last(
        packed.coefficients, lower_mlo, lower_mhi, backend
    )
    upper_lo, upper_hi = rigorous_point_interval_sum_axis_last(
        packed.coefficients, upper_mlo, upper_mhi, backend
    )
    final_lo = np.asarray(lower_lo, dtype=np.float64)
    final_hi = np.asarray(upper_hi, dtype=np.float64)
    if np.any(final_lo > final_hi):
        raise RuntimeError(
            "Packed monotonicity produced an inverted enclosure; treat as an internal error."
        )
    return final_lo, final_hi, {
        "signs": signs,
        "derivative_lower": np.asarray(derivative_lo, dtype=np.float64),
        "derivative_upper": np.asarray(derivative_hi, dtype=np.float64),
        "lower_fixed_range_lower": np.asarray(lower_lo, dtype=np.float64),
        "lower_fixed_range_upper": np.asarray(lower_hi, dtype=np.float64),
        "upper_fixed_range_lower": np.asarray(upper_lo, dtype=np.float64),
        "upper_fixed_range_upper": np.asarray(upper_hi, dtype=np.float64),
        "unique_sign_patterns": np.asarray(unique_sign_patterns, dtype=np.int8),
    }


def _packed_monotonicity_meta_at(meta, idx):
    signs = np.asarray(meta["signs"][idx], dtype=np.int8).reshape(-1)
    dlo = np.asarray(meta["derivative_lower"][idx], dtype=np.float64).reshape(-1)
    dhi = np.asarray(meta["derivative_upper"][idx], dtype=np.float64).reshape(-1)
    return {
        "signs": [int(v) for v in signs],
        "derivative_ranges": [
            Interval(float(lo), float(hi)) for lo, hi in zip(dlo, dhi)
        ],
        "lowered_execution": "packed-family",
    }




def dense_point_coefficient_tensor(packed: PackedPointPolynomialFamily):
    """Return the common-support family in a dense tensor-product power basis.

    The returned array has shape ``family_shape + plan.dense_shape``.  Sparse
    semantic storage is preserved outside this proof-kernel boundary; dense
    storage is used here because Bernstein coefficients occupy the complete
    tensor-product degree grid even when the original polynomial is sparse.
    """
    dense_size = int(np.prod(packed.plan.dense_shape, dtype=np.int64)) if packed.plan.dense_shape else 1
    flat = np.zeros(packed.family_shape + (dense_size,), dtype=np.float64)
    if packed.plan.term_count:
        flat[..., packed.plan.dense_term_index] = packed.coefficients
    return flat.reshape(packed.family_shape + packed.plan.dense_shape)


def _endpoint_power_endpoints(value: float, degree: int, backend):
    """Rigorous intervals enclosing ``value**k`` for ``k=0,...,degree``."""
    degree = int(degree)
    lo = np.empty(degree + 1, dtype=np.float64)
    hi = np.empty(degree + 1, dtype=np.float64)
    for k in range(degree + 1):
        v = point_power_interval(float(value), k, backend)
        lo[k] = v.lo
        hi[k] = v.hi
    return lo, hi


def bernstein_axis_transform_endpoints(degree: int, h: float, backend):
    r"""Return a rigorous one-dimensional power-to-Bernstein transform.

    Let ``d = degree`` and let

    ``p(q) = sum_{a=0}^d c_a q^a`` on ``q in [-h,h]``.

    After ``q = -h + 2 h t``, ``t in [0,1]``, the exact degree-``d`` Bernstein
    coefficients satisfy

    ``b_g = sum_a W[g,a] c_a``.

    Each returned interval ``[W_lo[g,a], W_hi[g,a]]`` encloses the exact real
    transform entry

    ``sum_{b=0}^{min(a,g)} C(a,b)(-h)^(a-b)(2h)^b C(g,b)/C(d,b)``.

    Only this small ``(d+1) x (d+1)`` matrix is assembled with scalar trusted
    arithmetic.  The expensive application to a polynomial family is handled by
    batched endpoint-array reductions.
    """
    degree = int(degree)
    h = float(h)
    if degree < 0:
        raise ValueError("Bernstein degree must be nonnegative.")
    if not np.isfinite(h) or h < 0.0:
        raise ValueError("Bernstein half-width must be finite and nonnegative.")
    if degree == 0:
        return np.ones((1, 1), dtype=np.float64), np.ones((1, 1), dtype=np.float64)

    neg_h = [point_power_interval(-h, k, backend) for k in range(degree + 1)]
    two_h = backend.mul(Interval(2.0, 2.0), Interval(h, h))
    two_h_pow = [Interval(1.0, 1.0)]
    for _ in range(degree):
        two_h_pow.append(backend.mul(two_h_pow[-1], two_h))

    # q_to_t[b,a] encloses the coefficient multiplying c_a in the t^b
    # power coefficient after q=-h+2ht.
    q_to_t = [[Interval(0.0, 0.0) for _ in range(degree + 1)]
              for _ in range(degree + 1)]
    for a in range(degree + 1):
        for b in range(a + 1):
            v = backend.mul(integer_interval(comb(a, b)), neg_h[a - b])
            v = backend.mul(v, two_h_pow[b])
            q_to_t[b][a] = v

    # power_to_bern[g,b] is the exact rational coefficient C(g,b)/C(d,b),
    # enclosed through trusted integer division.
    power_to_bern = [[Interval(0.0, 0.0) for _ in range(degree + 1)]
                     for _ in range(degree + 1)]
    for g in range(degree + 1):
        for b in range(g + 1):
            power_to_bern[g][b] = backend.div(
                integer_interval(comb(g, b)),
                integer_interval(comb(degree, b)),
            )

    lo = np.empty((degree + 1, degree + 1), dtype=np.float64)
    hi = np.empty_like(lo)
    for g in range(degree + 1):
        for a in range(degree + 1):
            total = Interval(0.0, 0.0)
            for b in range(min(a, g) + 1):
                total = backend.add(
                    total,
                    backend.mul(power_to_bern[g][b], q_to_t[b][a]),
                )
            lo[g, a] = total.lo
            hi[g, a] = total.hi
    return lo, hi


def _apply_interval_axis_transform(lo, hi, transform_lo, transform_hi, axis, backend):
    """Apply one rigorous interval linear transform along an arbitrary axis."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if lo.shape != hi.shape:
        raise ValueError("Coefficient endpoint arrays must have matching shapes.")
    axis = int(axis)
    if axis < 0:
        axis += lo.ndim
    if not 0 <= axis < lo.ndim:
        raise ValueError("Transform axis is out of range.")
    tlo = np.asarray(transform_lo, dtype=np.float64)
    thi = np.asarray(transform_hi, dtype=np.float64)
    if tlo.shape != thi.shape or tlo.ndim != 2:
        raise ValueError("Transform endpoint arrays must be matching matrices.")
    if tlo.shape[1] != lo.shape[axis]:
        raise ValueError("Transform input dimension does not match coefficient axis.")

    moved_lo = np.moveaxis(lo, axis, -1)
    moved_hi = np.moveaxis(hi, axis, -1)
    out_lo, out_hi = rigorous_interval_product_sum_axis_last(
        np.expand_dims(moved_lo, axis=-2),
        np.expand_dims(moved_hi, axis=-2),
        tlo,
        thi,
        backend,
    )
    return np.moveaxis(out_lo, -1, axis), np.moveaxis(out_hi, -1, axis)


def _reduce_interval_axis_with_factors(lo, hi, factor_lo, factor_hi, axis, backend):
    """Rigorously contract one coefficient axis against interval factors."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if lo.shape != hi.shape:
        raise ValueError("Coefficient endpoint arrays must have matching shapes.")
    axis = int(axis)
    if axis < 0:
        axis += lo.ndim
    moved_lo = np.moveaxis(lo, axis, -1)
    moved_hi = np.moveaxis(hi, axis, -1)
    flo = np.asarray(factor_lo, dtype=np.float64)
    fhi = np.asarray(factor_hi, dtype=np.float64)
    if flo.shape != fhi.shape or flo.shape[-1:] != (moved_lo.shape[-1],):
        raise ValueError("Fixed-axis factor endpoints do not match coefficient degree.")
    return rigorous_interval_product_sum_axis_last(
        moved_lo, moved_hi, flo, fhi, backend
    )


def packed_bernstein_coefficients(
    packed: PackedPointPolynomialFamily,
    h,
    backend,
    *,
    fixed: Mapping[int, float] | None = None,
    unresolved: Iterable[int] | None = None,
    _axis_transform_cache=None,
):
    """Compute rigorous Bernstein coefficient intervals for a packed family.

    ``fixed`` maps physical coordinate indices to exact stored binary64 endpoint
    values.  Coordinates in ``unresolved`` remain variables and are converted to
    Bernstein form.  The two sets must partition all physical coordinates.

    The returned endpoint arrays have shape

    ``family_shape + tuple(d_i+1 for i in unresolved)``.

    This routine is deliberately lower level than a range calculation so exact-
    rational audits can test the proof-bearing Bernstein coefficients directly.
    """
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    if h.size != packed.plan.m:
        raise ValueError("Half-width vector length does not match polynomial dimension.")
    if np.any(~np.isfinite(h)) or np.any(h < 0.0):
        raise ValueError("Half-widths must be finite and nonnegative.")

    fixed = {} if fixed is None else {int(i): float(v) for i, v in fixed.items()}
    if unresolved is None:
        unresolved = tuple(i for i in range(packed.plan.m) if i not in fixed)
    else:
        unresolved = tuple(int(i) for i in unresolved)
    if len(set(unresolved)) != len(unresolved):
        raise ValueError("Unresolved physical coordinates must be unique.")
    if any(i < 0 or i >= packed.plan.m for i in unresolved):
        raise ValueError("Unresolved physical coordinate is out of range.")
    if any(i < 0 or i >= packed.plan.m for i in fixed):
        raise ValueError("Fixed physical coordinate is out of range.")
    if set(fixed).intersection(unresolved):
        raise ValueError("A physical coordinate cannot be both fixed and unresolved.")
    if set(fixed).union(unresolved) != set(range(packed.plan.m)):
        raise ValueError("Fixed and unresolved coordinates must partition physical coordinates.")
    if tuple(sorted(unresolved)) != unresolved:
        raise ValueError("Unresolved physical coordinates must be in increasing order.")

    coeff = dense_point_coefficient_tensor(packed)
    lo = np.asarray(coeff, dtype=np.float64)
    hi = np.asarray(coeff, dtype=np.float64)
    family_ndim = len(packed.family_shape)
    active_variables = list(range(packed.plan.m))

    # Fold fixed physical coordinates first.  Descending order preserves the
    # original axis positions of all lower-numbered coordinates as axes vanish.
    for i in sorted(fixed, reverse=True):
        pos = active_variables.index(i)
        axis = family_ndim + pos
        degree = int(packed.plan.degree_vector[i])
        flo, fhi = _endpoint_power_endpoints(fixed[i], degree, backend)
        lo, hi = _reduce_interval_axis_with_factors(
            lo, hi, flo, fhi, axis, backend
        )
        active_variables.pop(pos)

    if tuple(active_variables) != unresolved:
        raise RuntimeError("Internal Bernstein variable bookkeeping mismatch.")

    # Each remaining physical coordinate is transformed independently.  This is
    # the dimension-separable core: no nested enumeration of multivariate beta /
    # gamma multi-indices occurs in the family hot path.
    if _axis_transform_cache is None:
        _axis_transform_cache = {}
    for pos, i in enumerate(active_variables):
        degree = int(packed.plan.degree_vector[i])
        cache_key = (int(i), degree, float(h[i]))
        if cache_key not in _axis_transform_cache:
            _axis_transform_cache[cache_key] = bernstein_axis_transform_endpoints(
                degree, float(h[i]), backend
            )
        wlo, whi = _axis_transform_cache[cache_key]
        axis = family_ndim + pos
        lo, hi = _apply_interval_axis_transform(
            lo, hi, wlo, whi, axis, backend
        )

    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64), {
        "degree_vector": tuple(int(packed.plan.degree_vector[i]) for i in unresolved),
        "unresolved_variables": unresolved,
        "fixed_variables": tuple(sorted(fixed)),
        "num_coefficients": int(np.prod(
            [int(packed.plan.degree_vector[i]) + 1 for i in unresolved],
            dtype=np.int64,
        )) if unresolved else 0,
        "lowered_execution": "packed-separable-bernstein",
    }


def _bernstein_range_from_coefficients(lo, hi, polynomial_ndim):
    """Take the Bernstein convex-hull range over trailing polynomial axes."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    polynomial_ndim = int(polynomial_ndim)
    if polynomial_ndim < 0 or polynomial_ndim > lo.ndim:
        raise ValueError("Invalid Bernstein polynomial-axis count.")
    if polynomial_ndim == 0:
        return lo, hi
    axes = tuple(range(lo.ndim - polynomial_ndim, lo.ndim))
    return np.min(lo, axis=axes), np.max(hi, axis=axes)


def range_packed_point_bernstein(packed: PackedPointPolynomialFamily, h, backend):
    """Rigorous full Bernstein convex-hull range for a packed family."""
    blo, bhi, meta = packed_bernstein_coefficients(
        packed, h, backend, fixed={}, unresolved=tuple(range(packed.plan.m))
    )
    lo, hi = _bernstein_range_from_coefficients(blo, bhi, packed.plan.m)
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64), {
        **meta,
        "degree_vector": tuple(packed.plan.degree_vector),
        "num_coefficients": int(np.prod(packed.plan.dense_shape, dtype=np.int64))
            if packed.plan.dense_shape else 1,
        "method": "bernstein",
    }


def range_packed_point_reduced_bernstein(
    packed: PackedPointPolynomialFamily,
    h,
    backend,
    *,
    monotonicity_meta=None,
):
    """Rigorous monotonicity-reduced Bernstein range for a packed family.

    Family members are grouped by their already-certified physical monotonicity
    sign pattern.  Proven-monotone coordinates are fixed at opposite endpoints
    for the lower and upper extremum problems; only unresolved coordinates enter
    the dimension-separable Bernstein transform.
    """
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    if monotonicity_meta is None:
        _, _, monotonicity_meta = range_packed_point_monotonicity(
            packed, h, backend
        )
    signs = np.asarray(monotonicity_meta["signs"], dtype=np.int8)
    expected_shape = packed.family_shape + (packed.plan.m,)
    if signs.shape != expected_shape:
        raise ValueError("Packed monotonicity signs do not match polynomial family shape.")

    flat_coeff = packed.coefficients.reshape((-1, packed.plan.term_count))
    flat_signs = signs.reshape((-1, packed.plan.m))
    count = flat_coeff.shape[0]
    final_lo = np.empty(count, dtype=np.float64)
    final_hi = np.empty(count, dtype=np.float64)
    per_member = np.empty(count, dtype=object)

    if packed.plan.m == 0:
        unique = np.empty((1, 0), dtype=np.int8)
        inverse = np.zeros(count, dtype=np.int64)
    else:
        unique, inverse = np.unique(flat_signs, axis=0, return_inverse=True)

    full_cost = int(np.prod(packed.plan.dense_shape, dtype=np.int64)) \
        if packed.plan.dense_shape else 1
    axis_transform_cache = {}

    for u, pattern in enumerate(unique):
        members = np.flatnonzero(inverse == u)
        unresolved = tuple(i for i, sign in enumerate(pattern) if int(sign) == 0)
        lower_fixed = {}
        upper_fixed = {}
        for i, sign in enumerate(pattern):
            if int(sign) > 0:
                lower_fixed[i] = -float(h[i])
                upper_fixed[i] = +float(h[i])
            elif int(sign) < 0:
                lower_fixed[i] = +float(h[i])
                upper_fixed[i] = -float(h[i])

        sub = PackedPointPolynomialFamily(
            packed.plan,
            flat_coeff[members],
            (len(members),),
        )
        if len(unresolved) == packed.plan.m:
            # No monotonicity dimension can be removed.  The lower and upper
            # extremum problems are identical, so one full transform serves
            # both sides rather than computing it twice.
            full_blo, full_bhi, lower_meta = packed_bernstein_coefficients(
                sub, h, backend, fixed={}, unresolved=unresolved,
                _axis_transform_cache=axis_transform_cache,
            )
            upper_meta = lower_meta
            lower_range_lo, upper_range_hi = _bernstein_range_from_coefficients(
                full_blo, full_bhi, len(unresolved)
            )
        else:
            lower_blo, lower_bhi, lower_meta = packed_bernstein_coefficients(
                sub, h, backend, fixed=lower_fixed, unresolved=unresolved,
                _axis_transform_cache=axis_transform_cache,
            )
            upper_blo, upper_bhi, upper_meta = packed_bernstein_coefficients(
                sub, h, backend, fixed=upper_fixed, unresolved=unresolved,
                _axis_transform_cache=axis_transform_cache,
            )
            lower_range_lo, _ = _bernstein_range_from_coefficients(
                lower_blo, lower_bhi, len(unresolved)
            )
            _, upper_range_hi = _bernstein_range_from_coefficients(
                upper_blo, upper_bhi, len(unresolved)
            )
        final_lo[members] = np.asarray(lower_range_lo, dtype=np.float64).reshape(-1)
        final_hi[members] = np.asarray(upper_range_hi, dtype=np.float64).reshape(-1)

        ncoef_side = int(lower_meta["num_coefficients"])
        if not unresolved:
            reported_coefficients = 0
        elif len(unresolved) == packed.plan.m:
            reported_coefficients = ncoef_side
        else:
            reported_coefficients = 2 * ncoef_side
        for member in members:
            per_member[member] = {
                "method": "monotonicity-reduced-bernstein",
                "unresolved_variables": unresolved,
                "num_coefficients": int(reported_coefficients),
                "full_bernstein_coefficients": full_cost,
                "degree_vector": tuple(packed.plan.degree_vector),
                "monotonicity_complete": len(unresolved) == 0,
                "lower_meta": lower_meta,
                "upper_meta": upper_meta,
                "lowered_execution": "packed-separable-reduced-bernstein",
            }

    if np.any(final_lo > final_hi):
        raise RuntimeError(
            "Packed reduced Bernstein produced an inverted enclosure; treat as an internal error."
        )
    return (
        final_lo.reshape(packed.family_shape),
        final_hi.reshape(packed.family_shape),
        per_member.reshape(packed.family_shape),
    )


def _packed_reduced_bernstein_meta_at(meta, idx):
    value = meta[idx]
    if isinstance(value, np.ndarray) and value.shape == tuple():
        value = value.item()
    return value


def range_point_polynomial_family(
    coeff_arrays, h, backend, method="portfolio", *, plan: PolynomialRangePlan | None = None
):
    """Range a same-shaped family of point-coefficient physical polynomials.

    ``termwise`` and ``monotonicity`` use the packed proof kernels from Step-9d
    phases 1--2.  ``bernstein`` uses the phase-3 dimension-separable full
    Bernstein transform.  ``portfolio`` intersects packed termwise,
    monotonicity, and monotonicity-reduced Bernstein enclosures; the historical
    scalar routines remain unchanged in :mod:`autoia.polynomial.ranging` as the
    independent regression oracle.
    """
    m = len(h)
    packed = pack_point_polynomial_family(coeff_arrays, m, plan=plan)
    indices = list(np.ndindex(packed.family_shape)) if packed.family_shape else [tuple()]
    meta = np.empty(packed.family_shape, dtype=object)

    if method == "termwise":
        lo, hi = range_packed_point_termwise(packed, h, backend)
        for idx in indices:
            meta[idx] = {
                "method": "termwise",
                "lowered_execution": "packed-family",
                "term_count": packed.plan.term_count,
            }
    elif method == "monotonicity":
        lo, hi, mm = range_packed_point_monotonicity(packed, h, backend)
        for idx in indices:
            meta[idx] = {
                "method": "monotonicity",
                **_packed_monotonicity_meta_at(mm, idx),
                "term_count": packed.plan.term_count,
            }
    elif method == "bernstein":
        lo, hi, bm = range_packed_point_bernstein(packed, h, backend)
        for idx in indices:
            meta[idx] = {
                **bm,
                "lowered_execution": "packed-separable-bernstein",
                "term_count": packed.plan.term_count,
            }
    elif method == "portfolio":
        term_lo, term_hi = range_packed_point_termwise(packed, h, backend)
        mono_lo, mono_hi, mm = range_packed_point_monotonicity(packed, h, backend)
        bern_lo, bern_hi, reduced_meta = range_packed_point_reduced_bernstein(
            packed, h, backend, monotonicity_meta=mm
        )
        lo = np.maximum(np.maximum(term_lo, mono_lo), bern_lo)
        hi = np.minimum(np.minimum(term_hi, mono_hi), bern_hi)
        if np.any(lo > hi):
            raise RuntimeError(
                "Rigorous packed portfolio components are disjoint; treat as an internal error."
            )
        for idx in indices:
            br = Interval(float(bern_lo[idx]), float(bern_hi[idx]))
            meta[idx] = {
                "method": "portfolio",
                "termwise": Interval(float(term_lo[idx]), float(term_hi[idx])),
                "monotonicity": Interval(float(mono_lo[idx]), float(mono_hi[idx])),
                "bernstein": br,
                "monotonicity_meta": _packed_monotonicity_meta_at(mm, idx),
                "bernstein_meta": _packed_reduced_bernstein_meta_at(reduced_meta, idx),
                # Historical Phase-2 compatibility tag retained because existing
                # regression consumers inspect this field.  The additional field
                # below records the actual Phase-3 Bernstein implementation.
                "lowered_execution": "packed-cheap-scalar-bernstein",
                "bernstein_execution": "packed-separable-reduced-bernstein",
                "term_count": packed.plan.term_count,
            }
    else:
        raise ValueError(f"Unknown range method {method}")

    ranges = interval_objects_from_endpoints(lo, hi)
    if packed.family_shape == tuple():
        return ranges, meta.item()
    return ranges, meta

def interval_coefficient_representatives(packed: PackedIntervalPolynomialFamily):
    """Choose ordinary floating-point point representatives inside coefficients."""
    lo = packed.lower
    hi = packed.upper
    with np.errstate(over="ignore", invalid="ignore"):
        rep = np.add(0.5 * lo, 0.5 * hi, dtype=np.float64)
    if np.any(~np.isfinite(rep)):
        raise FloatingPointError("Nonfinite interval-coefficient representative.")
    rep = np.minimum(np.maximum(rep, lo), hi)
    exact = lo == hi
    rep[exact] = lo[exact]
    return rep


def range_interval_polynomial_family(
    coeff_arrays,
    h,
    backend,
    method="portfolio",
    *,
    plan: PolynomialRangePlan | None = None,
):
    """Range a same-shaped family of interval-coefficient polynomials.

    This is the batched Step-9d counterpart of the scalar semantic-solve helper.
    It computes the direct interval-coefficient termwise enclosure and the
    coefficient-error termwise enclosure for the *entire family* using endpoint
    buffers.  Representative point-polynomial ``termwise``, ``bernstein``, and
    ``portfolio`` work uses the packed family layer; the portfolio uses the
    dimension-separable monotonicity-reduced Bernstein kernel.  Direct
    ``monotonicity`` retains the established scalar point-range path so the
    historical public scalar arithmetic order remains available unchanged.
    """
    m = len(h)
    packed = pack_interval_polynomial_family(coeff_arrays, m, plan=plan)
    monomial_endpoints = monomial_range_endpoints(packed.plan, h, backend)

    direct_lo, direct_hi = range_packed_interval_termwise(
        packed, h, backend, monomial_endpoints=monomial_endpoints
    )

    rep = interval_coefficient_representatives(packed)
    err_lo, err_hi = rigorous_interval_subtract_point_arrays(
        packed.lower, packed.upper, rep, backend
    )
    err_packed = PackedIntervalPolynomialFamily(
        packed.plan, err_lo, err_hi, packed.family_shape
    )
    error_lo, error_hi = range_packed_interval_termwise(
        err_packed, h, backend, monomial_endpoints=monomial_endpoints
    )

    point_meta = np.empty(packed.family_shape, dtype=object)
    if method in {"termwise", "bernstein", "portfolio"}:
        point_coeff_arrays = {
            alpha: rep[..., t] for t, alpha in enumerate(packed.plan.support)
        }
        point_ranges, point_meta = range_point_polynomial_family(
            point_coeff_arrays, h, backend, method, plan=packed.plan
        )
        if packed.family_shape == tuple():
            point_lo = np.asarray(point_ranges.lo, dtype=np.float64)
            point_hi = np.asarray(point_ranges.hi, dtype=np.float64)
            point_meta = np.asarray(point_meta, dtype=object)
        else:
            point_lo = np.empty(packed.family_shape, dtype=np.float64)
            point_hi = np.empty(packed.family_shape, dtype=np.float64)
            for idx in np.ndindex(packed.family_shape):
                point_lo[idx] = point_ranges[idx].lo
                point_hi[idx] = point_ranges[idx].hi
    else:
        point_lo = np.empty(packed.family_shape, dtype=np.float64)
        point_hi = np.empty(packed.family_shape, dtype=np.float64)
        indices = list(np.ndindex(packed.family_shape)) if packed.family_shape else [tuple()]
        for idx in indices:
            point_poly = {}
            for t, alpha in enumerate(packed.plan.support):
                value = float(rep[idx + (t,)])
                if value != 0.0:
                    point_poly[alpha] = value
            pr, pmeta = range_polynomial(point_poly, h, backend, method)
            point_lo[idx] = pr.lo
            point_hi[idx] = pr.hi
            point_meta[idx] = pmeta

    # Only one interval addition remains per family member, so preserve the
    # mature scalar backend semantics at this object boundary.
    decomposed_lo = np.empty(packed.family_shape, dtype=np.float64)
    decomposed_hi = np.empty(packed.family_shape, dtype=np.float64)
    indices = list(np.ndindex(packed.family_shape)) if packed.family_shape else [tuple()]
    for idx in indices:
        value = backend.add(
            Interval(float(point_lo[idx]), float(point_hi[idx])),
            Interval(float(error_lo[idx]), float(error_hi[idx])),
        )
        decomposed_lo[idx] = value.lo
        decomposed_hi[idx] = value.hi

    final_lo = np.maximum(direct_lo, decomposed_lo)
    final_hi = np.minimum(direct_hi, decomposed_hi)
    if np.any(final_lo > final_hi):
        raise RuntimeError(
            "Disjoint sound interval-coefficient polynomial enclosures; "
            "treat as an internal error."
        )

    ranges = interval_objects_from_endpoints(final_lo, final_hi)
    direct = interval_objects_from_endpoints(direct_lo, direct_hi)
    decomposed = interval_objects_from_endpoints(decomposed_lo, decomposed_hi)
    meta = np.empty(packed.family_shape, dtype=object)
    for idx in indices:
        meta[idx] = {
            "direct_termwise": direct if packed.family_shape == tuple() else direct[idx],
            "point_plus_coefficient_error": (
                decomposed if packed.family_shape == tuple() else decomposed[idx]
            ),
            "point_range_meta": point_meta.item() if packed.family_shape == tuple() else point_meta[idx],
            "lowered_execution": "packed-family",
            "term_count": packed.plan.term_count,
        }

    if packed.family_shape == tuple():
        return ranges, meta.item()
    return ranges, meta


__all__ = [
    "PolynomialRangePlan",
    "PackedPointPolynomialFamily",
    "PackedIntervalPolynomialFamily",
    "compile_polynomial_range_plan",
    "pack_point_polynomial_family",
    "pack_interval_polynomial_family",
    "monomial_range_endpoints",
    "interval_objects_from_endpoints",
    "range_packed_point_termwise",
    "range_packed_interval_termwise",
    "derivative_factor_endpoints",
    "range_packed_point_derivatives",
    "monotonicity_signs_from_derivative_endpoints",
    "range_packed_point_monotonicity",
    "range_point_polynomial_family",
    "interval_coefficient_representatives",
    "range_interval_polynomial_family",
]
