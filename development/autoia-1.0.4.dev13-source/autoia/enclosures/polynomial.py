"""Polynomial-plus-remainder scalar enclosure.

The numerical algorithms in this module are a behavior-preserving extraction
of the Version-0.11 ``ParameterizedEnclosure`` implementation.  Point
polynomial coefficients remain proposal/data values; all proof-critical
coefficient error and range calculations are delegated to ``autoia.polynomial``
and ``autoia.trusted``.
"""

from __future__ import annotations

import numpy as np

from ..trusted import Interval, point_interval, interval_neg, RigorousBinary64Backend
from ..polynomial import (
    zero_alpha,
    basis_alpha,
    total_degree,
    clean_poly,
    rigorous_poly_add_point,
    rigorous_poly_mul_truncated,
    build_reciprocal_predictor_point,
    range_polynomial,
    _degree_vector_and_bernstein_cost,
)
from .runtime import (
    active_range_policy,
    record_range_policy_event,
    interval_intersection,
    safe_interval_representative,
)


class ParameterizedEnclosure:
    """Scalar enclosure ``z(q) in P(q) + R``.

    ``P`` is a stored point-coefficient polynomial in the original centered
    uncertain inputs and ``R`` is a rigorous scalar interval remainder.
    """

    __array_priority__ = 1000
    representation_name = "polynomial"
    promotion_priority = 100
    supports_value_arithmetic = True

    def __init__(self, coeff, remainder, h, order,
                 backend=None, range_method="portfolio"):
        self.coeff = clean_poly(coeff)
        self.remainder = remainder
        self.h = np.asarray(h, dtype=float)
        self.m = len(self.h)
        self.order = int(order)
        self.backend = backend if backend is not None else RigorousBinary64Backend()
        self.range_method = range_method

        if np.any(self.h < 0.0) or np.any(~np.isfinite(self.h)):
            raise ValueError(
                "Version 0.6 requires finite nonnegative parameter half-widths."
            )

    @staticmethod
    def constant(value, h, order, backend=None, range_method="portfolio"):
        m = len(h)
        return ParameterizedEnclosure(
            {zero_alpha(m): float(value)}, Interval(0.0, 0.0), h, order,
            backend=backend, range_method=range_method,
        )

    @staticmethod
    def parameter(center, index, h, order, backend=None,
                  range_method="portfolio"):
        m = len(h)
        return ParameterizedEnclosure(
            {zero_alpha(m): float(center), basis_alpha(m, index): 1.0},
            Interval(0.0, 0.0), h, order,
            backend=backend, range_method=range_method,
        )

    def copy(self):
        return ParameterizedEnclosure(
            dict(self.coeff), self.remainder, self.h.copy(), self.order,
            self.backend, self.range_method,
        )

    def _coerce(self, other):
        if isinstance(other, ParameterizedEnclosure):
            if self.m != other.m or not np.all(self.h == other.h):
                raise ValueError("ParameterizedEnclosure domains do not match.")
            if self.backend.name != other.backend.name:
                raise ValueError("ParameterizedEnclosure backends do not match.")
            return other
        return ParameterizedEnclosure.constant(
            other, self.h, self.order, self.backend, self.range_method,
        )

    # ------------------------------------------------------------------
    # Representation contract used by ValidatedValue
    # ------------------------------------------------------------------
    def constant_like(self, value):
        return ParameterizedEnclosure.constant(
            value, self.h, self.order, self.backend, self.range_method
        )

    def negated(self):
        return -self

    def unary(self, op, *, domain=None):
        from .elementary import apply_polynomial_elementary
        return apply_polynomial_elementary(self, op, domain=domain)

    def binary_same(self, other, op):
        other = self._coerce(other)
        if op == "add":
            return self + other
        if op == "mul":
            return self * other
        if op == "div":
            return self / other
        raise ValueError(op)

    def convert_from(self, other):
        """Soundly convert another scalar representation to polynomial form."""
        name = getattr(other, "representation_name", None)
        if name == "polynomial":
            return other.copy()
        if name == "generator-polynomial" and hasattr(other, "collapse_to_polynomial"):
            return other.collapse_to_polynomial()
        if name != "interval" or not hasattr(other, "interval"):
            raise TypeError(
                f"PolynomialEnclosure cannot convert representation {name!r}."
            )

        # Sound but dependency-destroying conversion X = c + (X-c).
        I = other.interval
        c = safe_interval_representative(I)
        rem = other.backend.sub(I, point_interval(c))
        return ParameterizedEnclosure(
            {zero_alpha(other.m): c}, rem,
            other.h, other.order, other.backend, other.range_method,
        )

    def polynomial_range(self, method=None):
        return range_polynomial(
            self.coeff,
            self.h,
            self.backend,
            method=self.range_method if method is None else method,
        )

    def range(self, method=None):
        pr, meta = self.polynomial_range(method)
        return self.backend.add(pr, self.remainder), meta

    def is_affine_structured(self):
        return all(total_degree(alpha) <= 1 for alpha in self.coeff)

    def affine_data(self):
        if not self.is_affine_structured():
            raise NotImplementedError(
                "Current solve primitive requires an affine retained polynomial part. "
                "A scalar interval remainder is allowed and will be certified."
            )
        c0 = self.coeff.get(zero_alpha(self.m), 0.0)
        grad = np.zeros(self.m)
        for i in range(self.m):
            grad[i] = self.coeff.get(basis_alpha(self.m, i), 0.0)
        return c0, grad, self.remainder

    def __add__(self, other):
        other = self._coerce(other)
        coeff, coeff_error = rigorous_poly_add_point(
            self.coeff, other.coeff, self.h, self.backend
        )
        r = self.backend.add(self.remainder, other.remainder)
        r = self.backend.add(r, coeff_error)
        return ParameterizedEnclosure(
            coeff, r, self.h, min(self.order, other.order),
            self.backend, self.range_method,
        )

    __radd__ = __add__

    def __neg__(self):
        return ParameterizedEnclosure(
            {alpha: -c for alpha, c in self.coeff.items()},
            interval_neg(self.remainder),
            self.h, self.order, self.backend, self.range_method,
        )

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        order = min(self.order, other.order)

        kept, poly_extra = rigorous_poly_mul_truncated(
            self.coeff, other.coeff, order, self.h, self.backend
        )

        px_range, _ = range_polynomial(
            self.coeff, self.h, self.backend, self.range_method
        )
        py_range, _ = range_polynomial(
            other.coeff, self.h, self.backend, self.range_method
        )

        r = poly_extra
        r = self.backend.add(r, self.backend.mul(px_range, other.remainder))
        r = self.backend.add(r, self.backend.mul(py_range, self.remainder))
        r = self.backend.add(r, self.backend.mul(self.remainder, other.remainder))

        return ParameterizedEnclosure(
            kept, r, self.h, order, self.backend, self.range_method,
        )

    __rmul__ = __mul__

    def reciprocal(self):
        """Rigorous reciprocal from a point-polynomial proposal and residual."""
        policy = active_range_policy()
        proof_ranges = []

        Y, _ = self.range(self.range_method)
        proof_ranges.append((self.range_method, Y))

        if Y.lo <= 0.0 <= Y.hi and self.range_method != "monotonicity":
            Ym, _ = self.range("monotonicity")
            Y = interval_intersection(Y, Ym)
            proof_ranges.append(("monotonicity", Ym))

        degrees, bern_cost = _degree_vector_and_bernstein_cost(self.coeff, self.m)
        if (
            Y.lo <= 0.0 <= Y.hi
            and bern_cost <= policy.proof_bernstein_max_coefficients
        ):
            Yb, _ = self.range("bernstein")
            Y = interval_intersection(Y, Yb)
            proof_ranges.append(("bernstein", Yb))

        record_range_policy_event({
            "kind": "reciprocal-nonzero-proof",
            "methods_tried": tuple(name for name, _ in proof_ranges),
            "final_range": Y,
            "estimated_bernstein_coefficients": bern_cost,
            "success": not (Y.lo <= 0.0 <= Y.hi),
        })

        if Y.lo <= 0.0 <= Y.hi:
            from .domain import DomainRequirement, require_precomputed_domain
            require_precomputed_domain(
                self,
                DomainRequirement(
                    "reciprocal", "nonzero", 0.0,
                    "denominator separated from zero",
                ),
                Y,
                methods_tried=tuple(name for name, _ in proof_ranges),
                estimated_bernstein_coefficients=bern_cost,
            )

        center = safe_interval_representative(Y)
        Qcoeff = build_reciprocal_predictor_point(
            self.coeff, self.m, self.order, fallback_center=center
        )
        Q = ParameterizedEnclosure(
            Qcoeff, Interval(0.0, 0.0), self.h, self.order,
            self.backend, self.range_method,
        )

        one = ParameterizedEnclosure.constant(
            1.0, self.h, self.order, self.backend, self.range_method
        )
        residual = one - self * Q
        H, _ = residual.range(self.range_method)

        E_residual = self.backend.div(H, Y)

        invY = self.backend.div(Interval(1.0, 1.0), Y)
        Qrange, _ = Q.range(self.range_method)
        E_range = self.backend.sub(invY, Qrange)
        Elo = max(E_residual.lo, E_range.lo)
        Ehi = min(E_residual.hi, E_range.hi)
        if Elo > Ehi:
            raise RuntimeError(
                "Independent reciprocal certificates are disjoint; treat as "
                "an internal soundness error."
            )
        E = Interval(Elo, Ehi)

        return ParameterizedEnclosure(
            Qcoeff, E, self.h, self.order, self.backend, self.range_method
        )

    def __truediv__(self, other):
        other = self._coerce(other)
        return self * other.reciprocal()

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def power_constant(self, power, *, domain=None):
        from .power import power_representation
        return power_representation(self, power, domain=domain)

    def __pow__(self, power):
        return self.power_constant(power)

    def __repr__(self):
        r, _ = self.range()
        return (
            f"ParameterizedEnclosure(range=[{r.lo:.12g}, {r.hi:.12g}], "
            f"terms={len(self.coeff)}, remainder=[{self.remainder.lo:.3e},"
            f"{self.remainder.hi:.3e}])"
        )


# Backward-compatible internal name and public representation name refer to the
# exact same class object during the migration.
PolynomialEnclosure = ParameterizedEnclosure


def contains_pe(arr):
    obj = np.asarray(arr, dtype=object)
    return any(isinstance(v, ParameterizedEnclosure) for v in obj.flat)



def extract_polynomial_array(arr, prototype: ParameterizedEnclosure, name):
    """Extract retained point-polynomial coefficient arrays plus remainders.

    Returns
    -------
    coeff_arrays : dict[tuple[int, ...], numpy.ndarray]
        For every retained multi-index ``alpha`` appearing in any array entry,
        ``coeff_arrays[alpha]`` has the same shape as ``arr`` and contains the
        stored binary64 point coefficient of ``q**alpha`` for each entry.
    rem : numpy.ndarray(dtype=object)
        Same-shape array of rigorous scalar interval remainders.

    Notes
    -----
    Missing coefficients are exactly zero.  Stored polynomial coefficients are
    interpreted as exact real values of their binary64 data, consistent with
    the AutoIA polynomial contract.
    """
    obj = np.asarray(arr, dtype=object)
    rem = np.empty(obj.shape, dtype=object)
    coeff_arrays = {}

    converted = np.empty(obj.shape, dtype=object)
    for idx in np.ndindex(obj.shape):
        v = obj[idx]
        if not isinstance(v, ParameterizedEnclosure):
            v = prototype._coerce(v)
        converted[idx] = v
        rem[idx] = v.remainder
        for alpha in v.coeff:
            if alpha not in coeff_arrays:
                coeff_arrays[alpha] = np.zeros(obj.shape, dtype=float)

    # Always expose the constant coefficient array, even when it is identically
    # zero.  The semantic solve needs A_0 and b_0 as proposal data.
    z = zero_alpha(prototype.m)
    if z not in coeff_arrays:
        coeff_arrays[z] = np.zeros(obj.shape, dtype=float)

    for idx in np.ndindex(obj.shape):
        v = converted[idx]
        for alpha, value in v.coeff.items():
            coeff_arrays[alpha][idx] = float(value)

    return coeff_arrays, rem

def extract_affine_array(arr, prototype: ParameterizedEnclosure, name):
    """Extract point affine coefficients plus scalar interval remainders."""
    obj = np.asarray(arr, dtype=object)
    values = np.empty(obj.shape)
    deriv = np.empty((prototype.m,) + obj.shape)
    rem = np.empty(obj.shape, dtype=object)

    for idx in np.ndindex(obj.shape):
        v = obj[idx]
        if not isinstance(v, ParameterizedEnclosure):
            v = prototype._coerce(v)
        try:
            c0, g, r = v.affine_data()
        except NotImplementedError as exc:
            raise NotImplementedError(f"{name}{idx}: {exc}") from exc
        values[idx] = c0
        deriv[(slice(None),) + idx] = g
        rem[idx] = r

    return values, deriv, rem


__all__ = [
    "ParameterizedEnclosure",
    "PolynomialEnclosure",
    "contains_pe",
    "extract_affine_array",
    "extract_polynomial_array",
]
