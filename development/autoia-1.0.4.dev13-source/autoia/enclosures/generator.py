"""Shared-generator polynomial enclosure ``P(q) + g·eps + R``.

The physical uncertain inputs remain explicit in the point-coefficient
polynomial ``P(q)``.  Auxiliary generator variables ``eps_j in [-1, 1]`` carry
nonphysical approximation/certification error with stable provenance.  Two
values that descend from the same generator share the same generator id, so
linear cancellation is preserved automatically.

The representation is intentionally affine in the auxiliary generators.  When
an operation produces terms outside that family (for example ``eps_i*eps_j``
or ``q*eps_j``), those terms are rigorously ranged and returned as a new
unstructured generator.  This is sound and gives a monotone fallback rather
than inventing unsupported correlation.
"""

from __future__ import annotations

import itertools
import numpy as np

from ..trusted import Interval, point_interval, interval_neg, RigorousBinary64Backend
from ..polynomial import (
    zero_alpha,
    basis_alpha,
    total_degree,
    clean_poly,
    rigorous_poly_add_point,
    rigorous_poly_mul_truncated,
    range_polynomial,
)
from .runtime import safe_interval_representative


_GENERATOR_COUNTER = itertools.count(1)


def _fresh_generator_id() -> int:
    # Generator ids are labels, not proof state.  Process-wide uniqueness makes
    # accidental correlation between unrelated values impossible; correctness
    # does not depend on the numerical value of an id.
    return int(next(_GENERATOR_COUNTER))


def _zero_interval() -> Interval:
    return Interval(0.0, 0.0)


def _interval_abs_radius(I: Interval) -> float:
    return float(max(abs(I.lo), abs(I.hi)))


def _generator_range(generators, backend) -> Interval:
    if not generators:
        return _zero_interval()
    total = point_interval(0.0)
    for value in generators.values():
        total = backend.add(total, point_interval(abs(float(value))))
    radius = float(total.hi)
    return Interval(-radius, radius)


def _clean_generators(generators):
    return {
        int(k): float(v)
        for k, v in dict(generators or {}).items()
        if float(v) != 0.0
    }


def _generator_add(g1, g2, backend):
    """Add two generator maps with rigorous coefficient-rounding capture."""
    out = {}
    error = _zero_interval()
    for key in set(g1) | set(g2):
        a = float(g1.get(key, 0.0))
        b = float(g2.get(key, 0.0))
        if a == -b:
            # Exact-real identity for stored binary64 coefficients.
            continue
        exact = backend.add(point_interval(a), point_interval(b))
        proposal = float(a + b)
        if not np.isfinite(proposal):
            raise FloatingPointError("Overflow in shared-generator coefficient addition.")
        coeff_error = backend.sub(exact, point_interval(proposal))
        radius = _interval_abs_radius(coeff_error)
        if radius != 0.0:
            error = backend.add(error, Interval(-radius, radius))
        if proposal != 0.0:
            out[key] = proposal
    return out, error


def _scaled_generator_sum(g1, s1, g2, s2, backend):
    """Return ``s1*g1 + s2*g2`` plus rigorous coefficient-rounding error."""
    out = {}
    error = _zero_interval()
    for key in set(g1) | set(g2):
        a = float(g1.get(key, 0.0))
        b = float(g2.get(key, 0.0))
        t1 = backend.mul(point_interval(float(s1)), point_interval(a))
        t2 = backend.mul(point_interval(float(s2)), point_interval(b))
        exact = backend.add(t1, t2)
        proposal = float(float(s1) * a + float(s2) * b)
        if not np.isfinite(proposal):
            raise FloatingPointError("Overflow in shared-generator coefficient scaling.")
        coeff_error = backend.sub(exact, point_interval(proposal))
        radius = _interval_abs_radius(coeff_error)
        if radius != 0.0:
            error = backend.add(error, Interval(-radius, radius))
        if proposal != 0.0:
            out[key] = proposal
    return out, error


class SharedGeneratorPolynomialEnclosure:
    """Scalar enclosure ``z(q) in P(q) + sum_j g_j eps_j + R``.

    ``eps_j`` are auxiliary variables in ``[-1,1]``.  Generator ids encode
    provenance: equal ids refer to the same auxiliary variable and distinct ids
    are treated independently.  ``R`` is a final unstructured interval fallback;
    for finite errors the constructor normally canonicalizes it into one fresh
    generator so copies can preserve its provenance.
    """

    __array_priority__ = 1100
    representation_name = "generator-polynomial"
    promotion_priority = 150
    supports_value_arithmetic = True

    def __init__(self, coeff, remainder, h, order, backend=None,
                 range_method="portfolio", *, generators=None,
                 lift_remainder=True):
        self.coeff = clean_poly(coeff)
        self.generators = _clean_generators(generators)
        self.remainder = remainder
        self.h = np.asarray(h, dtype=float)
        self.m = len(self.h)
        self.order = int(order)
        self.backend = backend if backend is not None else RigorousBinary64Backend()
        self.range_method = range_method
        if np.any(self.h < 0.0) or np.any(~np.isfinite(self.h)):
            raise ValueError("Shared-generator representation requires finite nonnegative half-widths.")
        if not isinstance(self.remainder, Interval):
            raise TypeError("Shared-generator remainder must be an Interval.")
        if lift_remainder and not (self.remainder.lo == self.remainder.hi == 0.0):
            self._lift_finite_remainder_in_place()

    @staticmethod
    def constant(value, h, order, backend=None, range_method="portfolio"):
        m = len(h)
        return SharedGeneratorPolynomialEnclosure(
            {zero_alpha(m): float(value)}, _zero_interval(), h, order,
            backend=backend, range_method=range_method,
        )

    @staticmethod
    def parameter(center, index, h, order, backend=None,
                  range_method="portfolio"):
        m = len(h)
        return SharedGeneratorPolynomialEnclosure(
            {zero_alpha(m): float(center), basis_alpha(m, index): 1.0},
            _zero_interval(), h, order, backend=backend,
            range_method=range_method,
        )

    def _lift_finite_remainder_in_place(self):
        """Convert finite ``R`` to a polynomial shift plus one fresh generator."""
        R = self.remainder
        if not np.isfinite(R.lo) or not np.isfinite(R.hi):
            # Infinite residuals cannot be represented by a finite generator.
            return
        center = safe_interval_representative(R)
        shifted = self.backend.sub(R, point_interval(center))
        shifted_coeff, coeff_error = rigorous_poly_add_point(
            self.coeff,
            {zero_alpha(self.m): center},
            self.h,
            self.backend,
        )
        total_error = self.backend.add(shifted, coeff_error)
        radius = _interval_abs_radius(total_error)
        self.coeff = shifted_coeff
        if radius != 0.0:
            self.generators[_fresh_generator_id()] = radius
        self.remainder = _zero_interval()

    def copy(self):
        return SharedGeneratorPolynomialEnclosure(
            dict(self.coeff), self.remainder, self.h.copy(), self.order,
            self.backend, self.range_method, generators=dict(self.generators),
            lift_remainder=False,
        )

    @property
    def generator_ids(self):
        return tuple(sorted(self.generators))

    def generator_coefficient(self, generator_id):
        return float(self.generators.get(int(generator_id), 0.0))

    def generator_range(self):
        return _generator_range(self.generators, self.backend)

    def _coerce(self, other):
        if isinstance(other, SharedGeneratorPolynomialEnclosure):
            if self.m != other.m or not np.all(self.h == other.h):
                raise ValueError("Shared-generator enclosure domains do not match.")
            if self.backend.name != other.backend.name:
                raise ValueError("Shared-generator enclosure backends do not match.")
            return other
        return self.convert_from(other) if hasattr(other, "representation_name") else self.constant_like(other)

    def constant_like(self, value):
        return SharedGeneratorPolynomialEnclosure.constant(
            value, self.h, self.order, self.backend, self.range_method
        )

    def negated(self):
        return -self

    def unary(self, op, *, domain=None):
        from .elementary import apply_polynomial_elementary
        return apply_polynomial_elementary(self, op, domain=domain)

    def power_constant(self, power, *, domain=None):
        from .power import normalize_constant_exponent, integer_power_rep, apply_polynomial_real_power
        kind, p = normalize_constant_exponent(power)
        if kind == "integer":
            return integer_power_rep(self, p)
        return apply_polynomial_real_power(self, p, domain=domain)

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
        name = getattr(other, "representation_name", None)
        if name == self.representation_name:
            return other.copy()
        if name == "polynomial":
            return SharedGeneratorPolynomialEnclosure(
                dict(other.coeff), other.remainder, other.h, other.order,
                other.backend, other.range_method,
            )
        if name == "interval":
            I = other.interval
            c = safe_interval_representative(I)
            rem = other.backend.sub(I, point_interval(c))
            return SharedGeneratorPolynomialEnclosure(
                {zero_alpha(other.m): c}, rem, other.h, other.order,
                other.backend, other.range_method,
            )
        raise TypeError(
            f"SharedGeneratorPolynomialEnclosure cannot convert representation {name!r}."
        )

    def collapse_to_polynomial(self):
        """Soundly forget generator correlation and return ``P(q)+R``."""
        from .polynomial import PolynomialEnclosure
        rem = self.backend.add(self.remainder, self.generator_range())
        return PolynomialEnclosure(
            dict(self.coeff), rem, self.h, self.order,
            self.backend, self.range_method,
        )

    def polynomial_range(self, method=None):
        return range_polynomial(
            self.coeff, self.h, self.backend,
            method=self.range_method if method is None else method,
        )

    def range(self, method=None):
        pr, meta = self.polynomial_range(method)
        out = self.backend.add(pr, self.generator_range())
        out = self.backend.add(out, self.remainder)
        meta = dict(meta)
        meta["generator_count"] = len(self.generators)
        return out, meta

    def is_affine_structured(self):
        return all(total_degree(alpha) <= 1 for alpha in self.coeff)

    def with_remainder(self, remainder):
        """Return the same structured value plus a new unstructured remainder."""
        return SharedGeneratorPolynomialEnclosure(
            dict(self.coeff), remainder, self.h, self.order,
            self.backend, self.range_method, generators=dict(self.generators),
            lift_remainder=True,
        )

    def __add__(self, other):
        other = self._coerce(other)
        # Exact additive-inverse identity.  This is the key fan-out/cancellation
        # case for shared provenance and avoids introducing an artificial ulp
        # merely because a trusted generic addition routine expands 0 outward.
        if (
            self.remainder.lo == self.remainder.hi == 0.0
            and other.remainder.lo == other.remainder.hi == 0.0
            and self.generators.keys() == other.generators.keys()
            and all(other.generators[k] == -v for k, v in self.generators.items())
            and self.coeff.keys() == other.coeff.keys()
            and all(other.coeff[k] == -v for k, v in self.coeff.items())
        ):
            return self.constant_like(0.0)
        coeff, coeff_error = rigorous_poly_add_point(
            self.coeff, other.coeff, self.h, self.backend
        )
        generators, gen_error = _generator_add(
            self.generators, other.generators, self.backend
        )
        rem = self.backend.add(self.remainder, other.remainder)
        rem = self.backend.add(rem, coeff_error)
        rem = self.backend.add(rem, gen_error)
        return SharedGeneratorPolynomialEnclosure(
            coeff, rem, self.h, min(self.order, other.order), self.backend,
            self.range_method, generators=generators,
        )

    __radd__ = __add__

    def __neg__(self):
        return SharedGeneratorPolynomialEnclosure(
            {alpha: -c for alpha, c in self.coeff.items()},
            interval_neg(self.remainder), self.h, self.order,
            self.backend, self.range_method,
            generators={key: -value for key, value in self.generators.items()},
            lift_remainder=False,
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

        p0 = float(self.coeff.get(zero_alpha(self.m), 0.0))
        q0 = float(other.coeff.get(zero_alpha(self.m), 0.0))
        generators, gen_coeff_error = _scaled_generator_sum(
            self.generators, q0, other.generators, p0, self.backend
        )

        P_range, _ = range_polynomial(self.coeff, self.h, self.backend, self.range_method)
        Q_range, _ = range_polynomial(other.coeff, self.h, self.backend, self.range_method)
        dP_coeff = dict(self.coeff)
        dQ_coeff = dict(other.coeff)
        dP_coeff.pop(zero_alpha(self.m), None)
        dQ_coeff.pop(zero_alpha(self.m), None)
        dP_range, _ = range_polynomial(dP_coeff, self.h, self.backend, self.range_method)
        dQ_range, _ = range_polynomial(dQ_coeff, self.h, self.backend, self.range_method)
        G = self.generator_range()
        H = other.generator_range()

        rem = poly_extra
        rem = self.backend.add(rem, gen_coeff_error)
        # Terms not representable by constant-coefficient generators.
        rem = self.backend.add(rem, self.backend.mul(dP_range, H))
        rem = self.backend.add(rem, self.backend.mul(dQ_range, G))
        rem = self.backend.add(rem, self.backend.mul(G, H))
        rem = self.backend.add(rem, self.backend.mul(P_range, other.remainder))
        rem = self.backend.add(rem, self.backend.mul(Q_range, self.remainder))
        rem = self.backend.add(rem, self.backend.mul(G, other.remainder))
        rem = self.backend.add(rem, self.backend.mul(H, self.remainder))
        rem = self.backend.add(rem, self.backend.mul(self.remainder, other.remainder))

        return SharedGeneratorPolynomialEnclosure(
            kept, rem, self.h, order, self.backend, self.range_method,
            generators=generators,
        )

    __rmul__ = __mul__

    def reciprocal(self):
        # The current affine-generator family is not closed under reciprocal.
        # Collapse the generator contribution soundly into a polynomial remainder,
        # use the audited polynomial reciprocal, then re-lift the certified output.
        result = self.collapse_to_polynomial().reciprocal()
        return self.convert_from(result)

    def __truediv__(self, other):
        other = self._coerce(other)
        return self * other.reciprocal()

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def __pow__(self, power):
        return self.power_constant(power)

    def __repr__(self):
        r, _ = self.range()
        return (
            f"SharedGeneratorPolynomialEnclosure(range=[{r.lo:.12g}, {r.hi:.12g}], "
            f"terms={len(self.coeff)}, generators={len(self.generators)}, "
            f"remainder=[{self.remainder.lo:.3e},{self.remainder.hi:.3e}])"
        )



def extract_generator_polynomial_array(arr, prototype, name):
    """Extract ``P(q) + sum_g G_g eps_g + R`` array data.

    Parameters
    ----------
    arr : array-like
        Scalars may be shared-generator representations, ``ValidatedValue``
        wrappers, other sound AutoIA scalar representations convertible to the
        generator family, or ordinary real constants.
    prototype : SharedGeneratorPolynomialEnclosure
        Supplies the common domain, order, backend, and range policy.
    name : str
        Diagnostic label used in conversion errors.

    Returns
    -------
    coeff_arrays : dict[multi_index, ndarray(float)]
        Retained physical-input polynomial coefficients.
    generator_arrays : dict[int, ndarray(float)]
        One constant coefficient array for each auxiliary generator id.
    rem : ndarray(object)
        Residual interval remainders after the explicit polynomial and
        generator parts.

    Notes
    -----
    Equal generator ids retain their provenance across every matrix/vector
    entry.  Distinct ids remain independent.  This function does not merge
    generators merely because their numerical coefficients happen to match.
    """
    obj = np.asarray(arr, dtype=object)
    converted = np.empty(obj.shape, dtype=object)
    rem = np.empty(obj.shape, dtype=object)
    coeff_arrays = {}
    generator_arrays = {}

    for idx in np.ndindex(obj.shape):
        v = obj[idx]
        rep = getattr(v, "rep", v)
        if not isinstance(rep, SharedGeneratorPolynomialEnclosure):
            try:
                if hasattr(rep, "representation_name"):
                    rep = prototype.convert_from(rep)
                else:
                    rep = prototype.constant_like(float(rep))
            except Exception as exc:
                raise TypeError(f"{name}{idx}: cannot convert to generator representation") from exc
        converted[idx] = rep
        rem[idx] = rep.remainder
        for alpha in rep.coeff:
            if alpha not in coeff_arrays:
                coeff_arrays[alpha] = np.zeros(obj.shape, dtype=float)
        for gid in rep.generators:
            if gid not in generator_arrays:
                generator_arrays[gid] = np.zeros(obj.shape, dtype=float)

    z = zero_alpha(prototype.m)
    if z not in coeff_arrays:
        coeff_arrays[z] = np.zeros(obj.shape, dtype=float)

    for idx in np.ndindex(obj.shape):
        rep = converted[idx]
        for alpha, value in rep.coeff.items():
            coeff_arrays[alpha][idx] = float(value)
        for gid, value in rep.generators.items():
            generator_arrays[int(gid)][idx] = float(value)

    return coeff_arrays, generator_arrays, rem

def generator_matrix_from_values(values):
    """Return ``(generator_ids, G)`` for a flat sequence of generator values.

    Rows correspond to output values and columns to shared generator ids.  Values
    of other representation families contribute zero rows.
    """
    vals = list(values)
    ids = sorted({
        gid
        for value in vals
        for gid in getattr(getattr(value, "rep", value), "generators", {})
    })
    G = np.zeros((len(vals), len(ids)), dtype=float)
    col = {gid: j for j, gid in enumerate(ids)}
    for i, value in enumerate(vals):
        rep = getattr(value, "rep", value)
        for gid, coeff in getattr(rep, "generators", {}).items():
            G[i, col[gid]] = float(coeff)
    return tuple(ids), G


__all__ = [
    "SharedGeneratorPolynomialEnclosure",
    "generator_matrix_from_values",
    "extract_generator_polynomial_array",
]
