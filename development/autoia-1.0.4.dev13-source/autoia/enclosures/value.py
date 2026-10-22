"""Representation-independent scalar ``ValidatedValue`` wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import numpy as np

from ..trusted import Interval, point_interval, interval_neg, real_power_interval
from .runtime import new_approximation_source_mask, record_approximation_recombination


@dataclass(frozen=True)
class ExactAffineShadow:
    """Proof-only exact affine function of the physical input coordinates.

    The shadow is metadata, not a numerical enclosure.  It is retained only
    while ordinary model arithmetic remains exactly affine in the original
    physical inputs, with user constants interpreted as their exact binary64
    real values.  Its purpose is to make scalar comparison semantics immune to
    harmless outward-rounding noise at closed-box boundaries.
    """

    constant: Fraction
    coefficients: tuple[Fraction, ...]

    @classmethod
    def parameter(cls, dimension: int, index: int):
        coeff = [Fraction(0, 1)] * int(dimension)
        coeff[int(index)] = Fraction(1, 1)
        return cls(Fraction(0, 1), tuple(coeff))

    @classmethod
    def constant_value(cls, dimension: int, value):
        return cls(Fraction.from_float(float(value)),
                   tuple(Fraction(0, 1) for _ in range(int(dimension))))

    @property
    def is_constant(self):
        return all(a == 0 for a in self.coefficients)

    def __neg__(self):
        return ExactAffineShadow(-self.constant, tuple(-a for a in self.coefficients))

    def __add__(self, other):
        if len(self.coefficients) != len(other.coefficients):
            return None
        return ExactAffineShadow(
            self.constant + other.constant,
            tuple(a + b for a, b in zip(self.coefficients, other.coefficients)),
        )

    def scale(self, factor: Fraction):
        return ExactAffineShadow(
            self.constant * factor,
            tuple(a * factor for a in self.coefficients),
        )

    def range_on_box(self, box):
        lo = self.constant
        hi = self.constant
        for a, I in zip(self.coefficients, box):
            if a == 0:
                continue
            ilo = Fraction.from_float(float(I.lo))
            ihi = Fraction.from_float(float(I.hi))
            if a > 0:
                lo += a * ilo
                hi += a * ihi
            else:
                lo += a * ihi
                hi += a * ilo
        return lo, hi

    def decision_on_box(self, operator: str, box):
        lo, hi = self.range_on_box(box)
        z = Fraction(0, 1)
        if operator == "lt":
            if hi < z: return True
            if lo >= z: return False
        elif operator == "le":
            if hi <= z: return True
            if lo > z: return False
        elif operator == "gt":
            if lo > z: return True
            if hi <= z: return False
        elif operator == "ge":
            if lo >= z: return True
            if hi < z: return False
        elif operator == "eq":
            if lo == z and hi == z: return True
            if hi < z or lo > z: return False
        elif operator == "ne":
            if hi < z or lo > z: return True
            if lo == z and hi == z: return False
        return None

    def exactly_representable_single_root(self, box):
        active = [(i, a) for i, a in enumerate(self.coefficients) if a != 0]
        if len(active) != 1:
            return None
        i, a = active[0]
        root = -self.constant / a
        blo = Fraction.from_float(float(box[i].lo))
        bhi = Fraction.from_float(float(box[i].hi))
        if root < blo or root > bhi:
            return None
        rf = float(root)
        if not np.isfinite(rf) or Fraction.from_float(rf) != root:
            return None
        return int(i), rf


def same_domain(a, b):
    return (
        len(a.h) == len(b.h)
        and np.all(a.h == b.h)
        and a.backend.name == b.backend.name
    )



def _has_unstructured_remainder(rep) -> bool:
    rem = getattr(rep, "remainder", None)
    return bool(
        rem is not None
        and (float(getattr(rem, "lo", 0.0)) != 0.0 or float(getattr(rem, "hi", 0.0)) != 0.0)
    )

def _is_flow_representation(rep) -> bool:
    required = (
        "representation_name",
        "promotion_priority",
        "supports_value_arithmetic",
        "h",
        "order",
        "backend",
        "range_method",
        "copy",
        "range",
        "constant_like",
        "negated",
        "binary_same",
        "convert_from",
    )
    return all(hasattr(rep, name) for name in required)


class ValidatedValue:
    """Representation-independent scalar AutoIA value.

    Concrete representations implement a small structural contract.  Arithmetic
    is dispatched through that contract rather than through concrete
    ``isinstance`` checks.  For mixed operations, the representation with the
    larger ``promotion_priority`` becomes the target and soundly converts both
    operands before applying the same-representation operation.
    """

    __array_priority__ = 2000

    def __init__(self, representation, natural_interval=None, *, _exact_affine_shadow=None, _diag_approx_mask=0, _diag_add_source=True, _diag_recombined=False):
        if not _is_flow_representation(representation):
            raise TypeError(
                "ValidatedValue requires a scalar representation implementing "
                "the AutoIA flow-representation contract."
            )
        if not representation.supports_value_arithmetic:
            raise TypeError(
                f"Representation {representation.representation_name!r} is "
                "not enabled for arithmetic inside engineering-model execution."
            )
        self.rep = representation
        if natural_interval is not None and not isinstance(natural_interval, Interval):
            raise TypeError("natural_interval must be an AutoIA trusted Interval or None.")
        # Independent natural interval extension over the actual physical input
        # box.  This is proof metadata only: it is neither a physical q variable
        # nor an auxiliary epsilon generator.
        self.natural_interval = natural_interval
        if _exact_affine_shadow is not None and not isinstance(_exact_affine_shadow, ExactAffineShadow):
            raise TypeError("_exact_affine_shadow must be ExactAffineShadow or None.")
        self.exact_affine_shadow = _exact_affine_shadow
        # Non-proof configuration diagnostics.  This compact run-local bit mask
        # tracks ancestry of unstructured approximation/remainder sources.  It
        # is never consulted by any trusted enclosure operation.
        mask = int(_diag_approx_mask)
        if _diag_add_source and _has_unstructured_remainder(representation):
            mask |= new_approximation_source_mask()
        self._diag_approx_mask = mask
        self._diag_recombined = bool(_diag_recombined)

    @property
    def representation_name(self):
        return self.rep.representation_name

    @property
    def backend(self):
        return self.rep.backend

    @property
    def h(self):
        return self.rep.h

    @property
    def order(self):
        return self.rep.order

    @property
    def range_method(self):
        return self.rep.range_method

    def copy(self):
        return ValidatedValue(
            self.rep.copy(), self.natural_interval,
            _exact_affine_shadow=self.exact_affine_shadow,
            _diag_approx_mask=self._diag_approx_mask, _diag_add_source=False,
            _diag_recombined=self._diag_recombined,
        )

    def range(self, method=None):
        I, meta = self.rep.range(method)
        # ``natural_interval`` is an independently rigorous interval extension
        # propagated over the same physical input box.  Intersecting independent
        # enclosures is a theorem-backed tightening and never removes an exact
        # feasible value.  This is especially useful for structured matrix
        # theorems, which can attach broad spectral bounds without discarding a
        # dependency-preserving polynomial/generator representation.
        if isinstance(self.natural_interval, Interval):
            from .runtime import interval_intersection
            I = interval_intersection(I, self.natural_interval)
            meta = dict(meta)
            meta["natural_interval_intersection"] = True
        return I, meta

    def convert_to(self, prototype_representation):
        """Convert soundly to the representation family of ``prototype``."""
        return prototype_representation.convert_from(self.rep)

    def to_interval(self):
        # Use the same intersection portfolio as ``range`` so theorem-backed
        # natural intervals remain available when a structured value is
        # intentionally collapsed to interval form.
        from .interval import IntervalEnclosure
        I, _ = self.range(self.range_method)
        return IntervalEnclosure(
            I, self.h, self.order, self.backend, self.range_method
        )

    def to_polynomial(self):
        from .polynomial import PolynomialEnclosure
        prototype = PolynomialEnclosure.constant(
            0.0, self.h, self.order, self.backend, self.range_method
        )
        return prototype.convert_from(self.rep)

    def to_generator(self):
        from .generator import SharedGeneratorPolynomialEnclosure
        prototype = SharedGeneratorPolynomialEnclosure.constant(
            0.0, self.h, self.order, self.backend, self.range_method
        )
        return prototype.convert_from(self.rep)

    def _constant_like(self, value):
        v = float(value)
        return ValidatedValue(
            self.rep.constant_like(v), point_interval(v),
            _exact_affine_shadow=ExactAffineShadow.constant_value(len(self.h), v),
            _diag_approx_mask=0, _diag_add_source=False,
        )

    def _natural_binary(self, other, op):
        a = self.natural_interval
        b = other.natural_interval
        if a is None or b is None:
            return None
        try:
            if op == "add":
                return self.backend.add(a, b)
            if op == "mul":
                return self.backend.mul(a, b)
            if op == "div":
                return self.backend.div(a, b)
        except Exception:
            # The natural interval is an optional independent proof enclosure.
            # Losing it must never turn a stronger dependency-preserving proof
            # into a failure.
            return None
        return None

    def _coerce(self, other):
        if isinstance(other, ValidatedValue):
            if not same_domain(self.rep, other.rep):
                raise ValueError("ValidatedValue domains/backends do not match.")
            return other
        return self._constant_like(other)

    @staticmethod
    def _nonscalar_array(other):
        if isinstance(other, ValidatedValue):
            return None
        try:
            arr = np.asarray(other, dtype=object)
        except Exception:
            return None
        return arr if arr.ndim > 0 else None

    def _map_array(self, other, op, *, reverse=False):
        arr = self._nonscalar_array(other)
        if arr is None:
            return None
        out = np.empty(arr.shape, dtype=object)
        for idx in np.ndindex(arr.shape):
            v = arr[idx]
            if reverse:
                left = self._constant_like(v) if not isinstance(v, ValidatedValue) else v
                out[idx] = left._binary(self, op)
            else:
                out[idx] = self._binary(v, op)
        return out

    def _affine_binary(self, other, op):
        a = self.exact_affine_shadow
        b = other.exact_affine_shadow
        if a is None or b is None:
            return None
        if op == "add":
            return a + b
        if op == "mul":
            if a.is_constant:
                return b.scale(a.constant)
            if b.is_constant:
                return a.scale(b.constant)
            return None
        if op == "div":
            if b.is_constant and b.constant != 0:
                return a.scale(Fraction(1, 1) / b.constant)
            return None
        return None

    def _binary(self, other, op):
        mapped = self._map_array(other, op, reverse=False)
        if mapped is not None:
            return mapped
        other = self._coerce(other)

        natural = self._natural_binary(other, op)
        affine_shadow = self._affine_binary(other, op)

        overlap = self._diag_approx_mask & other._diag_approx_mask
        if overlap:
            record_approximation_recombination(overlap)
        inherited_mask = self._diag_approx_mask | other._diag_approx_mask
        inherited_recombined = bool(self._diag_recombined or other._diag_recombined or overlap)

        if self.rep.representation_name == other.rep.representation_name:
            return ValidatedValue(
                self.rep.binary_same(other.rep, op), natural,
                _exact_affine_shadow=affine_shadow,
                _diag_approx_mask=inherited_mask,
                _diag_recombined=inherited_recombined,
            )

        # Generic mixed-representation promotion.  The richer target is chosen
        # by an explicit representation priority; the target representation owns
        # the sound conversion rule.
        if self.rep.promotion_priority >= other.rep.promotion_priority:
            target = self.rep
        else:
            target = other.rep

        a = target.convert_from(self.rep)
        b = target.convert_from(other.rep)
        return ValidatedValue(
            a.binary_same(b, op), natural,
            _exact_affine_shadow=affine_shadow,
            _diag_approx_mask=inherited_mask,
            _diag_recombined=inherited_recombined,
        )

    def __add__(self, other):
        return self._binary(other, "add")

    __radd__ = __add__

    def __neg__(self):
        natural = None if self.natural_interval is None else interval_neg(self.natural_interval)
        return ValidatedValue(
            self.rep.negated(), natural,
            _exact_affine_shadow=(None if self.exact_affine_shadow is None else -self.exact_affine_shadow),
            _diag_approx_mask=self._diag_approx_mask, _diag_add_source=False,
            _diag_recombined=self._diag_recombined,
        )

    def unary(self, op):
        fn = getattr(self.rep, "unary", None)
        if not callable(fn):
            raise NotImplementedError(
                f"Representation {self.representation_name!r} does not support "
                f"elementary function {op!r}."
            )

        from .domain import elementary_requirement, require_domain
        from .elementary import trusted_elementary_range

        requirement = elementary_requirement(op)
        proven_domain = require_domain(self, requirement) if requirement is not None else None

        # ``abs`` has no domain obligation, but sign information is mathematically
        # decisive: on x>=0, abs(x)=x exactly; on x<=0, abs(x)=-x exactly.
        # The independently propagated natural interval can prove that sign even
        # when a structured polynomial/generator range is slightly wider because
        # of approximation remainder.  Feed that certified physical-domain range
        # into the built-in unary implementation so sign-definite ``abs`` keeps
        # the original dependency instead of falling back to [0,max(|x|)].
        refinement_domain = proven_domain
        if op in {"abs", "absolute"} and isinstance(self.natural_interval, Interval):
            refinement_domain = self.natural_interval

        # Preserve the frozen structural representation contract: third-party or
        # test representations may implement ``unary(op)`` without AutoIA's
        # internal ``domain=`` refinement hook.  Only the built-in families need
        # the tighter proven physical-domain range for their Taylor construction.
        if refinement_domain is not None and self.representation_name in {
            "interval", "polynomial", "generator-polynomial"
        }:
            rep_out = fn(op, domain=refinement_domain)
        else:
            rep_out = fn(op)

        source = proven_domain if proven_domain is not None else self.natural_interval
        natural_out = None
        if source is not None:
            try:
                natural_out = trusted_elementary_range(op, source, self.backend)
            except Exception:
                natural_out = None
        return ValidatedValue(
            rep_out, natural_out, _diag_approx_mask=self._diag_approx_mask,
            _diag_recombined=self._diag_recombined,
        )

    def __abs__(self):
        return self.unary("abs")

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        mapped = self._map_array(other, "sub", reverse=True)
        if mapped is not None:
            return mapped
        return self._coerce(other) - self

    def __mul__(self, other):
        return self._binary(other, "mul")

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary(other, "div")

    def __rtruediv__(self, other):
        mapped = self._map_array(other, "div", reverse=True)
        if mapped is not None:
            return mapped
        return self._coerce(other)._binary(self, "div")

    def __pow__(self, power):
        from .power import normalize_constant_exponent, integer_power_value
        from .domain import real_power_requirement, require_domain

        kind, p = normalize_constant_exponent(power)
        proven_domain = None
        if kind != "integer":
            requirement = real_power_requirement(float(p))
            proven_domain = require_domain(self, requirement)

        fn = getattr(self.rep, "power_constant", None)
        if callable(fn):
            if proven_domain is not None and self.representation_name in {
                "interval", "polynomial", "generator-polynomial"
            }:
                rep_out = fn(p, domain=proven_domain)
            else:
                rep_out = fn(p)
            source = proven_domain if proven_domain is not None else self.natural_interval
            natural_out = None
            if source is not None:
                try:
                    natural_out = real_power_interval(source, float(p), self.backend)
                except Exception:
                    natural_out = None
            return ValidatedValue(
                rep_out, natural_out, _diag_approx_mask=self._diag_approx_mask,
                _diag_recombined=self._diag_recombined,
            )
        if kind == "integer":
            return integer_power_value(self, p)
        raise NotImplementedError(
            f"Representation {self.representation_name!r} does not support "
            "non-integer real powers."
        )

    def __rpow__(self, base):
        raise NotImplementedError(
            "AutoIA does not yet support a validated exponent; only validated "
            "bases raised to constant real exponents are supported."
        )

    def _predicate(self, other, operator):
        from .branching import ValidatedPredicate
        other = self._coerce(other)
        return ValidatedPredicate(self - other, operator)

    def __lt__(self, other):
        return self._predicate(other, "lt")

    def __le__(self, other):
        return self._predicate(other, "le")

    def __gt__(self, other):
        return self._predicate(other, "gt")

    def __ge__(self, other):
        return self._predicate(other, "ge")

    def __eq__(self, other):
        try:
            return self._predicate(other, "eq")
        except Exception:
            return False

    def __ne__(self, other):
        try:
            return self._predicate(other, "ne")
        except Exception:
            return True

    def __bool__(self):
        # Python truth testing of a real scalar is equivalent to x != 0.
        return bool(self._predicate(0.0, "ne"))

    def __repr__(self):
        r, _ = self.range()
        return (
            f"ValidatedValue(rep='{self.representation_name}', "
            f"range=[{r.lo:.12g}, {r.hi:.12g}])"
        )


def contains_validated(arr):
    obj = np.asarray(arr, dtype=object)
    return any(isinstance(v, ValidatedValue) for v in obj.flat)


def first_validated(arrays):
    for arr in arrays:
        obj = np.asarray(arr, dtype=object)
        for v in obj.flat:
            if isinstance(v, ValidatedValue):
                return v
    return None


def unwrap_for_polynomial(arr, prototype):
    obj = np.asarray(arr, dtype=object)
    out = np.empty(obj.shape, dtype=object)
    for idx in np.ndindex(obj.shape):
        v = obj[idx]
        if isinstance(v, ValidatedValue):
            out[idx] = v.to_polynomial()
        else:
            out[idx] = v
    return out



def unwrap_for_generator(arr, prototype):
    """Return an object array of shared-generator representation objects."""
    from .generator import SharedGeneratorPolynomialEnclosure

    obj = np.asarray(arr, dtype=object)
    out = np.empty(obj.shape, dtype=object)
    if getattr(prototype, "representation_name", None) == "generator-polynomial":
        target = prototype
    else:
        target = SharedGeneratorPolynomialEnclosure.constant(
            0.0, prototype.h, prototype.order, prototype.backend, prototype.range_method
        )
    for idx in np.ndindex(obj.shape):
        v = obj[idx]
        rep = getattr(v, "rep", v)
        if hasattr(rep, "representation_name"):
            out[idx] = target.convert_from(rep)
        else:
            out[idx] = target.constant_like(float(rep))
    return out

def unwrap_for_interval(arr, prototype):
    from ..trusted import point_interval

    obj = np.asarray(arr, dtype=object)
    out = np.empty(obj.shape, dtype=object)
    for idx in np.ndindex(obj.shape):
        v = obj[idx]
        if isinstance(v, ValidatedValue):
            out[idx] = v.to_interval().interval
        else:
            out[idx] = point_interval(float(v))
    return out


def all_active_representations(arrays):
    kinds = set()
    for arr in arrays:
        obj = np.asarray(arr, dtype=object)
        for v in obj.flat:
            if isinstance(v, ValidatedValue):
                kinds.add(v.representation_name)
    return kinds


__all__ = [
    "ValidatedValue",
    "same_domain",
    "contains_validated",
    "first_validated",
    "unwrap_for_polynomial",
    "unwrap_for_generator",
    "unwrap_for_interval",
    "all_active_representations",
]
