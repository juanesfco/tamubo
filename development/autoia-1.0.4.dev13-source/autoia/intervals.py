"""Public interval construction and input-specification semantics.

The proof-critical numerical interval implementation lives in
:mod:`autoia.trusted`.  This public subclass adds stable AutoIA exception
translation and, beginning with AutoIA 1.0.1, explicit constructors for users
who want the *intended physical input set* to be defined by exact decimal or
rational endpoints rather than by already-rounded Python ``float`` values.

The three public construction semantics are deliberately distinct:

``Interval(lo, hi)``
    ``lo`` and ``hi`` are first canonicalized to binary64 and those stored
    binary64 values are interpreted as exact real endpoints.  This preserves
    the AutoIA 1.0.0 contract.

``Interval.enclosing(lo, hi)``
    ``lo`` and ``hi`` must be exact specifications (decimal strings,
    :class:`decimal.Decimal`, :class:`fractions.Fraction`, or integers).  AutoIA
    returns the tight adjacent-binary64 enclosure of those exact real endpoints.

``Interval.outward(lo, hi)``
    ``lo`` and ``hi`` are ordinary stored binary64 values, but the interval is
    padded by one adjacent binary64 number on each side.  This is a convenience
    operation, not a proof of how the original values were produced.

None of these constructors changes the semantics of ordinary numerical
constants inside an engineering model.  Such constants remain exact real
constants equal to their stored binary64 values unless a recognized semantic
primitive has a more specific contract.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

import numpy as np

from .trusted.interval import Interval as _TrustedInterval
from .trusted.interval import point_interval as _trusted_point_interval
from .trusted.environment import require_binary64_environment, nextafter_binary64
from .exceptions import InvalidInputError


_EXACT_SPEC_TYPES = (str, Decimal, Fraction, int, np.integer)
_MAX_FINITE_FRACTION = Fraction.from_float(float(np.finfo(np.float64).max))


def _source_label(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    return repr(value)


def _as_exact_fraction(value: Any, *, endpoint_name: str) -> Fraction:
    """Parse one exact endpoint specification into a rational number.

    Decimal strings and ``Decimal`` values are rational exactly; ``Fraction``
    and integer values are already exact.  Ordinary floating-point values are
    intentionally rejected here because by the time AutoIA sees them their
    original decimal intent has already been lost.  Users who mean stored
    binary64 values should use ``Interval`` or ``Interval.outward`` instead.
    """
    if isinstance(value, (bool, np.bool_)):
        raise InvalidInputError(
            f"{endpoint_name} must be an exact numeric specification, not bool."
        )
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value), 1)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise InvalidInputError(
                f"{endpoint_name} must be finite for Interval.enclosing().",
                context={"value": str(value)},
            )
        return Fraction(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise InvalidInputError(f"{endpoint_name} exact decimal string is empty.")
        try:
            dec = Decimal(text)
        except (InvalidOperation, ValueError) as exc:
            raise InvalidInputError(
                f"{endpoint_name} is not a valid exact decimal string.",
                context={"value": value},
            ) from exc
        if not dec.is_finite():
            raise InvalidInputError(
                f"{endpoint_name} must be finite for Interval.enclosing().",
                context={"value": value},
            )
        return Fraction(dec)
    if isinstance(value, (float, np.floating)):
        raise InvalidInputError(
            f"{endpoint_name} was supplied as a floating-point value. "
            "Interval.enclosing() requires an exact specification such as the "
            "string '0.1', Decimal('0.1'), Fraction(1, 10), or an integer. "
            "Use Interval(lo, hi) for exact stored-binary64 semantics or "
            "Interval.outward(lo, hi) for one-ulp padding."
        )
    raise InvalidInputError(
        f"Unsupported exact endpoint type {type(value).__name__!r} for {endpoint_name}. "
        "Use a decimal string, Decimal, Fraction, or integer."
    )


def _float_fraction(x: float) -> Fraction:
    return Fraction.from_float(float(x))


def _exact_fraction_binary64_bracket(q: Fraction, *, endpoint_name: str) -> tuple[float, float]:
    """Return finite binary64 neighbors ``(lo, hi)`` with ``lo <= q <= hi``.

    ``float(q)`` is used only as a proposal.  Exact integer-rational comparison
    verifies the final enclosure.  If the proposal were unexpectedly more than
    a few adjacent values away, the routine fails closed rather than trusting
    it.  This mirrors AutoIA's broader proposal-versus-proof architecture.
    """
    require_binary64_environment()

    if q > _MAX_FINITE_FRACTION or q < -_MAX_FINITE_FRACTION:
        raise InvalidInputError(
            f"{endpoint_name} lies outside the finite binary64 range required by AutoIA V1.",
            context={"exact_value": f"{q.numerator}/{q.denominator}"},
        )

    try:
        proposal = float(q)
    except (OverflowError, ValueError) as exc:
        raise InvalidInputError(
            f"Could not form a binary64 proposal for {endpoint_name}.",
            context={"exact_value": f"{q.numerator}/{q.denominator}"},
        ) from exc

    if not np.isfinite(proposal):
        raise InvalidInputError(
            f"{endpoint_name} cannot be enclosed by finite binary64 endpoints.",
            context={"exact_value": f"{q.numerator}/{q.denominator}"},
        )

    p = _float_fraction(proposal)
    if p == q:
        return float(proposal), float(proposal)

    # Python's exact-rational-to-float conversion is normally correctly rounded,
    # so one adjacent step suffices.  We nevertheless verify containment by exact
    # rational comparison and allow a small bounded correction sequence.  Soundness
    # comes from the exact final comparison, not from assuming proposal quality.
    if p < q:
        lo = float(proposal)
        hi = float(nextafter_binary64(lo, np.inf))
        for _ in range(8):
            if not np.isfinite(hi):
                break
            if _float_fraction(lo) <= q <= _float_fraction(hi):
                return lo, hi
            lo = hi
            hi = float(nextafter_binary64(hi, np.inf))
    else:
        hi = float(proposal)
        lo = float(nextafter_binary64(hi, -np.inf))
        for _ in range(8):
            if not np.isfinite(lo):
                break
            if _float_fraction(lo) <= q <= _float_fraction(hi):
                return lo, hi
            hi = lo
            lo = float(nextafter_binary64(lo, -np.inf))

    raise InvalidInputError(
        f"Could not verify a finite adjacent-binary64 enclosure for {endpoint_name}; failing closed.",
        context={"exact_value": f"{q.numerator}/{q.denominator}"},
    )


class Interval(_TrustedInterval):
    """Closed binary64-endpoint interval used for uncertain input boxes.

    Direct construction preserves the AutoIA 1.0.0 semantics: the stored
    binary64 endpoints themselves are interpreted as exact real numbers.
    See :meth:`enclosing` when the intended physical endpoints are exact
    decimals/rationals that may not be representable in binary64.
    """

    def __post_init__(self):
        try:
            super().__post_init__()
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidInputError(
                str(exc), context={"lo": self.lo, "hi": self.hi}
            ) from exc
        # Metadata is descriptive only; proof arithmetic uses only lo/hi.
        object.__setattr__(self, "_input_semantics", "binary64-exact")
        object.__setattr__(self, "_input_specification", {
            "semantics": "binary64-exact",
            "requested_lower": repr(float(self.lo)),
            "requested_upper": repr(float(self.hi)),
            "stored_lower": float(self.lo),
            "stored_upper": float(self.hi),
            "note": "Stored binary64 endpoints are interpreted as exact real endpoints.",
        })

    @classmethod
    def enclosing(cls, lo: Any, hi: Any) -> "Interval":
        """Enclose exact decimal/rational physical endpoints in binary64.

        Parameters are intentionally *exact* specifications: decimal strings,
        ``Decimal``, ``Fraction``, or integers.  Floating-point arguments are
        rejected because their original decimal/rational intent is unavailable.

        The returned interval has finite binary64 endpoints ``L`` and ``U``
        satisfying ``L <= lo_exact <= hi_exact <= U``.  If an exact endpoint is
        binary64-representable, that side is not widened.
        """
        qlo = _as_exact_fraction(lo, endpoint_name="lower endpoint")
        qhi = _as_exact_fraction(hi, endpoint_name="upper endpoint")
        if qlo > qhi:
            raise InvalidInputError(
                "Exact lower endpoint exceeds exact upper endpoint.",
                context={"lower": _source_label(lo), "upper": _source_label(hi)},
            )
        lo_down, lo_up = _exact_fraction_binary64_bracket(qlo, endpoint_name="lower endpoint")
        hi_down, hi_up = _exact_fraction_binary64_bracket(qhi, endpoint_name="upper endpoint")
        # For a physical interval [qlo,qhi], use the lower neighbor of qlo and
        # the upper neighbor of qhi.  The other two values are retained only as
        # independently verified bracketing evidence in metadata.
        out = cls(lo_down, hi_up)
        spec = {
            "semantics": "exact-real-enclosing",
            "requested_lower": _source_label(lo),
            "requested_upper": _source_label(hi),
            "requested_lower_fraction": f"{qlo.numerator}/{qlo.denominator}",
            "requested_upper_fraction": f"{qhi.numerator}/{qhi.denominator}",
            "stored_lower": float(out.lo),
            "stored_upper": float(out.hi),
            "lower_exact_bracket": (float(lo_down), float(lo_up)),
            "upper_exact_bracket": (float(hi_down), float(hi_up)),
            "containment_verified": True,
            "note": "Stored proof interval rigorously contains the exact requested real interval.",
        }
        object.__setattr__(out, "_input_semantics", "exact-real-enclosing")
        object.__setattr__(out, "_input_specification", spec)
        return out

    @classmethod
    def outward(cls, lo: float, hi: float) -> "Interval":
        """Pad stored binary64 endpoints outward by one adjacent value.

        This is a convenience constructor for conservative padding of already
        stored binary64 endpoints.  It does *not* certify the provenance of a
        value computed before AutoIA was called.  Use :meth:`enclosing` when the
        intended endpoints are exact decimal/rational quantities.
        """
        require_binary64_environment()
        try:
            flo = float(lo)
            fhi = float(hi)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidInputError("Interval.outward() endpoints must be finite floating-point values.") from exc
        if not (np.isfinite(flo) and np.isfinite(fhi)):
            raise InvalidInputError(
                "Interval.outward() requires finite binary64 endpoints.",
                context={"lo": flo, "hi": fhi},
            )
        if flo > fhi:
            raise InvalidInputError(
                "Lower endpoint exceeds upper endpoint.",
                context={"lo": flo, "hi": fhi},
            )
        padded_lo = float(nextafter_binary64(flo, -np.inf))
        padded_hi = float(nextafter_binary64(fhi, np.inf))
        if not (np.isfinite(padded_lo) and np.isfinite(padded_hi)):
            raise InvalidInputError(
                "One-ulp outward padding would require an infinite endpoint; AutoIA V1 requires finite endpoints.",
                context={"lo": flo, "hi": fhi},
            )
        out = cls(padded_lo, padded_hi)
        spec = {
            "semantics": "binary64-one-ulp-outward",
            "requested_lower_binary64": flo,
            "requested_upper_binary64": fhi,
            "stored_lower": float(out.lo),
            "stored_upper": float(out.hi),
            "containment_verified": True,
            "note": (
                "The stored binary64 inputs were padded outward by one adjacent value. "
                "This does not prove how those input floats were produced."
            ),
        }
        object.__setattr__(out, "_input_semantics", "binary64-one-ulp-outward")
        object.__setattr__(out, "_input_specification", spec)
        return out

    @property
    def input_semantics(self) -> str:
        return str(getattr(self, "_input_semantics", "binary64-exact"))

    @property
    def input_specification(self) -> dict:
        spec = getattr(self, "_input_specification", None)
        if spec is None:
            return {
                "semantics": "binary64-exact",
                "requested_lower": repr(float(self.lo)),
                "requested_upper": repr(float(self.hi)),
                "stored_lower": float(self.lo),
                "stored_upper": float(self.hi),
                "note": "Stored binary64 endpoints are interpreted as exact real endpoints.",
            }
        return dict(spec)


def describe_input_box(intervals) -> tuple[dict, ...]:
    """Return immutable-style descriptive metadata for a physical input box."""
    out = []
    for i, I in enumerate(intervals):
        if isinstance(I, Interval):
            spec = I.input_specification
        else:
            # Internal/trusted Interval values have the same old binary64-exact
            # semantics but carry no public-constructor metadata.
            spec = {
                "semantics": "binary64-exact",
                "requested_lower": repr(float(I.lo)),
                "requested_upper": repr(float(I.hi)),
                "stored_lower": float(I.lo),
                "stored_upper": float(I.hi),
                "note": "Stored binary64 endpoints are interpreted as exact real endpoints.",
            }
        spec = dict(spec)
        spec["index"] = int(i)
        out.append(spec)
    return tuple(out)


def attach_input_specification(result, intervals):
    """Attach non-proof descriptive input metadata to a public result object."""
    specs = describe_input_box(intervals)
    result.input_specification = specs
    result.input_semantics = tuple(spec["semantics"] for spec in specs)
    result.physical_input_box = tuple(
        (float(I.lo), float(I.hi)) for I in intervals
    )
    return result


def point_interval(x: float) -> Interval:
    """Interpret a stored binary64 value as an exact real point interval."""
    try:
        r = _trusted_point_interval(x)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidInputError(str(exc), context={"value": x}) from exc
    return Interval(r.lo, r.hi)


__all__ = [
    "Interval", "point_interval", "describe_input_box", "attach_input_specification",
]
