"""Public application-theorem interface for AutoIA.

This module intentionally exposes *certified values*, not AutoIA's internal
polynomial/generator/Bernstein representations or trusted backend objects.
Application experts supply mathematical theorems; AutoIA supplies rigorous
binary64 evaluation of the theorem's numerical bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from numbers import Integral, Real
from typing import Any, Callable, Iterable, Sequence
import math
import numpy as np

from ..api import autoia
from ..intervals import Interval as PublicInterval
from ..exceptions import CertificationError, InvalidInputError, NumericalLimitError
from ..backends import RigorousBinary64Backend as _Backend
from ..trusted import (
    Interval as _Interval,
    point_interval as _point_interval,
    integer_interval as _integer_interval,
    interval_neg as _interval_neg,
    exp_interval as _exp_interval,
    sqrt_interval as _sqrt_interval,
    log_interval as _log_interval,
    log10_interval as _log10_interval,
    sin_interval as _sin_interval,
    cos_interval as _cos_interval,
    erf_interval as _erf_interval,
    standard_normal_pdf_interval as _normal_pdf_interval,
    standard_normal_cdf_interval as _normal_cdf_interval,
    real_power_interval as _real_power_interval,
    ElementaryDomainError as _ElementaryDomainError,
    rigorous_dot_point_interval as _dot_point_interval,
)


def _as_exact_interval(value: Any, backend: _Backend) -> _Interval:
    if isinstance(value, CertifiedInterval):
        if value._backend is not backend:
            raise InvalidInputError("Certified values from different theorem contexts cannot be mixed.")
        return value._interval
    if isinstance(value, (Integral, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return _integer_interval(int(value), backend)
    if isinstance(value, (Real, np.floating)) and not isinstance(value, (bool, np.bool_)):
        x = float(value)
        if not math.isfinite(x):
            raise InvalidInputError("Application-theorem constants must be finite.")
        return _point_interval(x)
    raise InvalidInputError(
        "Expected a CertifiedInterval or finite real scalar.",
        context={"received_type": type(value).__name__},
    )


def _proof_call(operation: str, function: Callable[[], _Interval]) -> _Interval:
    """Translate proof-kernel failures to the stable public AutoIA taxonomy."""
    try:
        return function()
    except (ZeroDivisionError, _ElementaryDomainError) as exc:
        raise CertificationError(
            f"Application theorem could not certify {operation}: {exc}",
            context={"proof_obligation": operation},
        ) from exc
    except (OverflowError, FloatingPointError) as exc:
        raise NumericalLimitError(
            f"Application theorem numerical limit while evaluating {operation}: {exc}",
            context={"proof_operation": operation},
        ) from exc



class CertifiedInterval:
    """Closed machine-rigorous interval for application-theorem code.

    Users can inspect ``lower``/``upper`` and use ordinary scalar arithmetic.
    The underlying trusted interval and backend are deliberately private.
    """

    __slots__ = ("_interval", "_backend")
    __array_priority__ = 1000

    def __init__(self, interval: _Interval, backend: _Backend):
        self._interval = interval
        self._backend = backend

    @property
    def lower(self) -> float:
        return float(self._interval.lo)

    @property
    def upper(self) -> float:
        return float(self._interval.hi)

    @property
    def lo(self) -> float:
        return self.lower

    @property
    def hi(self) -> float:
        return self.upper

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def midpoint(self) -> float:
        return 0.5 * self.lower + 0.5 * self.upper

    def __repr__(self) -> str:
        return f"CertifiedInterval({self.lower:.17g}, {self.upper:.17g})"

    def _coerce(self, other: Any) -> _Interval:
        return _as_exact_interval(other, self._backend)

    def __add__(self, other: Any):
        return CertifiedInterval(_proof_call("addition", lambda: self._backend.add(self._interval, self._coerce(other))), self._backend)

    def __radd__(self, other: Any):
        return self.__add__(other)

    def __sub__(self, other: Any):
        return CertifiedInterval(_proof_call("subtraction", lambda: self._backend.sub(self._interval, self._coerce(other))), self._backend)

    def __rsub__(self, other: Any):
        return CertifiedInterval(_proof_call("subtraction", lambda: self._backend.sub(self._coerce(other), self._interval)), self._backend)

    def __mul__(self, other: Any):
        return CertifiedInterval(_proof_call("multiplication", lambda: self._backend.mul(self._interval, self._coerce(other))), self._backend)

    def __rmul__(self, other: Any):
        return self.__mul__(other)

    def __truediv__(self, other: Any):
        return CertifiedInterval(_proof_call("division with denominator separated from zero", lambda: self._backend.div(self._interval, self._coerce(other))), self._backend)

    def __rtruediv__(self, other: Any):
        return CertifiedInterval(_proof_call("division with denominator separated from zero", lambda: self._backend.div(self._coerce(other), self._interval)), self._backend)

    def __neg__(self):
        return CertifiedInterval(_interval_neg(self._interval), self._backend)

    def __abs__(self):
        lo, hi = self.lower, self.upper
        if lo >= 0.0:
            return self
        if hi <= 0.0:
            return -self
        # max(-lo,hi) is an exact comparison of stored binary64 endpoints.
        return CertifiedInterval(_Interval(0.0, max(-lo, hi)), self._backend)

    def __pow__(self, power: Any):
        if not isinstance(power, Real) or isinstance(power, (bool, np.bool_)):
            return NotImplemented
        return CertifiedInterval(_proof_call("real power domain", lambda: _real_power_interval(self._interval, float(power), self._backend)), self._backend)


@dataclass(frozen=True)
class ApplicationBox:
    """Public axis-aligned box passed to application theorems."""

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        lo = np.asarray(self.lower, dtype=float)
        hi = np.asarray(self.upper, dtype=float)
        if lo.ndim != 1 or hi.ndim != 1 or lo.shape != hi.shape:
            raise InvalidInputError("ApplicationBox lower/upper must be matching one-dimensional arrays.")
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise InvalidInputError("ApplicationBox endpoints must be finite.")
        if np.any(lo > hi):
            raise InvalidInputError("ApplicationBox has a lower endpoint above its upper endpoint.")
        object.__setattr__(self, "lower", lo.copy())
        object.__setattr__(self, "upper", hi.copy())

    @property
    def dimension(self) -> int:
        return int(self.lower.size)

    @classmethod
    def from_intervals(cls, intervals: Sequence[PublicInterval]):
        ints = list(intervals)
        return cls(
            np.asarray([I.lo for I in ints], dtype=float),
            np.asarray([I.hi for I in ints], dtype=float),
        )


@dataclass(frozen=True)
class UpperBound:
    """One application theorem's independently rigorous numerical upper bound."""

    value: float
    target: str
    theorem: str
    machine_rigorous: bool = True
    theorem_status: str = "user_asserted"
    assumptions_checked: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def __float__(self):
        return float(self.value)


@dataclass(frozen=True)
class LowerBound:
    value: float
    target: str
    theorem: str
    machine_rigorous: bool = True
    theorem_status: str = "user_asserted"
    assumptions_checked: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def __float__(self):
        return float(self.value)


@dataclass(frozen=True)
class BoundPortfolioResult:
    """Minimum of independently valid upper bounds for the same target."""

    value: float
    target: str
    winner: UpperBound
    members: tuple[UpperBound, ...]

    @property
    def machine_rigorous(self) -> bool:
        return bool(all(member.machine_rigorous for member in self.members))


@dataclass(frozen=True)
class LowerBoundPortfolioResult:
    """Maximum of independently valid lower bounds for the same target."""

    value: float
    target: str
    winner: LowerBound
    members: tuple[LowerBound, ...]

    @property
    def machine_rigorous(self) -> bool:
        return bool(all(member.machine_rigorous for member in self.members))


@dataclass(frozen=True)
class CertifiedSPDMatrix:
    """Application-theorem certificate for a symmetric positive-definite family."""

    dimension: int
    eigenvalue_lower: float
    eigenvalue_upper: float
    preconditioner: np.ndarray
    preconditioner_logdet: CertifiedInterval
    machine_rigorous: bool = True



class CertificationContext:
    """Small public toolbox for rigorous theorem-bound evaluation.

    It deliberately does not expose the trusted backend itself.
    """

    __slots__ = ("_backend", "_checked")

    def __init__(self):
        self._backend = _Backend()
        self._checked: list[str] = []

    @property
    def machine_rigorous(self) -> bool:
        return bool(self._backend.machine_rigorous)

    @property
    def checked_assumptions(self) -> tuple[str, ...]:
        return tuple(self._checked)

    def require(self, condition: Any, description: str) -> None:
        """Record a theorem assumption after checking it at runtime."""
        try:
            ok = bool(condition)
        except Exception as exc:
            raise InvalidInputError(f"Could not evaluate theorem assumption: {description}") from exc
        if not ok:
            raise CertificationError(
                f"Application-theorem assumption failed: {description}",
                context={"proof_obligation": "application theorem assumption"},
            )
        self._checked.append(str(description))

    def point(self, value: Any) -> CertifiedInterval:
        return CertifiedInterval(_as_exact_interval(value, self._backend), self._backend)

    def interval(self, lower: Any, upper: Any) -> CertifiedInterval:
        lo = float(lower); hi = float(upper)
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise InvalidInputError("Certified interval endpoints must be finite and ordered.")
        return CertifiedInterval(_Interval(lo, hi), self._backend)

    def intersect(self, *values: CertifiedInterval) -> CertifiedInterval:
        """Intersect independent certified enclosures of the same quantity.

        If every member encloses one common real quantity, their intersection is
        another rigorous enclosure and is never wider than any member.  This is a
        general proof-composition operation: AutoIA does not need to know why the
        independently derived enclosures are valid.
        """
        if len(values) == 1 and isinstance(values[0], (list, tuple)):
            values = tuple(values[0])
        if not values:
            raise InvalidInputError("At least one CertifiedInterval is required for intersection.")
        if not all(isinstance(v, CertifiedInterval) for v in values):
            raise InvalidInputError("intersect expects CertifiedInterval objects.")
        if any(v._backend is not self._backend for v in values):
            raise InvalidInputError("Certified values from different theorem contexts cannot be intersected.")
        lo = max(v.lower for v in values)
        hi = min(v.upper for v in values)
        if lo > hi:
            raise CertificationError(
                "Independent certified enclosures have empty intersection.",
                context={"proof_obligation": "certified enclosure intersection"},
            )
        return self.interval(lo, hi)

    def input_coordinate(self, box: ApplicationBox, index: int) -> CertifiedInterval:
        return self.interval(box.lower[int(index)], box.upper[int(index)])

    def enclose(self, function: Callable[[Any], Any], box: ApplicationBox | Sequence[PublicInterval], **autoia_options):
        """Ask ordinary AutoIA to enclose a function over a box.

        Scalar functions return ``CertifiedInterval``. Array-valued functions
        return an object array of ``CertifiedInterval`` objects with the same
        output shape.
        """
        if not isinstance(box, ApplicationBox):
            box = ApplicationBox.from_intervals(box)
        public_box = [PublicInterval(l, u) for l, u in zip(box.lower, box.upper)]
        if "backend" in autoia_options:
            raise InvalidInputError("cert.enclose manages its rigorous backend; do not pass backend= explicitly.")
        # Reuse this context's already-audited backend while keeping the backend
        # object itself private from application code.
        result = autoia(function, public_box, backend=self._backend, **autoia_options)
        lower = np.asarray(result.lower_shaped, dtype=float)
        upper = np.asarray(result.upper_shaped, dtype=float)
        if lower.shape == ():
            return self.interval(float(lower), float(upper))
        out = np.empty(lower.shape, dtype=object)
        for idx in np.ndindex(lower.shape):
            out[idx] = self.interval(lower[idx], upper[idx])
        return out

    def sum(self, values: Iterable[Any]) -> CertifiedInterval:
        total = self.point(0)
        for value in values:
            total = total + value
        return total

    def dot(self, point_values: Sequence[Any], certified_values: Sequence[Any]) -> CertifiedInterval:
        p = np.asarray(point_values)
        c = list(certified_values)
        if p.ndim != 1 or p.size != len(c):
            raise InvalidInputError("dot expects matching one-dimensional inputs.")
        # Use the general trusted reduction internally, but expose only this
        # stable public operation to application authors.
        intervals = np.asarray([_as_exact_interval(v, self._backend) for v in c], dtype=object)
        return CertifiedInterval(_proof_call("point-by-interval dot product", lambda: _dot_point_interval(p.astype(float), intervals, self._backend)), self._backend)

    def square(self, value: Any) -> CertifiedInterval:
        x = self.point(value) if not isinstance(value, CertifiedInterval) else value
        lo, hi = x.lower, x.upper
        if lo >= 0.0:
            return x * x
        if hi <= 0.0:
            return x * x
        u = max(-lo, hi)
        upper = (self.point(u) * self.point(u)).upper
        return self.interval(0.0, upper)

    def maximum(self, a: Any, b: Any) -> CertifiedInterval:
        A = self.point(a) if not isinstance(a, CertifiedInterval) else a
        B = self.point(b) if not isinstance(b, CertifiedInterval) else b
        return self.interval(max(A.lower, B.lower), max(A.upper, B.upper))

    def exp(self, value: Any) -> CertifiedInterval:
        x = self.point(value) if not isinstance(value, CertifiedInterval) else value
        return CertifiedInterval(_proof_call("exp", lambda: _exp_interval(x._interval, self._backend)), self._backend)

    def sqrt(self, value: Any) -> CertifiedInterval:
        x = self.point(value) if not isinstance(value, CertifiedInterval) else value
        return CertifiedInterval(_proof_call("sqrt real domain", lambda: _sqrt_interval(x._interval, self._backend)), self._backend)

    def log(self, value: Any) -> CertifiedInterval:
        x = self.point(value) if not isinstance(value, CertifiedInterval) else value
        return CertifiedInterval(_proof_call("log real domain", lambda: _log_interval(x._interval, self._backend)), self._backend)

    def log10(self, value: Any) -> CertifiedInterval:
        x = self.point(value) if not isinstance(value, CertifiedInterval) else value
        return CertifiedInterval(_proof_call("log10 real domain", lambda: _log10_interval(x._interval, self._backend)), self._backend)

    def sin(self, value: Any) -> CertifiedInterval:
        x = self.point(value) if not isinstance(value, CertifiedInterval) else value
        return CertifiedInterval(_proof_call("sin", lambda: _sin_interval(x._interval, self._backend)), self._backend)

    def cos(self, value: Any) -> CertifiedInterval:
        x = self.point(value) if not isinstance(value, CertifiedInterval) else value
        return CertifiedInterval(_proof_call("cos", lambda: _cos_interval(x._interval, self._backend)), self._backend)

    def erf(self, value: Any) -> CertifiedInterval:
        x = self.point(value) if not isinstance(value, CertifiedInterval) else value
        return CertifiedInterval(_proof_call("erf", lambda: _erf_interval(x._interval, self._backend)), self._backend)

    def standard_normal_pdf(self, value: Any) -> CertifiedInterval:
        x = self.point(value) if not isinstance(value, CertifiedInterval) else value
        return CertifiedInterval(_proof_call("standard normal PDF", lambda: _normal_pdf_interval(x._interval, self._backend)), self._backend)

    def standard_normal_cdf(self, value: Any) -> CertifiedInterval:
        x = self.point(value) if not isinstance(value, CertifiedInterval) else value
        return CertifiedInterval(_proof_call("standard normal CDF", lambda: _normal_cdf_interval(x._interval, self._backend)), self._backend)

    def _matrix_certified(self, A: Any) -> np.ndarray:
        arr = np.asarray(A, dtype=object)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise InvalidInputError("SPD theorem requires a square matrix.")
        out = np.empty(arr.shape, dtype=object)
        for idx in np.ndindex(arr.shape):
            v = arr[idx]
            out[idx] = v if isinstance(v, CertifiedInterval) else self.point(v)
        # Prove symmetry from shared certified objects or exact point constants.
        for i in range(arr.shape[0]):
            for j in range(i):
                a, b = out[i, j], out[j, i]
                same_object = a is b
                same_exact_point = (
                    a.lower == a.upper == b.lower == b.upper
                )
                if not (same_object or same_exact_point):
                    raise CertificationError(
                        "cert.spd/logdet/quad_solve require symmetry to be explicit: "
                        "assign the same CertifiedInterval object to mirrored uncertain entries.",
                        context={"proof_obligation": "matrix symmetry", "indices": (i, j)},
                    )
        return out

    def _spd_preconditioned(self, A: Any) -> CertifiedSPDMatrix:
        A = self._matrix_certified(A)
        n = A.shape[0]
        A0 = np.empty((n, n), dtype=float)
        for idx in np.ndindex(A.shape):
            A0[idx] = A[idx].midpoint
        A0 = 0.5 * (A0 + A0.T)
        try:
            L = np.linalg.cholesky(A0)
        except np.linalg.LinAlgError:
            scale = np.sqrt(np.maximum(np.abs(np.diag(A0)), 1.0))
            L = np.diag(scale)

        # B = L^{-1} A L^{-T}, using only certified scalar arithmetic.
        left = np.empty_like(A, dtype=object)
        for j in range(n):
            col = self.solve_lower_triangular(L, A[:, j])
            left[:, j] = col
        Bt = np.empty_like(A, dtype=object)
        for j in range(n):
            col = self.solve_lower_triangular(L, left.T[:, j])
            Bt[:, j] = col
        B = Bt.T

        gl = []
        gu = []
        for i in range(n):
            radius = self.point(0.0)
            for j in range(n):
                if i != j:
                    radius = radius + abs(B[i, j])
            gl.append((B[i, i] - radius).lower)
            gu.append((B[i, i] + radius).upper)
        lam_lo = min(gl); lam_hi = max(gu)
        if not (lam_lo > 0.0 and np.isfinite(lam_hi)):
            raise CertificationError(
                "Preconditioned SPD theorem is inconclusive on this interval matrix family.",
                context={"proof_obligation": "SPD positive eigenvalue lower bound",
                         "eigenvalue_lower": lam_lo, "eigenvalue_upper": lam_hi},
            )
        base = self.point(0.0)
        for i in range(n):
            self.require(float(L[i, i]) > 0.0, f"SPD preconditioner L[{i},{i}] positive")
            base = base + 2.0 * self.log(self.point(float(L[i, i])))
        return CertifiedSPDMatrix(
            n, float(lam_lo), float(lam_hi), np.asarray(L, dtype=float), base,
            self.machine_rigorous,
        )

    def spd(self, A: Any) -> CertifiedSPDMatrix:
        """Prove a symmetric interval matrix family positive definite."""
        return self._spd_preconditioned(A)

    def logdet(self, A: Any) -> CertifiedInterval:
        """Rigorous log-determinant interval for a proved-SPD matrix family."""
        proof = self._spd_preconditioned(A)
        n = proof.dimension
        perturb = self.interval(
            n * self.log(self.point(proof.eigenvalue_lower)).lower,
            n * self.log(self.point(proof.eigenvalue_upper)).upper,
        )
        return proof.preconditioner_logdet + perturb

    def quad_solve(self, A: Any, b: Sequence[Any]) -> CertifiedInterval:
        """Rigorous interval for ``b.T @ A^{-1} @ b`` for an SPD family."""
        proof = self._spd_preconditioned(A)
        vals = [v if isinstance(v, CertifiedInterval) else self.point(v) for v in b]
        if len(vals) != proof.dimension:
            raise InvalidInputError("SPD matrix and vector dimensions do not match.")
        z = self.solve_lower_triangular(proof.preconditioner, vals)
        norm2 = self.sum(self.square(v) for v in z)
        return self.interval(
            (norm2 / proof.eigenvalue_upper).lower,
            (norm2 / proof.eigenvalue_lower).upper,
        )

    def solve_lower_triangular(self, L: Any, rhs: Sequence[Any]) -> np.ndarray:
        """Natural rigorous forward substitution for a point lower-triangular matrix."""
        L = np.asarray(L, dtype=float)
        vals = list(rhs)
        n = len(vals)
        if L.shape != (n, n):
            raise InvalidInputError("L must be square and match rhs length.")
        out = np.empty(n, dtype=object)
        for j in range(n):
            numer = self.point(vals[j]) if not isinstance(vals[j], CertifiedInterval) else vals[j]
            for i in range(j):
                numer = numer - float(L[j, i]) * out[i]
            self.require(float(L[j, j]) != 0.0, f"L[{j},{j}] is nonzero")
            out[j] = numer / float(L[j, j])
        return out

    def upper_bound(self, value: CertifiedInterval, *, target: str = "quantity", details: dict[str, Any] | None = None) -> UpperBound:
        if not isinstance(value, CertifiedInterval):
            raise InvalidInputError(
                "upper_bound requires a CertifiedInterval; use cert.point(x) explicitly for an exact stored binary64 constant."
            )
        x = value
        return UpperBound(
            value=x.upper,
            target=str(target),
            theorem="unwrapped application bound",
            machine_rigorous=self.machine_rigorous,
            assumptions_checked=self.checked_assumptions,
            details={} if details is None else dict(details),
        )

    def lower_bound(self, value: CertifiedInterval, *, target: str = "quantity", details: dict[str, Any] | None = None) -> LowerBound:
        if not isinstance(value, CertifiedInterval):
            raise InvalidInputError(
                "lower_bound requires a CertifiedInterval; use cert.point(x) explicitly for an exact stored binary64 constant."
            )
        x = value
        return LowerBound(
            value=x.lower,
            target=str(target),
            theorem="unwrapped application bound",
            machine_rigorous=self.machine_rigorous,
            assumptions_checked=self.checked_assumptions,
            details={} if details is None else dict(details),
        )


class ApplicationTheorem:
    """Callable wrapper carrying explicit application-theorem metadata."""

    def __init__(self, function: Callable, *, name: str, target: str, theorem_status: str):
        self.function = function
        self.name = str(name)
        self.target = str(target)
        self.theorem_status = str(theorem_status)
        wraps(function)(self)

    def __call__(self, box: ApplicationBox | Sequence[PublicInterval] | tuple[Any, Any], *args, **kwargs) -> UpperBound:
        if isinstance(box, ApplicationBox):
            app_box = box
        elif isinstance(box, tuple) and len(box) == 2:
            app_box = ApplicationBox(box[0], box[1])
        else:
            app_box = ApplicationBox.from_intervals(box)
        cert = CertificationContext()
        result = self.function(app_box, cert, *args, **kwargs)
        if isinstance(result, UpperBound):
            value = float(result.value)
            details = dict(result.details)
        elif isinstance(result, CertifiedInterval):
            value = result.upper
            details = {}
        else:
            raise InvalidInputError(
                "An application upper-bound theorem must return CertifiedInterval or UpperBound. "
                "Use cert.point(x) explicitly for a fixed binary64 constant."
            )
        return UpperBound(
            value=value,
            target=self.target,
            theorem=self.name,
            machine_rigorous=cert.machine_rigorous,
            theorem_status=self.theorem_status,
            assumptions_checked=cert.checked_assumptions,
            details=details,
        )

    def __repr__(self):
        return f"ApplicationTheorem(name={self.name!r}, target={self.target!r}, status={self.theorem_status!r})"


def application_theorem(*, name: str, target: str = "quantity", theorem_status: str = "user_asserted"):
    """Decorate a user-supplied mathematical upper-bound theorem.

    ``theorem_status='user_asserted'`` means AutoIA certifies the numerical
    evaluation but does not claim to have proved the mathematical implication.
    Built-in, audited theorem templates may use ``'autoia_trusted'``.
    """
    if theorem_status not in {"user_asserted", "externally_verified", "autoia_trusted"}:
        raise InvalidInputError("Unknown theorem_status.")
    def decorate(function):
        return ApplicationTheorem(function, name=name, target=target, theorem_status=theorem_status)
    return decorate


def minimum_upper_bound(*bounds: UpperBound) -> BoundPortfolioResult:
    """Take the minimum of independently valid upper bounds for one target."""
    if len(bounds) == 1 and isinstance(bounds[0], (list, tuple)):
        bounds = tuple(bounds[0])
    if not bounds:
        raise InvalidInputError("At least one upper bound is required.")
    if not all(isinstance(b, UpperBound) for b in bounds):
        raise InvalidInputError("minimum_upper_bound expects UpperBound objects.")
    target = bounds[0].target
    if any(b.target != target for b in bounds):
        raise InvalidInputError("All portfolio bounds must target the same quantity.")
    winner = min(bounds, key=lambda b: b.value)
    return BoundPortfolioResult(float(winner.value), target, winner, tuple(bounds))


def maximum_lower_bound(*bounds: LowerBound) -> LowerBoundPortfolioResult:
    """Take the maximum of independently valid lower bounds for one target."""
    if len(bounds) == 1 and isinstance(bounds[0], (list, tuple)):
        bounds = tuple(bounds[0])
    if not bounds:
        raise InvalidInputError("At least one lower bound is required.")
    if not all(isinstance(b, LowerBound) for b in bounds):
        raise InvalidInputError("maximum_lower_bound expects LowerBound objects.")
    target = bounds[0].target
    if any(b.target != target for b in bounds):
        raise InvalidInputError("All portfolio bounds must target the same quantity.")
    winner = max(bounds, key=lambda b: b.value)
    return LowerBoundPortfolioResult(float(winner.value), target, winner, tuple(bounds))


__all__ = [
    "ApplicationBox",
    "CertifiedInterval",
    "CertificationContext",
    "UpperBound",
    "LowerBound",
    "BoundPortfolioResult",
    "LowerBoundPortfolioResult",
    "CertifiedSPDMatrix",
    "ApplicationTheorem",
    "application_theorem",
    "minimum_upper_bound",
    "maximum_lower_bound",
]
