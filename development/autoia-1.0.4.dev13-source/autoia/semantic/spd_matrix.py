"""Validated symmetric-positive-definite matrix-family semantics.

This module is intentionally application agnostic.  It handles matrix families
``A(p)`` whose scalar entries are ordinary AutoIA ``ValidatedValue`` objects
sharing the same physical uncertain variables.  The main proof is based on a
constant lower-triangular preconditioner ``L`` and the normalized family

    B(p) = L^{-1} A(p) L^{-T} = I + E(p).

If a rigorous norm bound proves ``||E(p)||_2 <= rho < 1`` throughout the
physical box, then every eigenvalue of ``B`` is positive and hence ``A`` is SPD.
The same theorem yields broad rigorous log-determinant and inverse-quadratic-
form bounds.  A truncated matrix-log series provides a dependency-preserving
``logdet`` value when the normalized family is sufficiently local.

No GP/covariance semantics live here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ..arrays import contains_validated, validated_dot, validated_trace
from ..exceptions import CertificationError, InvalidInputError
from ..trusted import (
    Interval, point_interval, abs_upper_interval,
    sqrt_interval, log_interval,
)
from ..enclosures import ValidatedValue, PolynomialEnclosure, IntervalEnclosure, SharedGeneratorPolynomialEnclosure
from ..enclosures.runtime import interval_intersection, safe_interval_representative
from ..enclosures.domain import request_sufficient_condition_refinement
from .linear_solve import GenericSolveContext


@dataclass(frozen=True)
class SPDMatrixCertificate:
    """Machine-rigorous certificate for one parameter-dependent SPD family."""

    dimension: int
    raw_identity_rho_upper: float
    raw_identity_rho_method: str
    eigenvalue_lower: float
    eigenvalue_upper: float
    eigenvalue_method: str
    series_scale: float
    series_rho_upper: float
    preconditioner: np.ndarray
    preconditioner_logdet: Interval
    logdet_spectral: Interval
    series_order: int
    machine_rigorous: bool

    @property
    def proved_spd(self) -> bool:
        return bool(self.eigenvalue_lower > 0.0)

    @property
    def rho_upper(self) -> float:
        """Backward-compatible alias for the radius used by the log series."""
        return float(self.series_rho_upper)

    @property
    def rho_method(self) -> str:
        return str(self.eigenvalue_method)



def _first_validated(*objects):
    for obj in objects:
        arr = np.asarray(obj, dtype=object)
        for item in arr.flat:
            if isinstance(item, ValidatedValue):
                return item
    return None


def _coerce_matrix(A, prototype: ValidatedValue) -> np.ndarray:
    arr = np.asarray(A, dtype=object)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise InvalidInputError("SPD matrix semantic requires a square matrix.")
    out = np.empty(arr.shape, dtype=object)
    for idx in np.ndindex(arr.shape):
        v = arr[idx]
        out[idx] = v if isinstance(v, ValidatedValue) else prototype._constant_like(float(v))
    return out


def _coerce_vector(b, prototype: ValidatedValue) -> np.ndarray:
    arr = np.asarray(b, dtype=object)
    if arr.ndim != 1:
        raise InvalidInputError("SPD inverse quadratic form requires a one-dimensional vector.")
    out = np.empty(arr.shape, dtype=object)
    for i, v in enumerate(arr):
        out[i] = v if isinstance(v, ValidatedValue) else prototype._constant_like(float(v))
    return out


def _rep_structurally_equal(a: ValidatedValue, b: ValidatedValue) -> bool:
    if a is b:
        return True
    if a.representation_name != b.representation_name:
        return False
    ra, rb = a.rep, b.rep
    name = a.representation_name
    if name == "interval":
        # Equal interval images do not prove two independently varying values
        # are the same expression.  Identity is handled above.
        return False
    if name == "polynomial":
        return (
            ra.coeff == rb.coeff
            and ra.remainder == rb.remainder
            and np.array_equal(ra.h, rb.h)
        )
    if name == "generator-polynomial":
        return (
            ra.coeff == rb.coeff
            and ra.generators == rb.generators
            and ra.remainder == rb.remainder
            and np.array_equal(ra.h, rb.h)
        )
    return False


def _identically_zero(v: ValidatedValue) -> bool:
    r = v.rep
    name = v.representation_name
    if name == "interval":
        return r.interval.lo == r.interval.hi == 0.0
    if name == "polynomial":
        return (
            all(float(c) == 0.0 for c in r.coeff.values())
            and r.remainder.lo == r.remainder.hi == 0.0
        )
    if name == "generator-polynomial":
        return (
            all(float(c) == 0.0 for c in r.coeff.values())
            and all(float(c) == 0.0 for c in r.generators.values())
            and r.remainder.lo == r.remainder.hi == 0.0
        )
    return False


def _prove_symmetric(A: np.ndarray) -> None:
    n = A.shape[0]
    for i in range(n):
        for j in range(i):
            a, b = A[i, j], A[j, i]
            if _rep_structurally_equal(a, b):
                continue
            try:
                d = a - b
            except Exception as exc:
                raise CertificationError(
                    "Could not prove that the validated matrix family is symmetric.",
                    context={"proof_obligation": "matrix symmetry", "indices": (i, j)},
                ) from exc
            if not _identically_zero(d):
                raise CertificationError(
                    "Validated SPD semantics require symmetry to be structurally proved. "
                    f"Entries ({i},{j}) and ({j},{i}) were not proved identical.",
                    context={"proof_obligation": "matrix symmetry", "indices": (i, j)},
                )


def _value_interval(v: ValidatedValue) -> Interval:
    I, _ = v.rep.range(getattr(v.rep, "range_method", None))
    natural = getattr(v, "natural_interval", None)
    if isinstance(natural, Interval):
        I = interval_intersection(I, natural)
    return I


def _nominal_scalar(v: ValidatedValue) -> float:
    rep = v.rep
    coeff = getattr(rep, "coeff", None)
    if isinstance(coeff, dict):
        z = tuple(0 for _ in range(len(v.h)))
        if z in coeff:
            return float(coeff[z])
    I = _value_interval(v)
    return safe_interval_representative(I)


def _nominal_matrix(A: np.ndarray) -> np.ndarray:
    out = np.empty(A.shape, dtype=float)
    for idx in np.ndindex(A.shape):
        out[idx] = _nominal_scalar(A[idx])
    # The exact family has already been proved symmetric.  Symmetrizing this
    # *proposal* matrix changes no proof statement and makes numerical Cholesky
    # less sensitive to proposal roundoff.
    return 0.5 * (out + out.T)


def _preconditioner_from_nominal(A0: np.ndarray) -> np.ndarray:
    n = A0.shape[0]
    try:
        L = np.linalg.cholesky(A0)
        if np.all(np.isfinite(L)) and np.all(np.diag(L) > 0.0):
            return np.asarray(L, dtype=float)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        pass

    # Safe proposal fallback: a positive diagonal scaling.  The proof below does
    # not require L to factor A0; it only requires the stored constant L to be
    # nonsingular.  A poor L merely makes rho larger and may trigger refinement.
    diag = np.diag(A0)
    scale = np.sqrt(np.maximum(np.abs(diag), 1.0))
    scale[~np.isfinite(scale)] = 1.0
    scale[scale <= 0.0] = 1.0
    return np.diag(scale)


def _left_solve_constant_lower(L: np.ndarray, M: np.ndarray, prototype: ValidatedValue) -> np.ndarray:
    M = np.asarray(M, dtype=object)
    if M.ndim == 1:
        M2 = M.reshape((-1, 1))
        vector = True
    elif M.ndim == 2:
        M2 = M
        vector = False
    else:
        raise InvalidInputError("Constant triangular solve expects a vector or matrix right-hand side.")
    n = L.shape[0]
    if L.shape != (n, n) or M2.shape[0] != n:
        raise InvalidInputError("Constant triangular preconditioner shape mismatch.")
    out = np.empty(M2.shape, dtype=object)
    for j in range(M2.shape[1]):
        for i in range(n):
            v = M2[i, j]
            if not isinstance(v, ValidatedValue):
                v = prototype._constant_like(float(v))
            for k in range(i):
                v = v - float(L[i, k]) * out[k, j]
            d = float(L[i, i])
            if not np.isfinite(d) or d == 0.0:
                raise CertificationError(
                    "Nominal SPD preconditioner is singular.",
                    context={"proof_obligation": "SPD preconditioner nonsingularity", "index": i},
                )
            out[i, j] = v / d
    return out[:, 0] if vector else out


def _normalized_family(A: np.ndarray, L: np.ndarray, prototype: ValidatedValue) -> np.ndarray:
    left = _left_solve_constant_lower(L, A, prototype)
    # B L^T = left  <=>  L B^T = left^T.
    return _left_solve_constant_lower(L, left.T, prototype).T


def _identity_minus(B: np.ndarray, prototype: ValidatedValue) -> np.ndarray:
    n = B.shape[0]
    E = np.empty(B.shape, dtype=object)
    for i in range(n):
        for j in range(n):
            E[i, j] = B[i, j] - (1.0 if i == j else 0.0)
    return E


def _spd_split_hint(E: np.ndarray, prototype: ValidatedValue) -> ValidatedValue:
    """Dependency-bearing scalar hint for physical-box refinement scoring."""
    total = prototype._constant_like(0.0)
    for v in E.flat:
        total = total + v * v
    return total


def _norm_upper(E: np.ndarray, backend) -> tuple[float, str]:
    n = E.shape[0]
    mag = np.empty((n, n), dtype=float)
    for idx in np.ndindex(E.shape):
        I = _value_interval(E[idx])
        mag[idx] = max(abs(float(I.lo)), abs(float(I.hi)))

    # Frobenius bound.
    fro2 = point_interval(0.0)
    for a in mag.flat:
        ai = point_interval(float(a))
        fro2 = backend.add(fro2, backend.mul(ai, ai))
    fro = sqrt_interval(fro2, backend).hi

    # General induced-norm inequality ||E||_2 <= sqrt(||E||_1 ||E||_inf).
    row_bounds = []
    col_bounds = []
    for i in range(n):
        s = point_interval(0.0)
        for j in range(n):
            s = backend.add(s, point_interval(float(mag[i, j])))
        row_bounds.append(float(s.hi))
    for j in range(n):
        s = point_interval(0.0)
        for i in range(n):
            s = backend.add(s, point_interval(float(mag[i, j])))
        col_bounds.append(float(s.hi))
    rmax = max(row_bounds, default=0.0)
    cmax = max(col_bounds, default=0.0)
    induced2 = backend.mul(point_interval(rmax), point_interval(cmax))
    induced = sqrt_interval(induced2, backend).hi

    if induced <= fro:
        return float(induced), "sqrt-one-times-infinity"
    return float(fro), "frobenius"


def _gershgorin_eigenvalue_bounds(B: np.ndarray, backend) -> tuple[float, float]:
    """Rigorous eigenvalue enclosure for a proved-symmetric matrix family."""
    n = B.shape[0]
    lowers = []
    uppers = []
    for i in range(n):
        diag = _value_interval(B[i, i])
        radius = point_interval(0.0)
        for j in range(n):
            if i == j:
                continue
            I = _value_interval(B[i, j])
            a = max(abs(float(I.lo)), abs(float(I.hi)))
            radius = backend.add(radius, point_interval(a))
        lower = backend.sub(Interval(diag.lo, diag.lo), Interval(radius.hi, radius.hi)).lo
        upper = backend.add(Interval(diag.hi, diag.hi), Interval(radius.hi, radius.hi)).hi
        lowers.append(float(lower)); uppers.append(float(upper))
    return min(lowers), max(uppers)


def _intersect_eigenvalue_bounds(raw_rho: float, g_lo: float, g_hi: float, backend):
    one = point_interval(1.0); r = point_interval(float(raw_rho))
    norm_lo = backend.sub(one, r).lo
    norm_hi = backend.add(one, r).hi
    lo = max(float(g_lo), float(norm_lo))
    hi = min(float(g_hi), float(norm_hi))
    if lo > hi:
        raise RuntimeError("Independent rigorous SPD eigenvalue bounds became disjoint.")
    return lo, hi


def _series_scaling_from_eigenvalues(lo: float, hi: float, backend):
    if not (lo > 0.0 and hi >= lo and np.isfinite(lo) and np.isfinite(hi)):
        raise CertificationError(
            "SPD eigenvalue enclosure is not finite and positive.",
            context={"proof_obligation": "SPD positive eigenvalue enclosure",
                     "eigenvalue_lower": float(lo), "eigenvalue_upper": float(hi)},
        )
    # Proposal scale only; it is subsequently treated as the exact real value
    # represented by this stored binary64 number.
    c = float(0.5 * lo + 0.5 * hi)
    if not np.isfinite(c) or c <= 0.0:
        c = float(lo)
    cI = point_interval(c)
    lo_rel = backend.sub(backend.div(point_interval(lo), cI), point_interval(1.0))
    hi_rel = backend.sub(backend.div(point_interval(hi), cI), point_interval(1.0))
    rho = max(abs(lo_rel.lo), abs(lo_rel.hi), abs(hi_rel.lo), abs(hi_rel.hi))
    return c, float(rho)


def _preconditioner_logdet(L: np.ndarray, backend) -> Interval:
    total = point_interval(0.0)
    for i, d in enumerate(np.diag(L)):
        if not np.isfinite(d) or d <= 0.0:
            raise CertificationError(
                "SPD preconditioner must have positive diagonal.",
                context={"proof_obligation": "SPD preconditioner", "index": i},
            )
        total = backend.add(total, log_interval(point_interval(float(d)), backend))
    return backend.mul(point_interval(2.0), total)


def _spectral_logdet_interval_from_eigenvalues(n: int, lam_lo: float, lam_hi: float,
                                                  base: Interval, backend) -> Interval:
    if lam_lo <= 0.0:
        raise CertificationError(
            "SPD proof did not establish a positive eigenvalue lower bound.",
            context={"proof_obligation": "SPD positivity", "eigenvalue_lower": float(lam_lo)},
        )
    lnlo = log_interval(point_interval(float(lam_lo)), backend)
    lnhi = log_interval(point_interval(float(lam_hi)), backend)
    nI = point_interval(float(n))
    perturb = Interval(
        backend.mul(nI, lnlo).lo,
        backend.mul(nI, lnhi).hi,
    )
    return backend.add(base, perturb)



def _matrix_matmul(A: np.ndarray, B: np.ndarray, prototype: ValidatedValue) -> np.ndarray:
    A = np.asarray(A, dtype=object); B = np.asarray(B, dtype=object)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise InvalidInputError("Matrix product shape mismatch in SPD semantic.")
    out = np.empty((A.shape[0], B.shape[1]), dtype=object)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            s = prototype._constant_like(0.0)
            for k in range(A.shape[1]):
                s = s + A[i, k] * B[k, j]
            out[i, j] = s
    return out


def _trace(A: np.ndarray, prototype: ValidatedValue) -> ValidatedValue:
    s = prototype._constant_like(0.0)
    for i in range(A.shape[0]):
        s = s + A[i, i]
    return s


def _symmetric_tail_interval(n: int, rho: float, order: int, backend) -> Interval:
    # sum_{k=r+1}^inf rho^k/k <= rho^(r+1)/((r+1)(1-rho)), 0<=rho<1.
    rI = point_interval(float(rho))
    power = point_interval(1.0)
    for _ in range(order + 1):
        power = backend.mul(power, rI)
    denom = backend.mul(
        point_interval(float(order + 1)),
        backend.sub(point_interval(1.0), rI),
    )
    tail_one = backend.div(power, denom)
    tail = backend.mul(point_interval(float(n)), tail_one)
    t = float(tail.hi)
    return Interval(-t, t)


def _inflate_value(value: ValidatedValue, extra: Interval, *, natural: Interval | None = None) -> ValidatedValue:
    rep = value.rep
    backend = value.backend
    name = value.representation_name
    if name == "polynomial":
        newrep = PolynomialEnclosure(
            dict(rep.coeff), backend.add(rep.remainder, extra), rep.h, rep.order,
            backend, rep.range_method,
        )
    elif name == "generator-polynomial":
        newrep = rep.with_remainder(backend.add(rep.remainder, extra))
    elif name == "interval":
        newrep = IntervalEnclosure(
            backend.add(rep.interval, extra), rep.h, rep.order, backend, rep.range_method
        )
    else:  # pragma: no cover - current V1 representations are exhaustive
        raise InvalidInputError(f"Unsupported representation {name!r} in SPD semantic.")
    return ValidatedValue(
        newrep,
        natural_interval=natural,
        _diag_approx_mask=getattr(value, "_diag_approx_mask", 0),
        _diag_add_source=False,
        _diag_recombined=getattr(value, "_diag_recombined", False),
    )


def _add_interval_constant(value: ValidatedValue, const: Interval, *, natural: Interval | None = None) -> ValidatedValue:
    c = safe_interval_representative(const)
    residual = value.backend.sub(const, point_interval(c))
    shifted = value + c
    return _inflate_value(shifted, residual, natural=natural)


def certify_spd_matrix(A, *, series_order: int = 4) -> tuple[np.ndarray, np.ndarray, SPDMatrixCertificate, ValidatedValue]:
    """Return normalized family ingredients and a rigorous SPD certificate.

    The returned tuple is ``(A_validated, E, certificate, prototype)``.  This is
    an internal/public-expert primitive used by :func:`validated_slogdet` and
    :func:`validated_quad_solve`.
    """
    prototype = _first_validated(A)
    if prototype is None:
        raise InvalidInputError("certify_spd_matrix expects at least one AutoIA validated matrix entry.")
    if isinstance(series_order, bool) or int(series_order) < 1:
        raise InvalidInputError("series_order must be a positive integer.")
    series_order = int(series_order)
    Av = _coerce_matrix(A, prototype)
    _prove_symmetric(Av)
    n = Av.shape[0]
    A0 = _nominal_matrix(Av)
    L = _preconditioner_from_nominal(A0)
    B = _normalized_family(Av, L, prototype)
    E = _identity_minus(B, prototype)
    raw_rho, raw_method = _norm_upper(E, prototype.backend)
    g_lo, g_hi = _gershgorin_eigenvalue_bounds(B, prototype.backend)
    lam_lo, lam_hi = _intersect_eigenvalue_bounds(
        raw_rho, g_lo, g_hi, prototype.backend
    )

    base = _preconditioner_logdet(L, prototype.backend)
    if lam_lo <= 0.0:
        margin = Interval(float(lam_lo), float(lam_hi))
        request_sufficient_condition_refinement(
            _spd_split_hint(E, prototype),
            operation="symmetric positive-definite matrix family",
            interval=margin,
            description="preconditioned symmetric eigenvalue lower bound positive",
            methods_tried=(f"preconditioned-{raw_method}", "preconditioned-gershgorin"),
        )
    spectral = _spectral_logdet_interval_from_eigenvalues(
        n, lam_lo, lam_hi, base, prototype.backend
    )
    scale, series_rho = _series_scaling_from_eigenvalues(
        lam_lo, lam_hi, prototype.backend
    )
    cert = SPDMatrixCertificate(
        dimension=n,
        raw_identity_rho_upper=float(raw_rho),
        raw_identity_rho_method=raw_method,
        eigenvalue_lower=float(lam_lo),
        eigenvalue_upper=float(lam_hi),
        eigenvalue_method="intersection(norm,g ershgorin)".replace("g ", "g"),
        series_scale=float(scale),
        series_rho_upper=float(series_rho),
        preconditioner=np.asarray(L, dtype=float),
        preconditioner_logdet=base,
        logdet_spectral=spectral,
        series_order=series_order,
        machine_rigorous=bool(prototype.backend.machine_rigorous),
    )
    return Av, B, cert, prototype


def validated_slogdet(A, *, series_order: int = 4):
    """Validated ``numpy.linalg.slogdet`` semantic for proved-SPD families.

    Returns ``(1.0, logdet_value)``.  The sign is exactly +1 after the SPD proof;
    the log-determinant retains shared physical dependence through a truncated
    matrix-log series and a rigorous norm-bounded tail.
    """
    if not contains_validated(A):
        return np.linalg.slogdet(np.asarray(A))
    Av, B, cert, prototype = certify_spd_matrix(A, series_order=series_order)
    n = Av.shape[0]
    scale = float(cert.series_scale)
    E = np.empty(B.shape, dtype=object)
    for i in range(n):
        for j in range(n):
            E[i, j] = B[i, j] / scale - (1.0 if i == j else 0.0)

    power = E
    series = _trace(power, prototype)
    for k in range(2, int(series_order) + 1):
        power = _matrix_matmul(power, E, prototype)
        term = _trace(power, prototype) / float(k)
        series = series + term if (k % 2 == 1) else series - term

    tail = _symmetric_tail_interval(n, cert.series_rho_upper, int(series_order), prototype.backend)
    series = _inflate_value(series, tail)
    scale_log = log_interval(point_interval(scale), prototype.backend)
    scale_const = prototype.backend.mul(point_interval(float(n)), scale_log)
    base_scaled = prototype.backend.add(cert.preconditioner_logdet, scale_const)
    logdet = _add_interval_constant(series, base_scaled, natural=cert.logdet_spectral)
    # Attach proof metadata for diagnostics without changing arithmetic semantics.
    logdet.spd_certificate = cert
    return 1.0, logdet


def validated_logdet(A, *, series_order: int = 4):
    """Validated log-determinant for symmetric positive-definite matrix families."""
    sign, logdet = validated_slogdet(A, series_order=series_order)
    return logdet


def validated_det(A, *, series_order: int = 4):
    """Validated determinant for SPD families, obtained from ``exp(logdet)``."""
    if not contains_validated(A):
        return np.linalg.det(np.asarray(A))
    return validated_logdet(A, series_order=series_order).unary("exp")


def _sum_square_interval(values: np.ndarray, backend) -> Interval:
    total = point_interval(0.0)
    for v in values:
        I = _value_interval(v)
        if I.lo <= 0.0 <= I.hi:
            hi = max(abs(I.lo), abs(I.hi))
            sq = backend.mul(point_interval(float(hi)), point_interval(float(hi)))
            sq = Interval(0.0, sq.hi)
        else:
            candidates = [I.lo * I.lo, I.hi * I.hi]
            loabs = min(abs(I.lo), abs(I.hi))
            hiabs = max(abs(I.lo), abs(I.hi))
            loI = backend.mul(point_interval(float(loabs)), point_interval(float(loabs)))
            hiI = backend.mul(point_interval(float(hiabs)), point_interval(float(hiabs)))
            sq = Interval(max(0.0, loI.lo), hiI.hi)
        total = backend.add(total, sq)
    return total


def _matrix_matvec(A: np.ndarray, x: np.ndarray, prototype: ValidatedValue) -> np.ndarray:
    A = np.asarray(A, dtype=object); x = np.asarray(x, dtype=object)
    if A.ndim != 2 or x.ndim != 1 or A.shape[1] != x.size:
        raise InvalidInputError("Matrix-vector product shape mismatch in SPD semantic.")
    out = np.empty(A.shape[0], dtype=object)
    for i in range(A.shape[0]):
        s = prototype._constant_like(0.0)
        for j in range(A.shape[1]):
            s = s + A[i, j] * x[j]
        out[i] = s
    return out


def _inverse_series_tail_interval(norm2: Interval, rho: float, order: int,
                                  scale: float, backend) -> Interval:
    """Bound the quadratic-form tail of the Neumann inverse series.

    For ``B = scale*(I+F)`` with ``||F||_2 <= rho < 1``, the omitted
    inverse-series operator after order ``r`` satisfies

        ||R_r||_2 <= rho^(r+1)/(1-rho).

    Hence ``|z^T R_r z|/scale`` is at most the same factor times an
    upper bound on ``||z||_2^2``.  All scalar arithmetic below is outward
    rounded by the trusted backend.
    """
    if not (0.0 <= float(rho) < 1.0):
        raise CertificationError(
            "SPD inverse-series theorem requires a spectral radius below one.",
            context={"proof_obligation": "SPD inverse-series convergence", "rho_upper": float(rho)},
        )
    rI = point_interval(float(rho))
    power = point_interval(1.0)
    for _ in range(int(order) + 1):
        power = backend.mul(power, rI)
    denom = backend.sub(point_interval(1.0), rI)
    factor = backend.div(power, denom)
    tail = backend.mul(factor, point_interval(max(0.0, float(norm2.hi))))
    tail = backend.div(tail, point_interval(float(scale)))
    t = float(tail.hi)
    return Interval(-t, t)


def validated_quad_solve(A, b, *, series_order: int = 4):
    """Return a rigorous enclosure of ``b.T @ solve(A,b)`` for SPD families.

    The primary structured route is a preconditioned Neumann series, not a
    natural interval solve.  With ``B=L^{-1} A L^{-T}=c(I+F)`` and
    ``||F||_2<1``, AutoIA retains the shared physical dependence in

        z^T [I - F + F^2 - ... + (-F)^r] z / c,

    where ``z=L^{-1}b``, and adds a rigorous norm-bounded tail.  An independent
    spectral theorem always supplies a broad positive enclosure.  When the
    ordinary dependency-preserving semantic linear solve also certifies, its
    range is intersected as an additional proof tightening; failure of that
    optional route does not invalidate the SPD theorem.
    """
    if not (contains_validated(A) or contains_validated(b)):
        A0 = np.asarray(A, dtype=float); b0 = np.asarray(b, dtype=float)
        return float(b0 @ np.linalg.solve(A0, b0))
    prototype = _first_validated(A, b)
    if prototype is None:  # defensive
        raise InvalidInputError("validated_quad_solve requires validated input.")
    Av, B, cert, _ = certify_spd_matrix(A, series_order=series_order)
    bv = _coerce_vector(b, prototype)
    if bv.size != Av.shape[0]:
        raise InvalidInputError("SPD matrix and vector dimensions do not match.")

    # Spectral theorem: z=L^{-1}b, q=z^T B^{-1}z.  If the eigenvalues of B
    # are in [lambda_L,lambda_U], then
    # ||z||^2/lambda_U <= q <= ||z||^2/lambda_L.
    z = _left_solve_constant_lower(cert.preconditioner, bv, prototype)
    z = np.asarray(z, dtype=object)
    norm2 = _sum_square_interval(z, prototype.backend)
    spectral = Interval(
        prototype.backend.div(
            point_interval(max(0.0, norm2.lo)),
            point_interval(cert.eigenvalue_upper),
        ).lo,
        prototype.backend.div(
            point_interval(norm2.hi),
            point_interval(cert.eigenvalue_lower),
        ).hi,
    )

    # Dependency-preserving inverse series around the rigorously rescaled
    # preconditioned family B = c(I+F).  The eigenvalue enclosure used to choose
    # c proves ||F||_2 <= series_rho_upper < 1 because F is symmetric.
    n = Av.shape[0]
    scale = float(cert.series_scale)
    F = np.empty(B.shape, dtype=object)
    for i in range(n):
        for j in range(n):
            F[i, j] = B[i, j] / scale - (1.0 if i == j else 0.0)

    power_z = z.copy()  # F^0 z
    q_series = validated_dot(z, power_z)
    if not isinstance(q_series, ValidatedValue):
        q_series = prototype._constant_like(float(q_series))
    for k in range(1, int(series_order) + 1):
        power_z = _matrix_matvec(F, power_z, prototype)
        term = validated_dot(z, power_z)
        if not isinstance(term, ValidatedValue):
            term = prototype._constant_like(float(term))
        q_series = q_series - term if (k % 2 == 1) else q_series + term
    q = q_series / scale
    tail = _inverse_series_tail_interval(
        norm2, cert.series_rho_upper, int(series_order), scale, prototype.backend
    )
    q = _inflate_value(q, tail, natural=spectral)

    # Optional independent tightening from the existing general semantic solve.
    # SPD has already proved invertibility, so a contraction failure here means
    # only that this sufficient solve theorem is inconclusive on the current
    # box.  It must not turn a valid SPD quadratic form into a model failure.
    solve_ctx = GenericSolveContext(prototype)
    solve_tightening = None
    try:
        x = solve_ctx.traced_solve(Av, bv)
        q_solve = validated_dot(bv, x)
        if not isinstance(q_solve, ValidatedValue):
            q_solve = prototype._constant_like(float(q_solve))
        q_solve_I, _ = q_solve.range(q_solve.range_method)
        solve_tightening = q_solve_I
        spectral = interval_intersection(spectral, q_solve_I)
        q.natural_interval = interval_intersection(q.natural_interval, spectral)
    except (RuntimeError, FloatingPointError, CertificationError):
        # Expected proof-route insufficiency.  The independently proved
        # Neumann/spectral enclosure above remains valid.
        pass

    q.spd_certificate = cert
    q.solve_certificates = tuple(solve_ctx.certificates)
    q.solve_tightening_interval = solve_tightening
    q.inverse_series_order = int(series_order)
    return q


__all__ = [
    "SPDMatrixCertificate",
    "certify_spd_matrix",
    "validated_slogdet",
    "validated_logdet",
    "validated_det",
    "validated_quad_solve",
]
