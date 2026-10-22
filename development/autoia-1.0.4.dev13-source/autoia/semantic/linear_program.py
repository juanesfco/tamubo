"""Certified semantic primitive for :func:`scipy.optimize.linprog`.

Version-1 contract
------------------
AutoIA certifies the *optimal value* ``res.fun`` of a linear program

    min_z  c(q)^T z
    s.t.   A_ub(q) z <= b_ub(q),
           l <= z <= u,

where the LP data may depend on the current validated physical inputs ``q``.
The ordinary HiGHS solve is proposal machinery only.  Soundness comes from
independent weak-duality lower witnesses and universally verified primal
feasibility upper witnesses.

The returned proxy deliberately does not expose ``res.x`` during validated
execution.  A parametric optimizer z(q) is a different semantic object and is
outside this first implementation.

Equality constraints are supported by an exact semantic reduction to the two
inequalities ``A_eq z <= b_eq`` and ``-A_eq z <= -b_eq``. Mixed-integer
semantics remain deliberately deferred.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..enclosures import ValidatedValue, IntervalEnclosure
from ..trusted import Interval
from .linear_program_rlt import try_rlt_lower_bound


# ---------------------------------------------------------------------------
# Small structural helpers
# ---------------------------------------------------------------------------


def _contains_validated(obj: Any) -> bool:
    if isinstance(obj, ValidatedValue):
        return True
    if obj is None:
        return False
    if isinstance(obj, dict):
        return any(_contains_validated(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_validated(v) for v in obj)
    if isinstance(obj, np.ndarray):
        try:
            return any(_contains_validated(v) for v in np.asarray(obj, dtype=object).flat)
        except Exception:
            return False
    return False


def _as_value(x: Any, prototype: ValidatedValue) -> ValidatedValue:
    if isinstance(x, ValidatedValue):
        return x
    if isinstance(x, (bool, np.bool_)):
        return prototype._constant_like(float(bool(x)))
    try:
        v = float(x)
    except Exception as exc:
        raise NotImplementedError(
            "AutoIA validated scipy.optimize.linprog currently requires real scalar LP coefficients."
        ) from exc
    if not np.isfinite(v):
        raise NotImplementedError(
            "AutoIA validated scipy.optimize.linprog currently requires finite c/A/b coefficients."
        )
    return prototype._constant_like(v)


def _value_interval(x: Any, prototype: ValidatedValue) -> Interval:
    return _as_value(x, prototype).range()[0]


def _representative(x: Any, prototype: ValidatedValue) -> float:
    v = _as_value(x, prototype)
    rep = v.rep
    coeff = getattr(rep, "coeff", None)
    if isinstance(coeff, dict):
        zero = tuple([0] * len(rep.h))
        c0 = float(coeff.get(zero, 0.0))
        if np.isfinite(c0):
            return c0
    I, _ = v.range()
    if not (np.isfinite(I.lo) and np.isfinite(I.hi)):
        raise FloatingPointError("Could not form a finite linprog proposal coefficient.")
    return float(0.5 * I.lo + 0.5 * I.hi)


def _scenario_value(x: Any, prototype: ValidatedValue, fraction: float) -> float:
    I = _value_interval(x, prototype)
    f = float(fraction)
    if f <= 0.0:
        return float(I.lo)
    if f >= 1.0:
        return float(I.hi)
    # Convex combination avoids unnecessary overflow from hi-lo.
    return float((1.0 - f) * I.lo + f * I.hi)


def _object_vector(x: Any, n: int | None = None, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=object)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if n is not None and arr.size != int(n):
        raise ValueError(f"{name} has length {arr.size}, expected {n}.")
    return arr


def _object_matrix(x: Any, ncols: int, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=object)
    if arr.ndim == 1 and arr.size == 0:
        return arr.reshape(0, ncols)
    if arr.ndim == 1 and ncols == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[1] != int(ncols):
        raise ValueError(f"{name} must have shape (m, {ncols}).")
    return arr


def _normalize_bounds(bounds: Any, n: int) -> tuple[np.ndarray, np.ndarray, list[tuple[Any, Any]]]:
    """Normalize the ordinary SciPy linprog bounds convention.

    AutoIA V1 requires bounds themselves to be constant.  ``None`` follows
    SciPy's default and means ``(0, None)`` for every variable.
    """
    if bounds is None:
        pairs = [(0.0, None)] * n
    else:
        if _contains_validated(bounds):
            raise NotImplementedError(
                "AutoIA validated scipy.optimize.linprog currently requires constant variable bounds."
            )
        # A single pair such as (0, None) applies to every variable.
        if isinstance(bounds, tuple) and len(bounds) == 2 and not isinstance(bounds[0], (tuple, list, np.ndarray)):
            pairs = [tuple(bounds)] * n
        else:
            try:
                seq = list(bounds)
            except Exception as exc:
                raise ValueError("linprog bounds must be None, one (lo,hi) pair, or n pairs.") from exc
            if len(seq) == 2 and n != 2 and not isinstance(seq[0], (tuple, list, np.ndarray)):
                pairs = [tuple(seq)] * n
            else:
                if len(seq) != n:
                    raise ValueError(f"linprog bounds contains {len(seq)} entries, expected {n}.")
                pairs = []
                for item in seq:
                    if item is None:
                        pairs.append((0.0, None))
                    else:
                        pair = tuple(item)
                        if len(pair) != 2:
                            raise ValueError("Each linprog bound must be a (lower, upper) pair.")
                        pairs.append(pair)

    lo = np.empty(n, dtype=float)
    hi = np.empty(n, dtype=float)
    clean = []
    for j, (a, b) in enumerate(pairs):
        # SciPy's None means an infinite endpoint.
        l = -np.inf if a is None else float(a)
        u = np.inf if b is None else float(b)
        if np.isnan(l) or np.isnan(u) or l > u:
            raise ValueError(f"Invalid linprog bounds at variable {j}: {(a,b)!r}.")
        lo[j], hi[j] = l, u
        clean.append((None if not np.isfinite(l) else float(l),
                      None if not np.isfinite(u) else float(u)))
    return lo, hi, clean


def _dedupe_vectors(vectors: Iterable[np.ndarray], tol: float = 0.0) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for raw in vectors:
        v = np.asarray(raw, dtype=float).reshape(-1)
        if np.any(~np.isfinite(v)):
            continue
        duplicate = False
        for old in out:
            if v.shape == old.shape and np.allclose(v, old, rtol=0.0, atol=tol):
                duplicate = True
                break
        if not duplicate:
            out.append(v.copy())
    return out


# ---------------------------------------------------------------------------
# Certified one-sided formulas
# ---------------------------------------------------------------------------


class _InvalidDualCandidate(Exception):
    pass


def _support_infimum(d: ValidatedValue, lower: float, upper: float, physical_box=None) -> ValidatedValue:
    """Return inf_{lower<=z<=upper} d*z as a validated value.

    For finite bounds the exact branch is retained.  If the sign of ``d`` is
    uncertain, AutoIA's ordinary path controller owns the resulting split.
    One-sided infinite bounds require the corresponding reduced-cost sign.
    """
    I, _ = d.range()
    exact_lo = exact_hi = None
    shadow = getattr(d, "exact_affine_shadow", None)
    if physical_box is not None and shadow is not None:
        try:
            exact_lo, exact_hi = shadow.range_on_box(physical_box)
        except Exception:
            exact_lo = exact_hi = None
    if exact_lo is not None:
        if exact_lo >= 0:
            I = Interval(max(float(I.lo), 0.0), float(I.hi))
        if exact_hi <= 0:
            I = Interval(float(I.lo), min(float(I.hi), 0.0))
        if exact_lo == 0 and exact_hi == 0:
            return d * 0.0
    lfin = np.isfinite(lower)
    ufin = np.isfinite(upper)

    if lfin and ufin:
        if I.lo >= 0.0:
            return float(lower) * d
        if I.hi <= 0.0:
            return float(upper) * d
        # The reduced-cost sign changes somewhere on the current physical box.
        # A single fixed dual witness then induces a piecewise support function.
        # V1 deliberately does NOT trigger hidden physical subdivision merely to
        # tighten this proposal certificate.  Collapse only this ambiguous
        # support term to its rigorous global lower endpoint; all sign-definite
        # terms retain their original dependency-bearing representation.
        product = d.backend.mul(I, Interval(float(lower), float(upper)))
        return d._constant_like(float(product.lo))

    if lfin and not ufin:
        if I.lo >= 0.0:
            return float(lower) * d
        raise _InvalidDualCandidate("reduced cost not uniformly nonnegative with +infinite upper bound")

    if not lfin and ufin:
        if I.hi <= 0.0:
            return float(upper) * d
        raise _InvalidDualCandidate("reduced cost not uniformly nonpositive with -infinite lower bound")

    # Free variable: the infimum is finite only when d is identically zero.
    if I.lo == 0.0 and I.hi == 0.0:
        return d * 0.0
    raise _InvalidDualCandidate("nonzero reduced cost for a free variable")


def _dual_lower_expression(
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    y: np.ndarray,
    prototype: ValidatedValue,
    physical_box=None,
) -> ValidatedValue:
    m, n = A.shape
    yy = np.asarray(y, dtype=float).reshape(-1)
    if yy.size != m:
        raise _InvalidDualCandidate("dual dimension mismatch")
    # Projection onto the sign-feasible dual cone is proposal-only.  Once fixed,
    # every entry is exactly <= 0 and weak duality applies independently of how
    # the vector was obtained.
    yy = np.minimum(yy, 0.0)

    total = prototype._constant_like(0.0)
    for i in range(m):
        if yy[i] != 0.0:
            total = total + float(yy[i]) * _as_value(b[i], prototype)

    for j in range(n):
        d = _as_value(c[j], prototype)
        for i in range(m):
            if yy[i] != 0.0:
                d = d - float(yy[i]) * _as_value(A[i, j], prototype)
        total = total + _support_infimum(d, float(lower[j]), float(upper[j]), physical_box)
    return total


def _primal_objective_expression(c, z, prototype):
    total = prototype._constant_like(0.0)
    for cj, zj in zip(c, np.asarray(z, dtype=float).reshape(-1)):
        if float(zj) != 0.0:
            total = total + float(zj) * _as_value(cj, prototype)
    return total


def _verify_primal_candidate(c, A, b, lower, upper, z, prototype, physical_box=None):
    zz = np.asarray(z, dtype=float).reshape(-1)
    n = len(lower)
    if zz.size != n or np.any(~np.isfinite(zz)):
        return None
    for j in range(n):
        if zz[j] < lower[j] or zz[j] > upper[j]:
            return None

    residual_upper = []
    for i in range(A.shape[0]):
        r = prototype._constant_like(0.0)
        for j in range(n):
            if zz[j] != 0.0:
                r = r + float(zz[j]) * _as_value(A[i, j], prototype)
        r = r - _as_value(b[i], prototype)
        RI, _ = r.range()
        exact_ok = None
        shadow = getattr(r, "exact_affine_shadow", None)
        if physical_box is not None and shadow is not None:
            try:
                exact_ok = shadow.decision_on_box("le", physical_box)
            except Exception:
                exact_ok = None
        effective_hi = 0.0 if exact_ok is True and RI.hi > 0.0 else float(RI.hi)
        residual_upper.append(effective_hi)
        if exact_ok is not True and RI.hi > 0.0:
            return None

    obj = _primal_objective_expression(c, zz, prototype)
    OI, _ = obj.range()
    return {
        "z": zz,
        "objective": obj,
        "objective_upper": float(OI.hi),
        "residual_upper": np.asarray(residual_upper, dtype=float),
    }


# ---------------------------------------------------------------------------
# Numerical proposal generation (never trusted)
# ---------------------------------------------------------------------------


def _numeric_data_at_fraction(c, A, b, prototype, fraction):
    cn = np.asarray([_scenario_value(v, prototype, fraction) for v in c], dtype=float)
    if A.size:
        An = np.empty(A.shape, dtype=float)
        for idx in np.ndindex(A.shape):
            An[idx] = _scenario_value(A[idx], prototype, fraction)
        bn = np.asarray([_scenario_value(v, prototype, fraction) for v in b], dtype=float)
    else:
        An = None
        bn = None
    return cn, An, bn


def _solve_proposal(real_linprog, c, A, b, bounds):
    try:
        return real_linprog(
            np.asarray(c, dtype=float),
            A_ub=None if A is None or np.size(A) == 0 else np.asarray(A, dtype=float),
            b_ub=None if b is None or np.size(b) == 0 else np.asarray(b, dtype=float),
            bounds=bounds,
            method="highs",
        )
    except Exception:
        return None


def _extract_dual(result, m):
    if result is None or not bool(getattr(result, "success", False)):
        return None
    if m == 0:
        return np.empty(0, dtype=float)
    block = getattr(result, "ineqlin", None)
    marginals = getattr(block, "marginals", None)
    if marginals is None:
        return None
    y = np.asarray(marginals, dtype=float).reshape(-1)
    if y.size != m or np.any(~np.isfinite(y)):
        return None
    return np.minimum(y, 0.0)


def _extract_primal(result, n):
    if result is None or not bool(getattr(result, "success", False)):
        return None
    x = getattr(result, "x", None)
    if x is None:
        return None
    z = np.asarray(x, dtype=float).reshape(-1)
    if z.size != n or np.any(~np.isfinite(z)):
        return None
    return z


def _independent_interval_robust_primal(
    real_linprog, c, A, b, lower, upper, bounds, prototype, *, margin_scale=0.0
):
    """Construct a conservative flat-interval robust LP *proposal*.

    Each uncertain product A_ij z_j is epigraphed by a temporary t_ij with
    t_ij >= a_lo z_j and t_ij >= a_hi z_j.  This deliberately discards shared
    q-dependency, but only to generate a useful fixed primal candidate.  AutoIA
    subsequently verifies that candidate against the original coupled values.
    """
    n = len(c)
    m = A.shape[0]
    if m == 0:
        cn = np.asarray([_representative(v, prototype) for v in c], dtype=float)
        return _solve_proposal(real_linprog, cn, None, None, bounds)

    # Variables are [z_0..z_{n-1}, t_00..t_{m-1,n-1}].
    nt = m * n
    N = n + nt
    cext = np.zeros(N, dtype=float)
    cext[:n] = np.asarray([_representative(v, prototype) for v in c], dtype=float)
    rows = []
    rhs = []

    def tidx(i, j):
        return n + i * n + j

    for i in range(m):
        bI = _value_interval(b[i], prototype)
        # Optional numerical inward margin.  This is proposal-only; proof never
        # trusts the robust LP or this heuristic margin.
        margin = float(margin_scale) * max(1.0, abs(float(bI.lo)))
        for j in range(n):
            aI = _value_interval(A[i, j], prototype)
            for a in (float(aI.lo), float(aI.hi)):
                row = np.zeros(N, dtype=float)
                row[j] = a
                row[tidx(i, j)] = -1.0
                rows.append(row)
                rhs.append(0.0)
        row = np.zeros(N, dtype=float)
        for j in range(n):
            row[tidx(i, j)] = 1.0
        rows.append(row)
        rhs.append(float(bI.lo) - margin)

    ext_bounds = list(bounds) + [(None, None)] * nt
    try:
        return real_linprog(
            cext,
            A_ub=np.asarray(rows, dtype=float),
            b_ub=np.asarray(rhs, dtype=float),
            bounds=ext_bounds,
            method="highs",
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Result proxy and active execution context
# ---------------------------------------------------------------------------


class CertifiedLinprogResult:
    """Read-compatible SciPy-like result with certified ``fun`` only."""

    __slots__ = ("_nominal", "fun", "autoia_certificate")

    def __init__(self, nominal, fun, certificate):
        self._nominal = nominal
        self.fun = fun
        self.autoia_certificate = certificate

    @property
    def success(self):
        return True

    @property
    def status(self):
        return 0

    @property
    def message(self):
        if not bool(self.autoia_certificate.get("value_upper_certified", True)):
            return (
                "Optimization value lower bound certified by AutoIA; no universal "
                "primal upper witness was proved; HiGHS was proposal-only."
            )
        return "Optimization value certified by AutoIA; HiGHS was proposal-only."

    @property
    def x(self):
        raise NotImplementedError(
            "AutoIA validated scipy.optimize.linprog currently certifies res.fun only; "
            "the parametric optimizer res.x is not yet a certified semantic output."
        )

    def __getattr__(self, name):
        return getattr(self._nominal, name)


class LinearProgramContext:
    """Context-local rigorous semantics for scipy.optimize.linprog."""

    def __init__(self, prototype: ValidatedValue, physical_box=None):
        self.prototype = prototype
        self.physical_box = None if physical_box is None else tuple(physical_box)
        self.certificates: list[dict[str, Any]] = []

    def traced_linprog(self, real_linprog, c, A_ub=None, b_ub=None,
                       A_eq=None, b_eq=None, bounds=(0, None), method="highs",
                       callback=None, options=None, x0=None, integrality=None):
        if integrality is not None:
            raise NotImplementedError(
                "AutoIA validated scipy.optimize.linprog does not support mixed-integer integrality semantics."
            )
        if callback is not None:
            raise NotImplementedError(
                "AutoIA validated scipy.optimize.linprog does not support callback semantics."
            )
        if x0 is not None:
            raise NotImplementedError(
                "AutoIA validated scipy.optimize.linprog does not currently use x0 semantics."
            )

        c_obj = _object_vector(c, name="c")
        n = int(c_obj.size)
        if n < 1:
            raise ValueError("linprog objective c must contain at least one variable.")

        if A_ub is None:
            A_ub_obj = np.empty((0, n), dtype=object)
            b_ub_obj = np.empty(0, dtype=object)
            if b_ub is not None:
                raise ValueError("A_ub is required when b_ub is supplied.")
        else:
            A_ub_obj = _object_matrix(A_ub, n, name="A_ub")
            if b_ub is None:
                raise ValueError("b_ub is required when A_ub is supplied.")
            b_ub_obj = _object_vector(b_ub, A_ub_obj.shape[0], name="b_ub")

        if A_eq is None:
            A_eq_obj = np.empty((0, n), dtype=object)
            b_eq_obj = np.empty(0, dtype=object)
            if b_eq is not None:
                raise ValueError("A_eq is required when b_eq is supplied.")
        else:
            A_eq_obj = _object_matrix(A_eq, n, name="A_eq")
            if b_eq is None:
                raise ValueError("b_eq is required when A_eq is supplied.")
            b_eq_obj = _object_vector(b_eq, A_eq_obj.shape[0], name="b_eq")

        # Exact semantic reduction of equalities to paired inequalities.  This
        # retains the original ValidatedValue dependency in both copies and lets
        # the existing weak-duality/primal-verification machinery remain the
        # single trusted implementation.
        if A_eq_obj.shape[0]:
            A_obj = np.concatenate((A_ub_obj, A_eq_obj, -A_eq_obj), axis=0)
            b_obj = np.concatenate((b_ub_obj, b_eq_obj, -b_eq_obj), axis=0)
        else:
            A_obj = A_ub_obj
            b_obj = b_ub_obj

        lower, upper, clean_bounds = _normalize_bounds(bounds, n)
        m_ub = int(A_ub_obj.shape[0])
        m_eq = int(A_eq_obj.shape[0])
        m = int(A_obj.shape[0])

        # Nominal center solve is diagnostic/proposal only.  We intentionally
        # use the user's requested method here for familiarity, but all other
        # proposal solves use HiGHS because marginals are needed when available.
        c0 = np.asarray([_representative(v, self.prototype) for v in c_obj], dtype=float)
        if m_ub:
            A0_ub = np.empty(A_ub_obj.shape, dtype=float)
            for idx in np.ndindex(A_ub_obj.shape):
                A0_ub[idx] = _representative(A_ub_obj[idx], self.prototype)
            b0_ub = np.asarray([_representative(v, self.prototype) for v in b_ub_obj], dtype=float)
        else:
            A0_ub = None
            b0_ub = None
        if m_eq:
            A0_eq = np.empty(A_eq_obj.shape, dtype=float)
            for idx in np.ndindex(A_eq_obj.shape):
                A0_eq[idx] = _representative(A_eq_obj[idx], self.prototype)
            b0_eq = np.asarray([_representative(v, self.prototype) for v in b_eq_obj], dtype=float)
        else:
            A0_eq = None
            b0_eq = None
        try:
            nominal = real_linprog(
                c0, A_ub=A0_ub, b_ub=b0_ub, A_eq=A0_eq, b_eq=b0_eq, bounds=clean_bounds,
                method=method, callback=None, options=options, x0=None,
                integrality=None,
            )
        except TypeError:
            # Older SciPy signatures do not expose every modern keyword.
            nominal = real_linprog(
                c0, A_ub=A0_ub, b_ub=b0_ub, A_eq=A0_eq, b_eq=b0_eq, bounds=clean_bounds,
                method=method, options=options,
            )
        if not bool(getattr(nominal, "success", False)):
            raise RuntimeError(
                "Validated linprog nominal proposal is not a finite optimal LP; "
                f"HiGHS/SciPy status={getattr(nominal, 'status', None)!r}: "
                f"{getattr(nominal, 'message', '')}"
            )

        # Proposal scenarios.  The same fraction is used for every coefficient
        # interval.  These scenarios are *not* trusted and need not correspond to
        # a realizable q; they merely generate useful sign-feasible dual/primal
        # witnesses.  Seven fractions deliberately include thirds, which is
        # helpful around common active-set transitions without an exponential
        # corner sweep.
        fractions = (0.0, 1.0/6.0, 1.0/3.0, 0.5, 2.0/3.0, 5.0/6.0, 1.0)
        proposal_results = []
        for f in fractions:
            cn, An, bn = _numeric_data_at_fraction(c_obj, A_obj, b_obj, self.prototype, f)
            r = _solve_proposal(real_linprog, cn, An, bn, clean_bounds)
            if r is not None:
                proposal_results.append(r)
        proposal_results.append(nominal)

        # ---- lower bound: weak-duality portfolio -------------------------
        dual_vectors = []
        for r in proposal_results:
            y = _extract_dual(r, m)
            if y is not None:
                dual_vectors.append(y)
        # y=0 is always sign-feasible and can be useful for bounded variables.
        dual_vectors.append(np.zeros(m, dtype=float))
        dual_vectors = _dedupe_vectors(dual_vectors, tol=0.0)

        lower_exprs = []
        lower_records = []
        for y in dual_vectors:
            try:
                expr = _dual_lower_expression(
                    c_obj, A_obj, b_obj, lower, upper, y, self.prototype, self.physical_box
                )
            except _InvalidDualCandidate:
                continue
            LI, _ = expr.range()
            if np.isfinite(LI.lo):
                lower_exprs.append(expr)
                lower_records.append((np.asarray(y, dtype=float).copy(), float(LI.lo), expr))

        if not lower_exprs:
            raise RuntimeError(
                "Validated linprog could not construct any finite weak-duality lower witness "
                "over this box. Subdivide the physical box or use finite variable bounds."
            )

        # Every candidate is globally valid on the current physical box.  V1
        # chooses the candidate with the largest certified global lower endpoint.
        # We intentionally do not switch dual witnesses as a function of q here:
        # that would be a parametric dual policy and can require active-set
        # subdivision.  Caller-controlled box subdivision remains available.
        selected_lower_record = max(lower_records, key=lambda t: t[1])
        best_lower = selected_lower_record[2]
        best_lower_I, _ = best_lower.range()
        lower_bound = float(best_lower_I.lo)
        fixed_dual_lower_bound = float(lower_bound)

        # ---- optional lifted McCormick/RLT lower portfolio ---------------
        # The lower endpoint of a parametric LP is itself a global optimization
        # problem over the physical inputs q and the LP variables z.  When the
        # coefficient enclosures retain polynomial dependence, build a shared
        # lifted outer relaxation of products such as q_i*z_j, certify that
        # constant LP independently, and keep the stronger lower theorem.
        # This is optional tightening only: unsupported or too-expensive lifts
        # leave the fixed-dual result above unchanged.
        rlt_cert, rlt_diag = try_rlt_lower_bound(
            real_linprog, c_obj, A_obj, b_obj, lower, upper, self.prototype,
            baseline_lower=fixed_dual_lower_bound,
            max_obbt_rounds=2, max_obbt_variables=8,
        )
        rlt_lower_bound = None if rlt_cert is None else float(rlt_cert.lower)
        if rlt_lower_bound is not None and np.isfinite(rlt_lower_bound) and rlt_lower_bound > lower_bound:
            lower_bound = rlt_lower_bound
            selected_lower_bound_method = "lifted-mccormick-rlt"
        else:
            selected_lower_bound_method = "fixed-dual"

        # ---- upper bound: universally verified primal portfolio ----------
        primal_vectors = []
        for r in proposal_results:
            z = _extract_primal(r, n)
            if z is not None:
                primal_vectors.append(z)

        # Conservative independent-interval robust proposals, first at the
        # exact flat interval boundary and then with tiny inward margins to avoid
        # proposal points that sit on a numerically awkward active face.
        for margin in (0.0, 1e-12, 1e-10, 1e-8):
            rr = _independent_interval_robust_primal(
                real_linprog, c_obj, A_obj, b_obj, lower, upper,
                clean_bounds, self.prototype, margin_scale=margin,
            )
            if rr is not None:
                zext = _extract_primal(rr, n + m*n)
                if zext is not None:
                    primal_vectors.append(zext[:n])

        # Simple anchors often give a universally feasible witness essentially
        # for free.  They are proposal-only and are independently checked below.
        zero = np.zeros(n, dtype=float)
        if np.all(lower <= zero) and np.all(zero <= upper):
            primal_vectors.append(zero)
        if np.all(np.isfinite(lower)):
            primal_vectors.append(lower.copy())
        if np.all(np.isfinite(upper)):
            primal_vectors.append(upper.copy())
        primal_vectors = _dedupe_vectors(primal_vectors, tol=0.0)

        verified = []
        for z in primal_vectors:
            rec = _verify_primal_candidate(
                c_obj, A_obj, b_obj, lower, upper, z, self.prototype, self.physical_box
            )
            if rec is not None:
                verified.append(rec)

        # The primal and dual arguments are logically independent.  Failure to
        # find a *fixed* primal point that is feasible for every q in the box does
        # not invalidate the weak-duality lower certificate already proved above.
        # In that case return a rigorous one-sided enclosure [L,+inf] rather than
        # discarding useful information by raising.  This distinction is
        # particularly important when the feasible optimizer itself moves with q.
        if verified:
            best_primal = min(verified, key=lambda r: r["objective_upper"])
            upper_bound = float(best_primal["objective_upper"])
            value_upper_certified = True
        else:
            best_primal = None
            upper_bound = np.inf
            value_upper_certified = False

        if not np.isfinite(lower_bound):
            raise RuntimeError("Validated linprog produced a nonfinite weak-duality lower certificate.")
        if value_upper_certified and not np.isfinite(upper_bound):
            raise RuntimeError("Validated linprog produced a nonfinite certified primal upper bound.")
        if lower_bound > upper_bound:
            raise RuntimeError(
                "Validated linprog internal soundness error: certified lower bound exceeds upper bound."
            )

        value_interval = Interval(lower_bound, upper_bound)
        fun = ValidatedValue(
            IntervalEnclosure(
                value_interval,
                self.prototype.h,
                self.prototype.order,
                self.prototype.backend,
                self.prototype.range_method,
            ),
            natural_interval=value_interval,
            _diag_add_source=False,
        )

        # Identify a diagnostic dual candidate whose current global lower
        # endpoint is largest.  This is not necessarily the pointwise-selected
        # witness on every path; ``dual_candidate_count`` makes that explicit.
        selected_y = selected_lower_record[0] if lower_records else np.empty(m)
        cert = {
            "operation": "scipy.optimize.linprog",
            "semantic_mode": "certified-optimal-value",
            "certificate_completeness": "two-sided" if value_upper_certified else "lower-only",
            "value_lower_certified": True,
            "value_upper_certified": bool(value_upper_certified),
            "optimal_value_lower": lower_bound,
            "optimal_value_upper": upper_bound,
            "certified_value_width": upper_bound - lower_bound,
            "selected_dual_y": selected_y,
            "dual_candidate_count": len(lower_records),
            "fixed_dual_lower_bound": fixed_dual_lower_bound,
            "rlt_lower_bound": rlt_lower_bound,
            "selected_lower_bound_method": selected_lower_bound_method,
            "rlt_status": rlt_diag.get("status"),
            "rlt_reason": rlt_diag.get("reason"),
            "rlt_initial_lower": rlt_diag.get("initial_lower"),
            "rlt_final_lower": rlt_diag.get("final_lower"),
            "rlt_obbt_rounds": int(rlt_diag.get("obbt_rounds", 0)),
            "rlt_lifted_variable_count": rlt_diag.get("lifted_variable_count"),
            "rlt_lifted_row_count": rlt_diag.get("lifted_row_count"),
            "rlt_monomial_count": rlt_diag.get("monomial_count"),
            "rlt_product_count": rlt_diag.get("product_count"),
            "rlt_remainder_variable_count": rlt_diag.get("remainder_variable_count"),
            "rlt_relevant_z_count": rlt_diag.get("relevant_z_count"),
            "inequality_constraint_count": m_ub,
            "equality_constraint_count": m_eq,
            "equality_semantic": "paired-inequalities" if m_eq else None,
            "selected_primal_z": (
                np.asarray(best_primal["z"], dtype=float).copy()
                if best_primal is not None else None
            ),
            "primal_residual_upper": (
                np.asarray(best_primal["residual_upper"], dtype=float).copy()
                if best_primal is not None else None
            ),
            # None means "not proved", not "proved infeasible".
            "primal_feasible_for_all_q": True if best_primal is not None else None,
            "universal_primal_witness_found": bool(best_primal is not None),
            "primal_candidate_count": len(primal_vectors),
            "verified_primal_candidate_count": len(verified),
            "proposal_method": "SciPy/HiGHS; proposal only",
            "proposal_success": bool(getattr(nominal, "success", False)),
            "machine_rigorous": bool(self.prototype.backend.machine_rigorous),
            "certifies": "res.fun only",
        }
        self.certificates.append(cert)
        return CertifiedLinprogResult(nominal, fun, cert)


_ACTIVE_LINEAR_PROGRAM_CONTEXT: ContextVar[LinearProgramContext | None] = ContextVar(
    "autoia_active_linear_program_context", default=None
)


class LinearProgramExecutionContext:
    def __init__(self, context: LinearProgramContext):
        self.context = context
        self._token = None

    def __enter__(self):
        self._token = _ACTIVE_LINEAR_PROGRAM_CONTEXT.set(self.context)
        return self.context

    def __exit__(self, exc_type, exc, tb):
        _ACTIVE_LINEAR_PROGRAM_CONTEXT.reset(self._token)
        return False


def active_linear_program_context() -> LinearProgramContext | None:
    return _ACTIVE_LINEAR_PROGRAM_CONTEXT.get()


def prepared_traced_linprog(real_linprog, *args, **kwargs):
    ctx = active_linear_program_context()
    if ctx is None:
        return real_linprog(*args, **kwargs)
    return ctx.traced_linprog(real_linprog, *args, **kwargs)


__all__ = [
    "CertifiedLinprogResult",
    "LinearProgramContext",
    "LinearProgramExecutionContext",
    "active_linear_program_context",
    "prepared_traced_linprog",
]
