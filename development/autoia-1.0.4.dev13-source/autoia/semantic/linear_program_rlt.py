"""Lifted McCormick/RLT lower bounds for parametric linear programs.

This module is an internal certificate-strengthening portfolio member used by
``autoia.semantic.linear_program``.  It does not change the user's SciPy model.

For the lower endpoint of a parametric minimization LP, the physical parameters
can be viewed as additional optimization variables over the current AutoIA box.
Polynomial coefficient dependence then creates products between physical
monomials and LP decision variables.  The routines below construct a constant-
data outer relaxation by

* sharing lifted variables for physical monomials,
* sharing lifted variables for each monomial/decision product,
* adding rigorous McCormick envelope rows,
* representing unstructured polynomial remainders by independent bounded
  auxiliary variables, and
* independently certifying the constant relaxation's optimal value by weak
  duality.  HiGHS remains proposal machinery only.

Optional optimization-based bound tightening (OBBT) uses *certified* extrema of
original decision variables over the current relaxation before rebuilding the
McCormick envelopes.  Every tightened bound therefore remains an outer bound
for the original feasible set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..enclosures import ValidatedValue
from ..polynomial import monomial_range_symmetric, total_degree, zero_alpha, basis_alpha
from ..trusted import Interval, point_interval, interval_neg


class RLTUnavailable(Exception):
    """The optional lifted relaxation is not applicable to this LP/box."""


@dataclass
class ConstantLPCertificate:
    lower: float
    dual_y: np.ndarray | None
    proposal_success: bool


@dataclass
class RLTRelaxation:
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    z_indices: tuple[int, ...]
    relevant_z: tuple[int, ...]
    monomial_count: int
    product_count: int
    remainder_variable_count: int
    variable_names: tuple[str, ...]

    @property
    def bounds_for_scipy(self):
        out = []
        for lo, hi in zip(self.lower, self.upper):
            out.append((None if not np.isfinite(lo) else float(lo),
                        None if not np.isfinite(hi) else float(hi)))
        return out


# ---------------------------------------------------------------------------
# Constant-LP certification
# ---------------------------------------------------------------------------


def _support_lower_constant(D: Interval, lo: float, hi: float, backend) -> float | None:
    """Rigorous lower bound on ``d*x`` for d in D and x in [lo,hi]."""
    lfin = np.isfinite(lo)
    ufin = np.isfinite(hi)
    if lfin and ufin:
        return float(backend.mul(D, Interval(float(lo), float(hi))).lo)
    if lfin and not ufin:
        if D.lo >= 0.0:
            return float(backend.mul(D, point_interval(float(lo))).lo)
        return None
    if not lfin and ufin:
        if D.hi <= 0.0:
            return float(backend.mul(D, point_interval(float(hi))).lo)
        return None
    if D.lo == 0.0 and D.hi == 0.0:
        return 0.0
    return None


def _constant_dual_lower(c, A, b, lower, upper, y, backend) -> float | None:
    c = np.asarray(c, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    m, n = A.shape
    if c.size != n or b.size != m or yy.size != m:
        return None
    if np.any(~np.isfinite(c)) or np.any(~np.isfinite(A)) or np.any(~np.isfinite(b)):
        return None
    if np.any(~np.isfinite(yy)):
        return None
    yy = np.minimum(yy, 0.0)

    total = Interval(0.0, 0.0)
    for i in range(m):
        if yy[i] != 0.0:
            total = backend.add(total, backend.mul(point_interval(float(b[i])),
                                                   point_interval(float(yy[i]))))

    for j in range(n):
        d = point_interval(float(c[j]))
        for i in range(m):
            if yy[i] != 0.0 and A[i, j] != 0.0:
                term = backend.mul(point_interval(float(A[i, j])),
                                   point_interval(float(yy[i])))
                d = backend.sub(d, term)
        s = _support_lower_constant(d, float(lower[j]), float(upper[j]), backend)
        if s is None or not np.isfinite(s):
            return None
        total = backend.add(total, point_interval(float(s)))
    return float(total.lo)


def certify_constant_lp_lower(real_linprog, c, A, b, lower, upper, backend) -> ConstantLPCertificate | None:
    """Certify a lower bound for a constant-data LP using a fixed dual witness.

    SciPy/HiGHS supplies candidate dual marginals.  The returned lower endpoint
    is independently recomputed with the trusted binary64 interval backend.
    """
    c = np.asarray(c, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    if A.ndim == 1 and A.size == 0:
        A = A.reshape(0, c.size)
    b = np.asarray(b, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    if A.ndim != 2 or A.shape[1] != c.size or A.shape[0] != b.size:
        return None

    bounds = [
        (None if not np.isfinite(lo) else float(lo),
         None if not np.isfinite(hi) else float(hi))
        for lo, hi in zip(lower, upper)
    ]
    try:
        r = real_linprog(
            c,
            A_ub=None if A.shape[0] == 0 else A,
            b_ub=None if A.shape[0] == 0 else b,
            bounds=bounds,
            method="highs",
        )
    except Exception:
        r = None

    candidates = []
    if r is not None and bool(getattr(r, "success", False)):
        if A.shape[0] == 0:
            candidates.append(np.empty(0, dtype=float))
        else:
            marg = getattr(getattr(r, "ineqlin", None), "marginals", None)
            if marg is not None:
                yy = np.asarray(marg, dtype=float).reshape(-1)
                if yy.size == A.shape[0] and np.all(np.isfinite(yy)):
                    candidates.append(np.minimum(yy, 0.0))
    # y=0 is a universally valid fallback and gives a finite bound whenever the
    # variable bounds alone control the objective.
    candidates.append(np.zeros(A.shape[0], dtype=float))

    best = None
    best_y = None
    for y in candidates:
        L = _constant_dual_lower(c, A, b, lower, upper, y, backend)
        if L is not None and np.isfinite(L) and (best is None or L > best):
            best = float(L)
            best_y = np.asarray(y, dtype=float).copy()
    if best is None:
        return None
    return ConstantLPCertificate(
        lower=float(best),
        dual_y=best_y,
        proposal_success=bool(r is not None and getattr(r, "success", False)),
    )


# ---------------------------------------------------------------------------
# Polynomial extraction and lifted relaxation construction
# ---------------------------------------------------------------------------


def _as_value(x: Any, prototype: ValidatedValue) -> ValidatedValue:
    if isinstance(x, ValidatedValue):
        return x
    return prototype._constant_like(float(x))


def _poly_parts(x: Any, prototype: ValidatedValue):
    """Return stored point polynomial and rigorous scalar remainder."""
    v = _as_value(x, prototype)
    rep = v.to_polynomial()
    coeff = {tuple(int(k) for k in alpha): float(a) for alpha, a in rep.coeff.items()}
    rem = rep.remainder
    if not (np.isfinite(rem.lo) and np.isfinite(rem.hi)):
        raise RLTUnavailable("nonfinite polynomial remainder")
    return coeff, rem


def _interval_is_zero(I: Interval) -> bool:
    return float(I.lo) == 0.0 and float(I.hi) == 0.0


class _LiftBuilder:
    def __init__(self, prototype: ValidatedValue, n_z: int, z_lower, z_upper):
        self.prototype = prototype
        self.backend = prototype.backend
        self.h = np.asarray(prototype.h, dtype=float)
        self.mq = len(self.h)
        self.n_z = int(n_z)
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.names: list[str] = []
        self.rows: list[dict[int, float]] = []
        self.rhs: list[float] = []
        self.obj: dict[int, float] = {}
        self.monomial_vars: dict[tuple[int, ...], int] = {}
        self.product_vars: dict[tuple[tuple[int, ...], int], int] = {}
        self.relevant_z: set[int] = set()
        self.product_count = 0
        self.remainder_variable_count = 0

        # Centered physical coordinates q_i in [-h_i,h_i].
        self.q_indices = []
        for i, hi in enumerate(self.h):
            self.q_indices.append(self._add_var(-float(hi), float(hi), f"q[{i}]"))
            self.monomial_vars[basis_alpha(self.mq, i)] = self.q_indices[-1]

        self.z_indices = []
        for j, (lo, up) in enumerate(zip(z_lower, z_upper)):
            self.z_indices.append(self._add_var(float(lo), float(up), f"z[{j}]"))

    def _add_var(self, lo: float, hi: float, name: str) -> int:
        if np.isnan(lo) or np.isnan(hi) or lo > hi:
            raise RLTUnavailable(f"invalid lifted variable bounds for {name}")
        idx = len(self.lower)
        self.lower.append(float(lo))
        self.upper.append(float(hi))
        self.names.append(str(name))
        return idx

    def _var_interval(self, idx: int) -> Interval:
        lo = float(self.lower[idx]); hi = float(self.upper[idx])
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise RLTUnavailable("McCormick lifting requires finite factor bounds")
        return Interval(lo, hi)

    def _add_obj(self, idx: int, coeff: float):
        a = float(coeff)
        if a == 0.0:
            return
        old = self.obj.get(idx)
        if old is None:
            self.obj[idx] = a
            return
        # Objective collisions are rare, but aggregate them through the trusted
        # backend and retain a point representative plus a conservative constant
        # correction would be required for a non-point sum.  Current construction
        # uses distinct lifted variables, so treat a non-exact collision as
        # unavailable rather than silently changing the objective.
        S = self.backend.add(point_interval(float(old)), point_interval(a))
        s = float(old + a)
        if not (S.lo == S.hi == s):
            raise RLTUnavailable("non-exact objective coefficient collision")
        if s == 0.0:
            self.obj.pop(idx, None)
        else:
            self.obj[idx] = s

    def _safe_row(self, terms, rhs_interval: Interval):
        """Add a point-coefficient row known to be weaker than interval row data.

        ``terms`` is an iterable of ``(variable_index, Interval coefficient)``.
        Coefficient aggregation is trusted.  If aggregation widens a coefficient,
        the RHS is relaxed by the worst possible coefficient-error contribution
        over that variable's bounds.
        """
        agg: dict[int, Interval] = {}
        for idx, CI in terms:
            idx = int(idx)
            if idx in agg:
                agg[idx] = self.backend.add(agg[idx], CI)
            else:
                agg[idx] = CI

        row: dict[int, float] = {}
        rhs = Interval(float(rhs_interval.lo), float(rhs_interval.hi))
        for idx, CI in agg.items():
            if not (np.isfinite(CI.lo) and np.isfinite(CI.hi)):
                raise RLTUnavailable("nonfinite lifted row coefficient")
            if CI.lo == CI.hi:
                a = float(CI.lo)
            else:
                a = float(0.5 * float(CI.lo) + 0.5 * float(CI.hi))
                a = float(min(max(a, float(CI.lo)), float(CI.hi)))
            E = self.backend.sub(point_interval(a), CI)  # a - exact coefficient
            if not _interval_is_zero(E):
                X = self._var_interval(idx)
                err = self.backend.mul(E, X)
                rhs = self.backend.add(rhs, point_interval(float(err.hi)))
            if a != 0.0:
                row[idx] = a
        if not np.isfinite(rhs.hi):
            raise RLTUnavailable("could not form finite safe lifted row RHS")
        self.rows.append(row)
        self.rhs.append(float(rhs.hi))

    def _add_exact_row(self, row: dict[int, float], rhs: float):
        terms = [(idx, point_interval(float(a))) for idx, a in row.items() if float(a) != 0.0]
        self._safe_row(terms, point_interval(float(rhs)))

    def _add_mccormick(self, xidx: int, zidx: int, yidx: int):
        X = self._var_interval(xidx)
        Z = self._var_interval(zidx)
        lx, ux = float(X.lo), float(X.hi)
        lz, uz = float(Z.lo), float(Z.hi)
        B = self.backend

        # -y + lx*z + lz*x <= lx*lz
        self._safe_row(
            [(yidx, point_interval(-1.0)),
             (zidx, point_interval(lx)),
             (xidx, point_interval(lz))],
            B.mul(point_interval(lx), point_interval(lz)),
        )
        # -y + ux*z + uz*x <= ux*uz
        self._safe_row(
            [(yidx, point_interval(-1.0)),
             (zidx, point_interval(ux)),
             (xidx, point_interval(uz))],
            B.mul(point_interval(ux), point_interval(uz)),
        )
        # y - ux*z - lz*x <= -ux*lz
        self._safe_row(
            [(yidx, point_interval(1.0)),
             (zidx, point_interval(-ux)),
             (xidx, point_interval(-lz))],
            interval_neg(B.mul(point_interval(ux), point_interval(lz))),
        )
        # y - lx*z - uz*x <= -lx*uz
        self._safe_row(
            [(yidx, point_interval(1.0)),
             (zidx, point_interval(-lx)),
             (xidx, point_interval(-uz))],
            interval_neg(B.mul(point_interval(lx), point_interval(uz))),
        )
        self.product_count += 1

    def ensure_monomial(self, alpha: tuple[int, ...]) -> int:
        alpha = tuple(int(k) for k in alpha)
        deg = total_degree(alpha)
        if deg <= 0:
            raise ValueError("constant monomial has no lifted variable")
        old = self.monomial_vars.get(alpha)
        if old is not None:
            return old
        # Recursive shared DAG: m_alpha = m_beta * q_i.
        i = next(k for k, a in enumerate(alpha) if a > 0)
        beta = list(alpha); beta[i] -= 1; beta = tuple(beta)
        if total_degree(beta) == 0:
            xidx = self.q_indices[i]
        else:
            xidx = self.ensure_monomial(beta)
        qidx = self.q_indices[i]
        MI = monomial_range_symmetric(alpha, self.h, self.backend)
        if not (np.isfinite(MI.lo) and np.isfinite(MI.hi)):
            raise RLTUnavailable("nonfinite monomial bounds")
        midx = self._add_var(float(MI.lo), float(MI.hi), f"m{alpha}")
        self.monomial_vars[alpha] = midx
        self._add_mccormick(xidx, qidx, midx)
        return midx

    def monomial_times_z(self, alpha: tuple[int, ...], j: int) -> int:
        alpha = tuple(int(k) for k in alpha)
        if total_degree(alpha) == 0:
            return self.z_indices[int(j)]
        key = (alpha, int(j))
        old = self.product_vars.get(key)
        if old is not None:
            return old
        zidx = self.z_indices[int(j)]
        # Product lifting is useful only with finite decision-variable bounds.
        self._var_interval(zidx)
        midx = self.ensure_monomial(alpha)
        YI = self.backend.mul(self._var_interval(midx), self._var_interval(zidx))
        yidx = self._add_var(float(YI.lo), float(YI.hi), f"m{alpha}*z[{j}]")
        self.product_vars[key] = yidx
        self.relevant_z.add(int(j))
        self._add_mccormick(midx, zidx, yidx)
        return yidx

    def remainder_var(self, I: Interval, name: str) -> int:
        if not (np.isfinite(I.lo) and np.isfinite(I.hi)):
            raise RLTUnavailable("nonfinite remainder interval")
        idx = self._add_var(float(I.lo), float(I.hi), name)
        self.remainder_variable_count += 1
        return idx

    def remainder_times_z(self, I: Interval, j: int, name: str) -> int:
        zidx = self.z_indices[int(j)]
        self._var_interval(zidx)
        ridx = self.remainder_var(I, name)
        TI = self.backend.mul(self._var_interval(ridx), self._var_interval(zidx))
        tidx = self._add_var(float(TI.lo), float(TI.hi), name + f"*z[{j}]")
        self.relevant_z.add(int(j))
        self._add_mccormick(ridx, zidx, tidx)
        return tidx

    def add_objective_coefficient(self, cj, j: int):
        coeff, rem = _poly_parts(cj, self.prototype)
        for alpha, a in coeff.items():
            if a == 0.0:
                continue
            idx = self.monomial_times_z(alpha, j)
            self._add_obj(idx, a)
        if not _interval_is_zero(rem):
            tidx = self.remainder_times_z(rem, j, f"rc[{j}]")
            self._add_obj(tidx, 1.0)

    def add_constraint(self, Arow, bi, row_index: int):
        terms = []
        rhs = 0.0

        def add(idx, a):
            a = float(a)
            if a != 0.0:
                terms.append((int(idx), point_interval(a)))

        for j, aij in enumerate(Arow):
            coeff, rem = _poly_parts(aij, self.prototype)
            for alpha, a in coeff.items():
                if a == 0.0:
                    continue
                idx = self.monomial_times_z(alpha, j)
                add(idx, a)
            if not _interval_is_zero(rem):
                tidx = self.remainder_times_z(rem, j, f"rA[{row_index},{j}]")
                add(tidx, 1.0)

        bcoeff, brem = _poly_parts(bi, self.prototype)
        z0 = zero_alpha(self.mq)
        for alpha, a in bcoeff.items():
            if alpha == z0:
                # A z - b <= 0  ->  nonconstant terms <= b0.
                rhs = float(a)
            elif a != 0.0:
                midx = self.ensure_monomial(alpha)
                add(midx, -float(a))
        if not _interval_is_zero(brem):
            ridx = self.remainder_var(brem, f"rb[{row_index}]")
            add(ridx, -1.0)

        self._safe_row(terms, point_interval(rhs))

    def finish(self) -> RLTRelaxation:
        N = len(self.lower)
        c = np.zeros(N, dtype=float)
        for idx, a in self.obj.items():
            c[int(idx)] = float(a)
        A = np.zeros((len(self.rows), N), dtype=float)
        for i, row in enumerate(self.rows):
            for idx, a in row.items():
                A[i, int(idx)] = float(a)
        b = np.asarray(self.rhs, dtype=float)
        return RLTRelaxation(
            c=c,
            A=A,
            b=b,
            lower=np.asarray(self.lower, dtype=float),
            upper=np.asarray(self.upper, dtype=float),
            z_indices=tuple(int(i) for i in self.z_indices),
            relevant_z=tuple(sorted(int(j) for j in self.relevant_z)),
            monomial_count=sum(1 for a in self.monomial_vars if total_degree(a) >= 2),
            product_count=int(self.product_count),
            remainder_variable_count=int(self.remainder_variable_count),
            variable_names=tuple(self.names),
        )


def build_rlt_relaxation(c, A, b, lower, upper, prototype, z_lower=None, z_upper=None) -> RLTRelaxation:
    c = np.asarray(c, dtype=object).reshape(-1)
    A = np.asarray(A, dtype=object)
    b = np.asarray(b, dtype=object).reshape(-1)
    n = c.size
    if A.ndim != 2 or A.shape[1] != n or A.shape[0] != b.size:
        raise RLTUnavailable("LP dimensions do not match")
    zl = np.asarray(lower if z_lower is None else z_lower, dtype=float).reshape(-1)
    zu = np.asarray(upper if z_upper is None else z_upper, dtype=float).reshape(-1)
    if zl.size != n or zu.size != n:
        raise RLTUnavailable("decision-bound dimension mismatch")
    B = _LiftBuilder(prototype, n, zl, zu)
    for j, cj in enumerate(c):
        B.add_objective_coefficient(cj, j)
    for i in range(A.shape[0]):
        B.add_constraint(A[i, :], b[i], i)
    return B.finish()


# ---------------------------------------------------------------------------
# Portfolio driver with certified OBBT
# ---------------------------------------------------------------------------


def _tighten_z_bounds(real_linprog, relaxation: RLTRelaxation, prototype, z_lower, z_upper):
    """One certified OBBT sweep over decision variables used in lifted products."""
    zl = np.asarray(z_lower, dtype=float).copy()
    zu = np.asarray(z_upper, dtype=float).copy()
    records = []
    changed = False
    for j in relaxation.relevant_z:
        vidx = relaxation.z_indices[j]
        old_lo, old_up = float(zl[j]), float(zu[j])
        obj = np.zeros_like(relaxation.c)
        # A zero lower bound is extremely common for LP weights/flows.  Skipping
        # lower-side OBBT there is a cost heuristic only: it cannot weaken rigor,
        # and it avoids a whole auxiliary LP per round in the common nonnegative
        # case.  Nonzero lower bounds still receive a certified tightening solve.
        if old_lo == 0.0:
            lo_cert = None
        else:
            obj[vidx] = 1.0
            lo_cert = certify_constant_lp_lower(
                real_linprog, obj, relaxation.A, relaxation.b,
                relaxation.lower, relaxation.upper, prototype.backend,
            )
            obj[vidx] = 0.0
        # Minimize -z_j; a lower bound L on min(-z_j) implies z_j <= -L.
        obj[vidx] = -1.0
        up_cert = certify_constant_lp_lower(
            real_linprog, obj, relaxation.A, relaxation.b,
            relaxation.lower, relaxation.upper, prototype.backend,
        )
        new_lo, new_up = old_lo, old_up
        if lo_cert is not None and np.isfinite(lo_cert.lower):
            new_lo = max(new_lo, float(lo_cert.lower))
        if up_cert is not None and np.isfinite(up_cert.lower):
            new_up = min(new_up, float(-up_cert.lower))
        # A tiny inversion from outward arithmetic means the optional tightening
        # was inconclusive; never shrink past a proved nonempty interval.
        if new_lo <= new_up:
            zl[j], zu[j] = new_lo, new_up
            if new_lo > old_lo or new_up < old_up:
                changed = True
        records.append({
            "z_index": int(j),
            "old_lower": old_lo,
            "old_upper": old_up,
            "new_lower": float(zl[j]),
            "new_upper": float(zu[j]),
            "lower_certificate": None if lo_cert is None else float(lo_cert.lower),
            "negative_upper_certificate": None if up_cert is None else float(up_cert.lower),
        })
    return zl, zu, changed, records


def try_rlt_lower_bound(
    real_linprog,
    c,
    A,
    b,
    lower,
    upper,
    prototype: ValidatedValue,
    *,
    baseline_lower: float | None = None,
    max_obbt_rounds: int = 2,
    max_obbt_variables: int = 8,
):
    """Return a rigorous lifted McCormick/RLT lower certificate or ``None``.

    The routine is deliberately optional.  Unsupported/expensive structures
    simply leave the existing fixed-dual certificate unchanged.
    """
    diagnostics = {
        "status": "not-used",
        "reason": None,
        "initial_lower": None,
        "final_lower": None,
        "obbt_rounds": 0,
        "obbt_records": [],
    }
    try:
        zl = np.asarray(lower, dtype=float).copy()
        zu = np.asarray(upper, dtype=float).copy()
        R = build_rlt_relaxation(c, A, b, lower, upper, prototype, zl, zu)
        cert = certify_constant_lp_lower(
            real_linprog, R.c, R.A, R.b, R.lower, R.upper, prototype.backend
        )
        if cert is None:
            diagnostics["reason"] = "constant relaxation lower certificate unavailable"
            return None, diagnostics
        diagnostics.update({
            "status": "used",
            "initial_lower": float(cert.lower),
            "final_lower": float(cert.lower),
            "lifted_variable_count": int(len(R.c)),
            "lifted_row_count": int(R.A.shape[0]),
            "monomial_count": int(R.monomial_count),
            "product_count": int(R.product_count),
            "remainder_variable_count": int(R.remainder_variable_count),
            "relevant_z_count": int(len(R.relevant_z)),
            "selected_dual_y": None if cert.dual_y is None else cert.dual_y.copy(),
        })

        # OBBT is cost-capped and proof-optional.  It can only improve the lower
        # relaxation bound; failure of any tightening solve leaves the previous
        # certified RLT result intact.  To avoid multiplying LP proposal work on
        # cases already handled well by the cheaper fixed-dual theorem, automatic
        # OBBT is triggered only when the first lifted relaxation has already
        # demonstrated a genuine improvement over that baseline.
        baseline = None if baseline_lower is None else float(baseline_lower)
        scale = max(1.0, abs(float(cert.lower)), 0.0 if baseline is None else abs(baseline))
        demonstrated_gain = (
            baseline is None
            or float(cert.lower) > baseline + 1.0e-12 * scale
        )
        if not R.relevant_z:
            diagnostics["obbt_skipped_reason"] = "no decision variable participates in a lifted product"
        elif not demonstrated_gain:
            diagnostics["obbt_skipped_reason"] = "initial lifted relaxation did not improve fixed-dual baseline"
        elif len(R.relevant_z) <= int(max_obbt_variables):
            best = cert
            for k in range(int(max_obbt_rounds)):
                zl2, zu2, changed, recs = _tighten_z_bounds(
                    real_linprog, R, prototype, zl, zu
                )
                diagnostics["obbt_records"].append(tuple(recs))
                if not changed:
                    break
                zl, zu = zl2, zu2
                R2 = build_rlt_relaxation(c, A, b, lower, upper, prototype, zl, zu)
                cert2 = certify_constant_lp_lower(
                    real_linprog, R2.c, R2.A, R2.b, R2.lower, R2.upper, prototype.backend
                )
                if cert2 is None:
                    break
                # Both relaxations are rigorous.  Retain the numerically larger
                # certified lower endpoint; the rebuilt one should normally win.
                if cert2.lower >= best.lower:
                    best = cert2
                    R = R2
                diagnostics["obbt_rounds"] = k + 1
                diagnostics["final_lower"] = float(max(diagnostics["final_lower"], cert2.lower))
                diagnostics["lifted_variable_count"] = int(len(R.c))
                diagnostics["lifted_row_count"] = int(R.A.shape[0])
                diagnostics["selected_dual_y"] = None if best.dual_y is None else best.dual_y.copy()
            cert = best
        else:
            diagnostics["obbt_skipped_reason"] = (
                f"{len(R.relevant_z)} lifted decision variables exceeds cap {int(max_obbt_variables)}"
            )
        diagnostics["final_lower"] = float(cert.lower)
        return cert, diagnostics
    except RLTUnavailable as exc:
        diagnostics["reason"] = str(exc)
        return None, diagnostics
    except (FloatingPointError, OverflowError) as exc:
        # A trusted numerical limit can make this optional relaxation unavailable
        # without invalidating the already-established fixed-dual theorem.
        diagnostics["reason"] = f"{type(exc).__name__}: {exc}"
        return None, diagnostics


__all__ = [
    "RLTUnavailable",
    "ConstantLPCertificate",
    "RLTRelaxation",
    "build_rlt_relaxation",
    "certify_constant_lp_lower",
    "try_rlt_lower_bound",
]
