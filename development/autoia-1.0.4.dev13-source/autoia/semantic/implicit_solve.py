"""Validated semantic primitives for common implicit nonlinear solves.

Version-1 target
----------------
* ``scipy.optimize.root_scalar`` with an explicit finite bracket.
* ``scipy.optimize.fsolve`` for square systems with a finite numeric initial guess.

The ordinary SciPy solve is a *proposal* only.  AutoIA independently certifies
an implicit solution using a dependency-bearing affine predictor in the active
uncertain inputs plus a rigorous correction box.  No SciPy iteration or
convergence flag is trusted as a proof.

The scalar bracketed path additionally proves a uniform endpoint sign change
and a uniform nonzero derivative over the supplied bracket (piecewise when
needed), so the certified local solution is the unique root in that bracket for
every physical parameter value in the current AutoIA box.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace
import types
from typing import Any
import inspect

import numpy as np

from ..enclosures import ValidatedValue, IntervalEnclosure, DomainRefinementRequired
from ..trusted import Interval, point_interval


# ---------------------------------------------------------------------------
# Small forward-mode dual used only to obtain derivative expressions through
# the already-supported AutoIA scalar language.  ``value`` and derivative
# entries may themselves be ValidatedValue objects, so derivative bounds retain
# dependence on the physical inputs.
# ---------------------------------------------------------------------------


def _zeros(n):
    return np.zeros(int(n), dtype=object)


def _as_deriv(v, n):
    a = np.asarray(v, dtype=object).reshape(-1)
    if len(a) != int(n):
        raise ValueError("Implicit dual derivative size mismatch.")
    return a


class ImplicitDual:
    __array_priority__ = 3000
    __autoia_semantic_dual__ = True

    def __init__(self, value, derivative):
        self.value = value
        self.derivative = np.asarray(derivative, dtype=object).reshape(-1)

    @property
    def nderiv(self):
        return len(self.derivative)

    def _coerce(self, other):
        if isinstance(other, ImplicitDual):
            if other.nderiv != self.nderiv:
                raise ValueError("Implicit dual derivative dimensions do not match.")
            return other
        return ImplicitDual(other, _zeros(self.nderiv))

    def __add__(self, other):
        other = self._coerce(other)
        return ImplicitDual(self.value + other.value, self.derivative + other.derivative)

    __radd__ = __add__

    def __neg__(self):
        return ImplicitDual(-self.value, -self.derivative)

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        d = np.empty(self.nderiv, dtype=object)
        for i in range(self.nderiv):
            d[i] = self.derivative[i] * other.value + self.value * other.derivative[i]
        return ImplicitDual(self.value * other.value, d)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._coerce(other)
        denom = other.value * other.value
        d = np.empty(self.nderiv, dtype=object)
        for i in range(self.nderiv):
            d[i] = (
                self.derivative[i] * other.value
                - self.value * other.derivative[i]
            ) / denom
        return ImplicitDual(self.value / other.value, d)

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def __pow__(self, power):
        if isinstance(power, (ImplicitDual, ValidatedValue)):
            raise NotImplementedError(
                "AutoIA validated implicit solves currently require constant exponents."
            )
        p = float(power)
        value = self.value ** p
        factor = p * (self.value ** (p - 1.0))
        d = np.empty(self.nderiv, dtype=object)
        for i in range(self.nderiv):
            d[i] = factor * self.derivative[i]
        return ImplicitDual(value, d)

    def __rpow__(self, base):
        # d base^x = base^x log(base) dx, with constant positive base.
        b = float(base)
        if not b > 0.0:
            raise NotImplementedError(
                "Validated implicit dual exponentiation requires a positive constant base."
            )
        value = b ** self.value
        factor = np.log(b) * value
        d = np.empty(self.nderiv, dtype=object)
        for i in range(self.nderiv):
            d[i] = factor * self.derivative[i]
        return ImplicitDual(value, d)

    def __autoia_unary__(self, op: str):
        x = self.value
        if op == "sqrt":
            value = x ** 0.5
            factor = 0.5 / value
        elif op == "exp":
            # Use the same local semantic unary path on the primal value.
            value = x.unary("exp") if isinstance(x, ValidatedValue) else np.exp(x)
            factor = value
        elif op == "log":
            value = x.unary("log") if isinstance(x, ValidatedValue) else np.log(x)
            factor = 1.0 / x
        elif op == "log10":
            if isinstance(x, ValidatedValue):
                value = x.unary("log10")
                # The derivative contains the exact real constant ln(10).
                # Construct it through AutoIA's rigorous elementary semantics
                # rather than treating the host libm value as exact.
                ln10 = x._constant_like(10.0).unary("log")
                factor = 1.0 / (x * ln10)
            else:
                value = np.log10(x)
                factor = 1.0 / (x * np.log(10.0))
        elif op == "sin":
            value = x.unary("sin") if isinstance(x, ValidatedValue) else np.sin(x)
            c = x.unary("cos") if isinstance(x, ValidatedValue) else np.cos(x)
            factor = c
        elif op == "cos":
            value = x.unary("cos") if isinstance(x, ValidatedValue) else np.cos(x)
            s = x.unary("sin") if isinstance(x, ValidatedValue) else np.sin(x)
            factor = -s
        elif op == "log1p":
            value = x.unary("log1p") if isinstance(x, ValidatedValue) else np.log1p(x)
            factor = 1.0 / (1.0 + x)
        elif op == "sinh":
            value = x.unary("sinh") if isinstance(x, ValidatedValue) else np.sinh(x)
            factor = x.unary("cosh") if isinstance(x, ValidatedValue) else np.cosh(x)
        elif op == "cosh":
            value = x.unary("cosh") if isinstance(x, ValidatedValue) else np.cosh(x)
            factor = x.unary("sinh") if isinstance(x, ValidatedValue) else np.sinh(x)
        elif op == "tanh":
            value = x.unary("tanh") if isinstance(x, ValidatedValue) else np.tanh(x)
            factor = 1.0 - value * value
        elif op in {"arctan", "atan"}:
            value = x.unary("arctan") if isinstance(x, ValidatedValue) else np.arctan(x)
            factor = 1.0 / (1.0 + x * x)
        elif op == "erf":
            # Nominal derivative is fine for proposal construction.  A trusted
            # derivative bound additionally needs a rigorous enclosure of
            # 2/sqrt(pi), which is intentionally deferred rather than silently
            # treating the host constant pi as exact.
            if isinstance(x, ValidatedValue):
                raise NotImplementedError(
                    "AutoIA validated implicit differentiation through erf is deferred pending a trusted 2/sqrt(pi) constant enclosure."
                )
            import math
            value = math.erf(float(x))
            ex = np.exp(-(float(x) ** 2))
            factor = (2.0 / np.sqrt(np.pi)) * ex
        else:
            raise NotImplementedError(
                f"AutoIA implicit differentiation does not support unary operation {op!r}."
            )
        d = np.empty(self.nderiv, dtype=object)
        for i in range(self.nderiv):
            d[i] = factor * self.derivative[i]
        return ImplicitDual(value, d)

    # Branch predicates in residual code are predicates on the primal value.
    def __lt__(self, other):
        return self.value < (other.value if isinstance(other, ImplicitDual) else other)

    def __le__(self, other):
        return self.value <= (other.value if isinstance(other, ImplicitDual) else other)

    def __gt__(self, other):
        return self.value > (other.value if isinstance(other, ImplicitDual) else other)

    def __ge__(self, other):
        return self.value >= (other.value if isinstance(other, ImplicitDual) else other)

    def __eq__(self, other):
        return self.value == (other.value if isinstance(other, ImplicitDual) else other)

    def __ne__(self, other):
        return self.value != (other.value if isinstance(other, ImplicitDual) else other)

    def __bool__(self):
        return bool(self.value)


# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------


def _is_scalar_leaf(obj):
    if isinstance(obj, (ValidatedValue, ImplicitDual)):
        return True
    if np.isscalar(obj):
        return True
    return False


def _map_structure(obj, fn):
    if _is_scalar_leaf(obj):
        return fn(obj)
    if isinstance(obj, tuple):
        return tuple(_map_structure(v, fn) for v in obj)
    if isinstance(obj, list):
        return [_map_structure(v, fn) for v in obj]
    if isinstance(obj, np.ndarray):
        out = np.empty(obj.shape, dtype=object)
        for idx in np.ndindex(obj.shape):
            out[idx] = _map_structure(obj[idx], fn)
        return out
    # Treat other ordinary objects as constants passed through unchanged.
    return obj


def _active_validated_leaves(obj):
    leaves = []

    def visit(x):
        if isinstance(x, ValidatedValue):
            leaves.append(x)
        return x

    _map_structure(obj, visit)
    return leaves


def _closure_contents(func):
    closure = getattr(func, "__closure__", None) or ()
    out = []
    for cell in closure:
        try:
            out.append(cell.cell_contents)
        except ValueError:
            out.append(None)
    return out


def _active_validated_closure_leaves(func):
    return _active_validated_leaves(tuple(_closure_contents(func)))


def _make_cell(value):
    def outer(x):
        def inner():
            return x
        return inner
    return outer(value).__closure__[0]


def _clone_with_closure_transform(func, transform):
    if not isinstance(func, types.FunctionType):
        return func
    closure = getattr(func, "__closure__", None)
    if not closure:
        return func
    new_cells = []
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            return func
        new_cells.append(_make_cell(_map_structure(value, transform)))
    clone = types.FunctionType(
        func.__code__, func.__globals__, name=func.__name__,
        argdefs=func.__defaults__, closure=tuple(new_cells),
    )
    clone.__kwdefaults__ = getattr(func, "__kwdefaults__", None)
    clone.__annotations__ = dict(getattr(func, "__annotations__", {}))
    clone.__dict__.update(getattr(func, "__dict__", {}))
    clone.__module__ = getattr(func, "__module__", None)
    clone.__qualname__ = getattr(func, "__qualname__", func.__name__)
    clone.__doc__ = getattr(func, "__doc__", None)
    return clone


def _seed_function_closure(func, leaf_indices, nderiv, offset=0):
    def seed(x):
        if isinstance(x, ValidatedValue):
            d = _zeros(nderiv)
            d[offset + leaf_indices[id(x)]] = 1.0
            return ImplicitDual(_representative_scalar(x), d)
        return x
    return _clone_with_closure_transform(func, seed)


def _nominalize_function_closure(func):
    return _clone_with_closure_transform(func, _representative_scalar)


def _representative_scalar(x):
    if not isinstance(x, ValidatedValue):
        return x
    rep = x.rep
    coeff = getattr(rep, "coeff", None)
    if isinstance(coeff, dict):
        z = tuple([0] * len(rep.h))
        c = float(coeff.get(z, 0.0))
        if np.isfinite(c):
            return c
    I, _ = x.range()
    if np.isfinite(I.lo) and np.isfinite(I.hi):
        return float(0.5 * I.lo + 0.5 * I.hi)
    raise FloatingPointError("Could not construct a finite implicit-solve proposal value.")


def _nominalize(obj):
    return _map_structure(obj, _representative_scalar)


def _seed_structure(obj, leaf_indices, nderiv, offset=0):
    """Return structure where validated scalar leaves are seeded duals.

    ``leaf_indices`` maps ``id(ValidatedValue)`` occurrences only for the exact
    leaf objects in this argument structure.  Repeated references receive the
    same derivative seed, preserving aliasing at the argument level.
    """
    def seed(x):
        if isinstance(x, ValidatedValue):
            d = _zeros(nderiv)
            d[offset + leaf_indices[id(x)]] = 1.0
            return ImplicitDual(_representative_scalar(x), d)
        return x
    return _map_structure(obj, seed)


def _zero_dual_structure(obj, nderiv):
    def seed(x):
        if isinstance(x, ValidatedValue):
            return ImplicitDual(x, _zeros(nderiv))
        return x
    return _map_structure(obj, seed)


def _interval_correction(prototype: ValidatedValue, lo: float, hi: float):
    rep = IntervalEnclosure(
        Interval(float(lo), float(hi)),
        prototype.h,
        prototype.order,
        prototype.backend,
        prototype.range_method,
    )
    return ValidatedValue(rep)


def _range_of(value):
    if isinstance(value, ValidatedValue):
        return value.range()[0]
    v = float(value)
    return point_interval(v)


def _abs_upper(I):
    return max(abs(float(I.lo)), abs(float(I.hi)))


def _strict_sign(I):
    if I.hi < 0.0:
        return -1
    if I.lo > 0.0:
        return +1
    return 0


def _as_dual_scalar(x, nderiv, seed_index=None):
    d = _zeros(nderiv)
    if seed_index is not None:
        d[int(seed_index)] = 1.0
    return ImplicitDual(x, d)


def _extract_dual_scalar(value):
    arr = np.asarray(value, dtype=object)
    if arr.ndim != 0:
        raise ValueError("root_scalar residual must return one scalar value.")
    v = arr.item()
    if not isinstance(v, ImplicitDual):
        raise NotImplementedError(
            "AutoIA could not differentiate the root_scalar residual through the current supported language."
        )
    return v


def _extract_dual_vector(value, n):
    arr = np.asarray(value, dtype=object).reshape(-1)
    if len(arr) != int(n):
        raise ValueError("fsolve residual dimension must equal the unknown dimension.")
    if not all(isinstance(v, ImplicitDual) for v in arr):
        raise NotImplementedError(
            "AutoIA could not differentiate every fsolve residual component through the current supported language."
        )
    return arr


def _nominal_callable(func):
    base = getattr(func, "__autoia_original_function__", func)
    base = _nominalize_function_closure(base)

    def numeric(*args, **kwargs):
        out = base(*args, **kwargs)
        arr = np.asarray(out, dtype=float)
        return float(arr.item()) if arr.ndim == 0 else arr

    return numeric


def _linear_combination_point(weights, values, prototype):
    total = prototype._constant_like(0.0)
    for w, v in zip(np.asarray(weights, dtype=float).reshape(-1), values):
        if float(w) != 0.0:
            total = total + float(w) * v
    return total


# ---------------------------------------------------------------------------
# Certified result proxies
# ---------------------------------------------------------------------------


class CertifiedRootScalarResult:
    """Read-compatible proxy with a certified dependency-bearing ``root``."""

    def __init__(self, nominal_result, root, certificate):
        self._nominal = nominal_result
        self.root = root
        self.autoia_certificate = certificate

    @property
    def converged(self):
        return True

    @property
    def flag(self):
        return "converged (AutoIA certified)"

    def __getattr__(self, name):
        return getattr(self._nominal, name)


# ---------------------------------------------------------------------------
# Main semantic context
# ---------------------------------------------------------------------------


@dataclass
class _DerivativePiece:
    interval: Interval
    sign: int


class ImplicitSolveContext:
    """Context-local rigorous semantics for SciPy root_scalar and fsolve."""

    def __init__(self, prototype: ValidatedValue):
        self.prototype = prototype
        self.certificates = []

    # ----- scalar ---------------------------------------------------------

    def _scalar_derivative_over_y(self, func, y_value, args):
        # One derivative seed: derivative with respect to y.
        ydual = _as_dual_scalar(y_value, 1, 0)
        dual_args = _zero_dual_structure(args, 1)
        out = _extract_dual_scalar(func(ydual, *dual_args))
        return out.derivative[0]

    def _prove_scalar_bracket(self, func, bracket, args, max_pieces=128):
        a, b = map(float, bracket)
        if not (np.isfinite(a) and np.isfinite(b) and a < b):
            raise ValueError("Validated root_scalar requires a finite increasing two-point bracket.")

        fa = _range_of(func(a, *args))
        fb = _range_of(func(b, *args))
        sa, sb = _strict_sign(fa), _strict_sign(fb)
        if sa == 0 or sb == 0 or sa == sb:
            raise RuntimeError(
                "Validated root_scalar could not prove a uniform strict sign change at the supplied bracket endpoints."
            )

        # Piecewise derivative proof.  Start coarse and bisect only pieces whose
        # derivative enclosure contains zero.  All accepted pieces must have the
        # same derivative sign.
        pieces = [(a, b)]
        accepted = []
        while pieces:
            lo, hi = pieces.pop()
            yI = _interval_correction(self.prototype, lo, hi)
            try:
                d = self._scalar_derivative_over_y(func, yI, args)
                DI = _range_of(d)
                s = _strict_sign(DI)
            except (ZeroDivisionError, DomainRefinementRequired):
                # A reciprocal/domain proof failure inside the derivative trace
                # is first a root-variable localization problem.  Preserve the
                # historical validated root_scalar behavior by bisecting the
                # y-bracket piece rather than leaking this internal proof signal
                # to the outer physical-box controller.
                DI = None
                s = 0
            if s != 0:
                accepted.append(_DerivativePiece(DI, s))
                continue
            if len(accepted) + len(pieces) + 2 > max_pieces:
                raise RuntimeError(
                    "Validated root_scalar could not prove derivative separation from zero over the supplied bracket within the derivative-subdivision budget."
                )
            mid = 0.5 * lo + 0.5 * hi
            if not (lo < mid < hi):
                raise RuntimeError(
                    "Validated root_scalar derivative subdivision reached binary64 resolution before proving uniqueness."
                )
            pieces.append((mid, hi))
            pieces.append((lo, mid))

        signs = {p.sign for p in accepted}
        if len(signs) != 1:
            raise RuntimeError(
                "Validated root_scalar derivative changes sign across the supplied bracket; uniqueness was not certified."
            )
        derivative_sign = signs.pop()
        # Endpoint orientation must agree with monotonicity.
        if derivative_sign > 0 and not (sa < 0 < sb):
            raise RuntimeError("Validated root_scalar endpoint signs are inconsistent with the proven increasing residual.")
        if derivative_sign < 0 and not (sb < 0 < sa):
            raise RuntimeError("Validated root_scalar endpoint signs are inconsistent with the proven decreasing residual.")
        return fa, fb, accepted

    def traced_root_scalar(self, real_root_scalar, func, *call_args, **call_kwargs):
        bound = inspect.signature(real_root_scalar).bind_partial(
            func, *call_args, **call_kwargs
        )
        bracket = bound.arguments.get("bracket", None)
        if bracket is None:
            raise NotImplementedError(
                "AutoIA Version-1 validated root_scalar requires an explicit bracket=[a,b]."
            )
        if _active_validated_leaves(bracket):
            raise NotImplementedError(
                "AutoIA validated root_scalar currently requires constant numeric bracket endpoints."
            )
        method = bound.arguments.get("method", None)
        bracket_methods = {None, "bisect", "brentq", "brenth", "ridder", "toms748"}
        if method not in bracket_methods:
            raise NotImplementedError(
                "AutoIA validated root_scalar currently supports the bracketed methods "
                "bisect, brentq, brenth, ridder, toms748, or method=None with a bracket."
            )
        args = bound.arguments.get("args", ())
        if args is None:
            args = ()
        if not isinstance(args, tuple):
            args = tuple(args) if isinstance(args, (list, np.ndarray)) else (args,)

        active = _active_validated_leaves(args) + _active_validated_closure_leaves(func)
        if not active:
            return real_root_scalar(func, *call_args, **call_kwargs)

        # Prove the supplied bracket has exactly one root for every parameter
        # before relying on SciPy even as a nominal proposal engine.
        fa, fb, derivative_pieces = self._prove_scalar_bracket(func, bracket, args)

        # Nominal SciPy solve is proposal only.
        nominal_args = _nominalize(args)
        bound.arguments["f"] = _nominal_callable(func)
        bound.arguments["args"] = nominal_args
        nominal = real_root_scalar(*bound.args, **bound.kwargs)
        if not bool(getattr(nominal, "converged", False)):
            raise RuntimeError("Nominal root_scalar proposal did not converge.")
        y0 = float(nominal.root)
        if not np.isfinite(y0):
            raise FloatingPointError("Nominal root_scalar proposal is nonfinite.")

        # Differentiate nominal residual with respect to y and active argument
        # leaves, then build an affine implicit-function predictor.
        unique_active = []
        seen = set()
        for v in active:
            if id(v) not in seen:
                seen.add(id(v))
                unique_active.append(v)
        leaf_index = {id(v): i for i, v in enumerate(unique_active)}
        npar = len(unique_active)
        nderiv = 1 + npar
        ydual = _as_dual_scalar(y0, nderiv, 0)
        seeded_args = _seed_structure(args, leaf_index, nderiv, offset=1)
        seeded_func = _seed_function_closure(func, leaf_index, nderiv, offset=1)
        dout = _extract_dual_scalar(seeded_func(ydual, *seeded_args))
        Jy = float(dout.derivative[0])
        Ja = np.asarray([float(dout.derivative[1+i]) for i in range(npar)], dtype=float)
        if not np.isfinite(Jy) or Jy == 0.0 or np.any(~np.isfinite(Ja)):
            raise RuntimeError("Could not construct a finite nonsingular root_scalar implicit predictor.")
        sensitivity = -Ja / Jy

        predictor = self.prototype._constant_like(y0)
        for s, v in zip(sensitivity, unique_active):
            a0 = float(_representative_scalar(v))
            predictor = predictor + float(s) * (v - a0)

        residual = func(predictor, *args)
        rI = _range_of(residual)
        C = 1.0 / Jy
        c = _abs_upper(self.prototype.backend.mul(point_interval(C), rI))

        # Self-consistent correction tube for e in [-d,d].
        d = max(1e-15, 2.0 * c)
        rho = np.inf
        derivative_tube = None
        for _ in range(24):
            ytube = predictor + _interval_correction(self.prototype, -d, d)
            dy = self._scalar_derivative_over_y(func, ytube, args)
            derivative_tube = _range_of(dy)
            D = self.prototype.backend.sub(
                point_interval(1.0),
                self.prototype.backend.mul(point_interval(C), derivative_tube),
            )
            rho = _abs_upper(D)
            if not rho < 1.0:
                d *= 2.0
                continue
            delta = self.prototype.backend.sub(point_interval(1.0), point_interval(rho)).lo
            if not delta > 0.0:
                d *= 2.0
                continue
            required = self.prototype.backend.div(point_interval(c), point_interval(delta)).hi
            if required <= d:
                d = float(required)
                break
            d = float(required * 1.05)
        else:
            raise RuntimeError(
                "Validated root_scalar local contraction certificate failed within the correction iteration budget."
            )

        root = predictor + _interval_correction(self.prototype, -d, d)
        rootI = _range_of(root)
        a, b = map(float, bracket)
        if rootI.lo < a or rootI.hi > b:
            raise RuntimeError(
                "Validated root_scalar correction enclosure is not contained in the supplied bracket."
            )

        cert = {
            "operation": "scipy.optimize.root_scalar",
            "semantic_mode": "bracketed-unique-implicit-root",
            "nominal_root": y0,
            "bracket": (a, b),
            "endpoint_residual_ranges": (fa, fb),
            "bracket_derivative_piece_ranges": tuple(p.interval for p in derivative_pieces),
            "nominal_Jy": Jy,
            "nominal_Ja": Ja,
            "implicit_sensitivity": sensitivity,
            "predictor": predictor,
            "predictor_residual_range": rI,
            "preconditioner_C": C,
            "forcing_upper": c,
            "derivative_tube_range": derivative_tube,
            "rho_upper": float(rho),
            "correction_radius": float(d),
            "root_range": rootI,
            "unique_in_bracket": True,
            "machine_rigorous": bool(self.prototype.backend.machine_rigorous),
        }
        self.certificates.append(cert)
        return CertifiedRootScalarResult(nominal, root, cert)

    # ----- vector ---------------------------------------------------------

    def _vector_jacobian_over_tube(self, func, y_values, args):
        y_obj = np.asarray(y_values, dtype=object).reshape(-1)
        n = len(y_obj)
        ydual = np.empty(n, dtype=object)
        for j in range(n):
            ydual[j] = _as_dual_scalar(y_obj[j], n, j)
        dual_args = _zero_dual_structure(args, n)
        out = _extract_dual_vector(func(ydual, *dual_args), n)
        J = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                J[i, j] = out[i].derivative[j]
        return J

    def traced_fsolve(self, real_fsolve, func, x0, *call_args, **call_kwargs):
        bound = inspect.signature(real_fsolve).bind_partial(
            func, x0, *call_args, **call_kwargs
        )
        args = bound.arguments.get("args", ())
        if args is None:
            args = ()
        if not isinstance(args, tuple):
            args = tuple(args) if isinstance(args, (list, np.ndarray)) else (args,)
        active = _active_validated_leaves(args) + _active_validated_closure_leaves(func)
        if _active_validated_leaves(x0):
            raise NotImplementedError(
                "AutoIA validated fsolve requires a constant numeric initial guess x0; "
                "uncertainty belongs in the residual parameters, not the algorithmic proposal."
            )
        x0_arr = np.asarray(x0, dtype=float).reshape(-1)
        if not active:
            return real_fsolve(func, x0, *call_args, **call_kwargs)
        if np.any(~np.isfinite(x0_arr)):
            raise ValueError("Validated fsolve requires a finite numeric initial guess.")

        # Nominal proposal. Force full_output internally so convergence is known,
        # while preserving the caller's requested return form later.
        requested_full_output = bool(bound.arguments.get("full_output", False))
        bound.arguments["func"] = _nominal_callable(func)
        bound.arguments["x0"] = x0_arr
        bound.arguments["args"] = _nominalize(args)
        bound.arguments["full_output"] = True
        nominal_x, info, ier, mesg = real_fsolve(*bound.args, **bound.kwargs)
        nominal_x = np.asarray(nominal_x, dtype=float).reshape(-1)
        n = len(nominal_x)
        if int(ier) != 1 or np.any(~np.isfinite(nominal_x)):
            raise RuntimeError(f"Nominal fsolve proposal did not converge: {mesg}")

        # Active argument leaves and nominal derivative model.
        unique_active = []
        seen = set()
        for v in active:
            if id(v) not in seen:
                seen.add(id(v))
                unique_active.append(v)
        leaf_index = {id(v): i for i, v in enumerate(unique_active)}
        npar = len(unique_active)
        nderiv = n + npar

        ydual = np.empty(n, dtype=object)
        for j in range(n):
            ydual[j] = _as_dual_scalar(float(nominal_x[j]), nderiv, j)
        seeded_args = _seed_structure(args, leaf_index, nderiv, offset=n)
        seeded_func = _seed_function_closure(func, leaf_index, nderiv, offset=n)
        dout = _extract_dual_vector(seeded_func(ydual, *seeded_args), n)
        Jy = np.empty((n, n), dtype=float)
        Ja = np.empty((n, npar), dtype=float)
        for i in range(n):
            for j in range(n):
                Jy[i, j] = float(dout[i].derivative[j])
            for k in range(npar):
                Ja[i, k] = float(dout[i].derivative[n+k])
        if np.any(~np.isfinite(Jy)) or np.any(~np.isfinite(Ja)):
            raise RuntimeError("Could not construct finite fsolve implicit Jacobians.")
        C = np.linalg.solve(Jy, np.eye(n))
        sensitivity = -(C @ Ja)

        predictor = np.empty(n, dtype=object)
        for j in range(n):
            vj = self.prototype._constant_like(float(nominal_x[j]))
            for k, arg_v in enumerate(unique_active):
                if float(sensitivity[j, k]) != 0.0:
                    a0 = float(_representative_scalar(arg_v))
                    vj = vj + float(sensitivity[j, k]) * (arg_v - a0)
            predictor[j] = vj

        residual = np.asarray(func(predictor, *args), dtype=object).reshape(-1)
        if len(residual) != n:
            raise ValueError("Validated fsolve requires a square residual system.")
        Cr = np.empty(n, dtype=object)
        c_vec = np.empty(n, dtype=float)
        for i in range(n):
            Cr[i] = _linear_combination_point(C[i, :], residual, self.prototype)
            c_vec[i] = _abs_upper(_range_of(Cr[i]))
        c = float(np.max(c_vec))

        d = max(1e-15, 2.0 * c)
        rho = np.inf
        J_tube_ranges = None
        B = None
        for _ in range(24):
            ytube = np.empty(n, dtype=object)
            for j in range(n):
                ytube[j] = predictor[j] + _interval_correction(self.prototype, -d, d)
            Jv = self._vector_jacobian_over_tube(func, ytube, args)
            J_tube_ranges = np.empty((n, n), dtype=object)
            for i in range(n):
                for j in range(n):
                    J_tube_ranges[i, j] = _range_of(Jv[i, j])

            # Rigorous D = I - C J interval matrix.
            B = np.empty((n, n), dtype=float)
            for i in range(n):
                for j in range(n):
                    total = point_interval(0.0)
                    for k in range(n):
                        term = self.prototype.backend.mul(
                            point_interval(float(C[i, k])), J_tube_ranges[k, j]
                        )
                        total = self.prototype.backend.add(total, term)
                    dij = self.prototype.backend.sub(
                        point_interval(1.0 if i == j else 0.0), total
                    )
                    B[i, j] = _abs_upper(dij)
            row_sums = np.sum(B, axis=1)
            # np.sum here is proposal arithmetic over nonnegative rigorous entry
            # bounds. Round the final row sum outward through the trusted backend.
            rigorous_rows = []
            for i in range(n):
                s = point_interval(0.0)
                for j in range(n):
                    s = self.prototype.backend.add(s, point_interval(float(B[i, j])))
                rigorous_rows.append(s.hi)
            rho = float(max(rigorous_rows))
            if not rho < 1.0:
                d *= 2.0
                continue
            delta = self.prototype.backend.sub(point_interval(1.0), point_interval(rho)).lo
            if not delta > 0.0:
                d *= 2.0
                continue
            required = self.prototype.backend.div(point_interval(c), point_interval(delta)).hi
            if required <= d:
                d = float(required)
                break
            d = float(required * 1.05)
        else:
            raise RuntimeError(
                "Validated fsolve local contraction certificate failed within the correction iteration budget."
            )

        out = np.empty(n, dtype=object)
        for j in range(n):
            out[j] = predictor[j] + _interval_correction(self.prototype, -d, d)

        cert = {
            "operation": "scipy.optimize.fsolve",
            "semantic_mode": "local-unique-implicit-system",
            "nominal_root": nominal_x,
            "nominal_Jy": Jy,
            "nominal_Ja": Ja,
            "implicit_sensitivity": sensitivity,
            "predictor": predictor,
            "preconditioner_C": C,
            "preconditioned_residual": Cr,
            "forcing_component_upper": c_vec,
            "forcing_upper": c,
            "jacobian_tube_ranges": J_tube_ranges,
            "defect_abs_upper": B,
            "rho_upper": rho,
            "correction_radius": d,
            "machine_rigorous": bool(self.prototype.backend.machine_rigorous),
        }
        self.certificates.append(cert)

        if requested_full_output:
            info_out = dict(info)
            info_out["autoia_certificate"] = cert
            return out, info_out, 1, "converged (AutoIA certified)"
        return out


# Context-local stable hooks used by SciPy shadow dispatch.
_ACTIVE_IMPLICIT_CONTEXT: ContextVar[ImplicitSolveContext | None] = ContextVar(
    "autoia_active_implicit_solve_context", default=None
)


class ImplicitSolveExecutionContext:
    def __init__(self, context: ImplicitSolveContext):
        self.context = context
        self._token = None

    def __enter__(self):
        self._token = _ACTIVE_IMPLICIT_CONTEXT.set(self.context)
        return self.context

    def __exit__(self, exc_type, exc, tb):
        _ACTIVE_IMPLICIT_CONTEXT.reset(self._token)
        return False


def prepared_traced_root_scalar(real_root_scalar, func, *args, **kwargs):
    ctx = _ACTIVE_IMPLICIT_CONTEXT.get()
    if ctx is None:
        return real_root_scalar(func, *args, **kwargs)
    return ctx.traced_root_scalar(real_root_scalar, func, *args, **kwargs)


def prepared_traced_fsolve(real_fsolve, func, x0, *args, **kwargs):
    ctx = _ACTIVE_IMPLICIT_CONTEXT.get()
    if ctx is None:
        return real_fsolve(func, x0, *args, **kwargs)
    return ctx.traced_fsolve(real_fsolve, func, x0, *args, **kwargs)


__all__ = [
    "ImplicitDual",
    "ImplicitSolveContext",
    "ImplicitSolveExecutionContext",
    "CertifiedRootScalarResult",
    "prepared_traced_root_scalar",
    "prepared_traced_fsolve",
]
