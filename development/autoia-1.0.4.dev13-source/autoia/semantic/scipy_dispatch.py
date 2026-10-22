"""Local compatibility dispatch for common SciPy normal-distribution APIs.

This module is intentionally *not* a SciPy dependency of AutoIA.  It activates
only when SciPy has already been imported by the user's model module (which
means the relevant SciPy modules are present in ``sys.modules``).  AutoIA then
replaces references only inside its private shadow execution namespace.
SciPy modules, classes, ufuncs, and distribution objects are never mutated.

Validated entry points currently supported
-------------------------------------------
* ``scipy.special.erf(x)``
* ``scipy.special.ndtr(x)``
* ``scipy.stats.norm.pdf(x, loc=..., scale=...)``
* ``scipy.stats.norm.cdf(x, loc=..., scale=...)``
* frozen ``scipy.stats.norm(loc=..., scale=...).pdf/cdf``
* ``scipy.stats.Normal(mu=..., sigma=...).pdf/cdf`` when that API exists
* ``scipy.optimize.root_scalar`` with validated parameters and an explicit bracket
* ``scipy.optimize.fsolve`` for square systems with validated parameters
* ``scipy.optimize.linprog`` optimal-value certification for parametric LP data
* ``scipy.interpolate.RegularGridInterpolator`` with fixed data and ``method="linear"``

Recognized-but-not-yet-certified entry points
-------------------------------------------
* ``scipy.linalg.eigh`` with AutoIA-dependent matrices fails explicitly closed

For validated execution, normal location/scale parameters must be constant real
scalars.  Inverse CDFs, survival functions, logarithmic variants, and uncertain
distribution parameters are deliberately outside this compatibility layer.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np

from ..arrays import contains_validated as array_contains_validated
from ..special import erf as autoia_erf
from ..special import standard_normal_cdf, normal_pdf, normal_cdf


def _contains_validated(obj: Any) -> bool:
    try:
        return array_contains_validated(obj)
    except Exception:
        return False


def _validated_call_present(*values: Any) -> bool:
    return any(_contains_validated(v) for v in values)


def _reject_out_and_extra(kind: str, args, kwargs) -> None:
    if args:
        raise NotImplementedError(
            f"AutoIA validated {kind} currently supports only the ordinary input argument."
        )
    out = kwargs.pop("out", None)
    if out is not None:
        raise NotImplementedError(
            f"AutoIA validated {kind} does not yet support SciPy out= semantics."
        )
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise NotImplementedError(
            f"AutoIA validated {kind} does not yet support SciPy keyword(s): {names}."
        )


def _make_special_erf(real_func):
    def traced(x, *args, **kwargs):
        if not _contains_validated(x):
            return real_func(x, *args, **kwargs)
        local_kwargs = dict(kwargs)
        _reject_out_and_extra("scipy.special.erf", args, local_kwargs)
        return autoia_erf(x)

    traced.__name__ = getattr(real_func, "__name__", "erf")
    traced.__doc__ = getattr(real_func, "__doc__", None)
    return traced


def _make_special_ndtr(real_func):
    def traced(x, *args, **kwargs):
        if not _contains_validated(x):
            return real_func(x, *args, **kwargs)
        local_kwargs = dict(kwargs)
        _reject_out_and_extra("scipy.special.ndtr", args, local_kwargs)
        return standard_normal_cdf(x)

    traced.__name__ = getattr(real_func, "__name__", "ndtr")
    traced.__doc__ = getattr(real_func, "__doc__", None)
    return traced


def _parse_norm_loc_scale(args, kwargs) -> tuple[Any, Any]:
    """Parse the public ``scipy.stats.norm`` loc/scale convention."""
    if len(args) > 2:
        raise NotImplementedError(
            "AutoIA validated scipy.stats.norm pdf/cdf supports only x plus optional loc and scale."
        )
    kw = dict(kwargs)
    unexpected = set(kw) - {"loc", "scale"}
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise NotImplementedError(
            f"AutoIA validated scipy.stats.norm pdf/cdf does not support keyword(s): {names}."
        )
    if len(args) >= 1 and "loc" in kw:
        raise TypeError("loc supplied by both positional and keyword argument")
    if len(args) >= 2 and "scale" in kw:
        raise TypeError("scale supplied by both positional and keyword argument")
    loc = args[0] if len(args) >= 1 else kw.get("loc", 0.0)
    scale = args[1] if len(args) >= 2 else kw.get("scale", 1.0)
    return loc, scale


def _parse_frozen_norm_parameters(args, kwargs) -> tuple[Any, Any]:
    if len(args) > 2:
        raise NotImplementedError(
            "AutoIA scipy.stats.norm frozen compatibility supports only loc and scale."
        )
    kw = dict(kwargs)
    unexpected = set(kw) - {"loc", "scale"}
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise NotImplementedError(
            f"AutoIA scipy.stats.norm frozen compatibility does not support keyword(s): {names}."
        )
    if len(args) >= 1 and "loc" in kw:
        raise TypeError("loc supplied by both positional and keyword argument")
    if len(args) >= 2 and "scale" in kw:
        raise TypeError("scale supplied by both positional and keyword argument")
    return (
        args[0] if len(args) >= 1 else kw.get("loc", 0.0),
        args[1] if len(args) >= 2 else kw.get("scale", 1.0),
    )


class _FrozenNormProxy:
    __slots__ = ("_target", "_loc", "_scale")

    def __init__(self, target, loc, scale):
        self._target = target
        self._loc = loc
        self._scale = scale

    def pdf(self, x):
        if not _contains_validated(x):
            return self._target.pdf(x)
        return normal_pdf(x, mean=self._loc, std=self._scale)

    def cdf(self, x):
        if not _contains_validated(x):
            return self._target.cdf(x)
        return normal_cdf(x, mean=self._loc, std=self._scale)

    def __getattr__(self, name):
        return getattr(self._target, name)


class _NormProxy:
    __slots__ = ("_target",)

    def __init__(self, target):
        self._target = target

    def pdf(self, x, *args, **kwargs):
        if not _validated_call_present(x, *args, *kwargs.values()):
            return self._target.pdf(x, *args, **kwargs)
        loc, scale = _parse_norm_loc_scale(args, kwargs)
        return normal_pdf(x, mean=loc, std=scale)

    def cdf(self, x, *args, **kwargs):
        if not _validated_call_present(x, *args, *kwargs.values()):
            return self._target.cdf(x, *args, **kwargs)
        loc, scale = _parse_norm_loc_scale(args, kwargs)
        return normal_cdf(x, mean=loc, std=scale)

    def __call__(self, *args, **kwargs):
        # Creating a frozen distribution is not itself proof-critical when loc
        # and scale are constants.  Construct the real SciPy object so all
        # unrelated attributes/methods remain available, then intercept pdf/cdf.
        if _validated_call_present(*args, *kwargs.values()):
            raise NotImplementedError(
                "AutoIA scipy.stats.norm frozen distributions currently require constant loc and scale."
            )
        loc, scale = _parse_frozen_norm_parameters(args, kwargs)
        target = self._target(*args, **kwargs)
        return _FrozenNormProxy(target, loc, scale)

    def __getattr__(self, name):
        return getattr(self._target, name)

    def __dir__(self):
        return sorted(set(dir(self._target)) | {"pdf", "cdf"})


class _NormalInstanceProxy:
    __slots__ = ("_target", "_mu", "_sigma")

    def __init__(self, target, mu, sigma):
        self._target = target
        self._mu = mu
        self._sigma = sigma

    def pdf(self, x, /, *, method=None):
        if not _contains_validated(x):
            return self._target.pdf(x, method=method)
        if method is not None:
            raise NotImplementedError(
                "AutoIA validated scipy.stats.Normal.pdf currently requires method=None."
            )
        return normal_pdf(x, mean=self._mu, std=self._sigma)

    def cdf(self, x, y=None, /, *, method=None):
        if not _validated_call_present(x, y):
            if y is None:
                return self._target.cdf(x, method=method)
            return self._target.cdf(x, y, method=method)
        if method is not None:
            raise NotImplementedError(
                "AutoIA validated scipy.stats.Normal.cdf currently requires method=None."
            )
        if y is None:
            return normal_cdf(x, mean=self._mu, std=self._sigma)
        # For a continuous normal distribution, the two-argument CDF is the
        # probability between x and y.  Difference-of-CDFs preserves dependency
        # through AutoIA arithmetic and is mathematically exact.
        return normal_cdf(y, mean=self._mu, std=self._sigma) - normal_cdf(
            x, mean=self._mu, std=self._sigma
        )

    def __getattr__(self, name):
        return getattr(self._target, name)


class _NormalClassProxy:
    __slots__ = ("_target",)

    def __init__(self, target):
        self._target = target

    def __call__(self, mu=None, sigma=None, **kwargs):
        if _validated_call_present(mu, sigma):
            raise NotImplementedError(
                "AutoIA scipy.stats.Normal currently requires constant mu and sigma."
            )
        target = self._target(mu=mu, sigma=sigma, **kwargs)
        # Read the normalized public parameters back from SciPy so default
        # values are preserved exactly (mu=0, sigma=1 when omitted).
        return _NormalInstanceProxy(target, float(target.mu), float(target.sigma))

    def __getattr__(self, name):
        return getattr(self._target, name)

    def __dir__(self):
        return dir(self._target)


class _StatsProxy:
    __slots__ = ("_target", "_norm", "_Normal")

    def __init__(self, target, norm_proxy, Normal_proxy):
        self._target = target
        self._norm = norm_proxy
        self._Normal = Normal_proxy

    def __getattr__(self, name):
        if name == "norm":
            return self._norm
        if name == "Normal" and self._Normal is not None:
            return self._Normal
        return getattr(self._target, name)

    def __dir__(self):
        extras = {"norm"}
        if self._Normal is not None:
            extras.add("Normal")
        return sorted(set(dir(self._target)) | extras)


class _SpecialProxy:
    __slots__ = ("_target", "_erf", "_ndtr")

    def __init__(self, target, erf_func, ndtr_func):
        self._target = target
        self._erf = erf_func
        self._ndtr = ndtr_func

    def __getattr__(self, name):
        if name == "erf":
            return self._erf
        if name == "ndtr":
            return self._ndtr
        return getattr(self._target, name)

    def __dir__(self):
        return sorted(set(dir(self._target)) | {"erf", "ndtr"})


def _prepare_optimize_callback(func):
    if getattr(func, "__autoia_dispatch_prepared__", False):
        return func
    # A callback imported from another pure-Python module can still be prepared
    # locally when it is handed to a recognized SciPy semantic primitive.
    try:
        from .dispatch import prepare_model_for_dispatch, _prepared_traced_solve
        return prepare_model_for_dispatch(func, _prepared_traced_solve)
    except Exception:
        return func


def _function_closure_contains_validated(func):
    closure = getattr(func, "__closure__", None) or ()
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if _contains_validated(value):
            return True
    return False


def _make_optimize_root_scalar(real_func):
    def traced(func, *args, **kwargs):
        call_args = kwargs.get("args", args[0] if args else ())
        bracket_value = kwargs.get("bracket", args[2] if len(args) >= 3 else None)
        if not (
            _validated_call_present(call_args, bracket_value)
            or _function_closure_contains_validated(func)
        ):
            return real_func(func, *args, **kwargs)
        func = _prepare_optimize_callback(func)
        from .implicit_solve import prepared_traced_root_scalar
        return prepared_traced_root_scalar(real_func, func, *args, **kwargs)
    traced.__name__ = getattr(real_func, "__name__", "root_scalar")
    traced.__doc__ = getattr(real_func, "__doc__", None)
    return traced


def _make_optimize_fsolve(real_func):
    def traced(func, x0, *args, **kwargs):
        call_args = kwargs.get("args", args[0] if args else ())
        if not (
            _validated_call_present(call_args, x0)
            or _function_closure_contains_validated(func)
        ):
            return real_func(func, x0, *args, **kwargs)
        func = _prepare_optimize_callback(func)
        from .implicit_solve import prepared_traced_fsolve
        return prepared_traced_fsolve(real_func, func, x0, *args, **kwargs)
    traced.__name__ = getattr(real_func, "__name__", "fsolve")
    traced.__doc__ = getattr(real_func, "__doc__", None)
    return traced


def _make_optimize_linprog(real_func):
    def traced(*args, **kwargs):
        # Unlike ordinary elementwise operations, even a constant-data LP call
        # inside an AutoIA execution must be certified: a floating solver result
        # is only a proposal for the mathematical optimum.  The active semantic
        # context therefore decides whether to intercept, not merely the presence
        # of ValidatedValue arguments.
        from .linear_program import active_linear_program_context, prepared_traced_linprog
        if active_linear_program_context() is None:
            return real_func(*args, **kwargs)
        return prepared_traced_linprog(real_func, *args, **kwargs)
    traced.__name__ = getattr(real_func, "__name__", "linprog")
    traced.__doc__ = getattr(real_func, "__doc__", None)
    return traced


class _OptimizeProxy:
    __slots__ = ("_target", "_root_scalar", "_fsolve", "_linprog")

    def __init__(self, target, root_scalar_func, fsolve_func, linprog_func):
        self._target = target
        self._root_scalar = root_scalar_func
        self._fsolve = fsolve_func
        self._linprog = linprog_func

    def __getattr__(self, name):
        if name == "root_scalar":
            return self._root_scalar
        if name == "fsolve":
            return self._fsolve
        if name == "linprog":
            return self._linprog
        return getattr(self._target, name)

    def __dir__(self):
        return sorted(set(dir(self._target)) | {"root_scalar", "fsolve", "linprog"})




class _RegularGridInterpolatorClassProxy:
    __slots__ = ("_target",)

    def __init__(self, target):
        self._target = target

    def __call__(self, *args, **kwargs):
        points = args[0] if len(args) >= 1 else kwargs.get("points")
        values = args[1] if len(args) >= 2 else kwargs.get("values")
        if _validated_call_present(points, values):
            raise NotImplementedError(
                "AutoIA validated RegularGridInterpolator requires fixed numeric "
                "grid coordinates and fixed numeric table values; uncertainty is "
                "currently supported only in the query coordinates."
            )
        target = self._target(*args, **kwargs)
        from .interpolation import RegularGridInterpolatorProxy
        return RegularGridInterpolatorProxy(target)

    def __getattr__(self, name):
        return getattr(self._target, name)

    def __dir__(self):
        return dir(self._target)


class _InterpolateProxy:
    __slots__ = ("_target", "_rgi")

    def __init__(self, target, rgi_proxy):
        self._target = target
        self._rgi = rgi_proxy

    def __getattr__(self, name):
        if name == "RegularGridInterpolator":
            return self._rgi
        return getattr(self._target, name)

    def __dir__(self):
        return sorted(set(dir(self._target)) | {"RegularGridInterpolator"})


def _make_linalg_eigh(real_func):
    def traced(a, b=None, *args, **kwargs):
        if not _validated_call_present(a, b):
            return real_func(a, b, *args, **kwargs)
        raise NotImplementedError(
            "AutoIA recognizes scipy.linalg.eigh with AutoIA-dependent matrices, "
            "but validated symmetric/generalized-symmetric eigenvalue semantics "
            "are not implemented in this V1 build. No certificate was produced."
        )
    traced.__name__ = getattr(real_func, "__name__", "eigh")
    traced.__doc__ = getattr(real_func, "__doc__", None)
    return traced


class _LinalgProxy:
    __slots__ = ("_target", "_eigh")

    def __init__(self, target, eigh_func):
        self._target = target
        self._eigh = eigh_func

    def __getattr__(self, name):
        if name == "eigh":
            return self._eigh
        return getattr(self._target, name)

    def __dir__(self):
        return sorted(set(dir(self._target)) | {"eigh"})


class _ScipyProxy:
    __slots__ = ("_target", "_stats", "_special", "_optimize", "_interpolate", "_linalg")

    def __init__(self, target, stats_proxy, special_proxy, optimize_proxy=None, interpolate_proxy=None, linalg_proxy=None):
        self._target = target
        self._stats = stats_proxy
        self._special = special_proxy
        self._optimize = optimize_proxy
        self._interpolate = interpolate_proxy
        self._linalg = linalg_proxy

    def __getattr__(self, name):
        if name == "stats" and self._stats is not None:
            return self._stats
        if name == "special" and self._special is not None:
            return self._special
        if name == "optimize" and self._optimize is not None:
            return self._optimize
        if name == "interpolate" and self._interpolate is not None:
            return self._interpolate
        if name == "linalg" and self._linalg is not None:
            return self._linalg
        return getattr(self._target, name)

    def __dir__(self):
        return sorted(set(dir(self._target)) | {"stats", "special", "optimize", "interpolate", "linalg"})


def _frozen_norm_proxy_for(value, real_norm):
    dist = getattr(value, "dist", None)
    # SciPy frozen continuous distributions may hold a cloned distribution
    # generator rather than the exact module-level ``stats.norm`` singleton.
    # Identify the normal generator by its specific class and public name.
    if dist is None or not (
        isinstance(dist, real_norm.__class__)
        and getattr(dist, "name", None) == getattr(real_norm, "name", "norm") == "norm"
    ):
        return None
    args = tuple(getattr(value, "args", ()))
    kwargs = dict(getattr(value, "kwds", {}))
    loc, scale = _parse_frozen_norm_parameters(args, kwargs)
    return _FrozenNormProxy(value, loc, scale)


def apply_scipy_shadow_replacements(shadow: dict[str, Any]) -> None:
    """Replace SciPy references in one private AutoIA shadow namespace.

    The function is a no-op unless SciPy has already been imported by the user.
    It mutates only ``shadow``; no object in ``sys.modules`` is modified.
    """
    scipy_mod = sys.modules.get("scipy")
    stats_mod = sys.modules.get("scipy.stats")
    special_mod = sys.modules.get("scipy.special")
    optimize_mod = sys.modules.get("scipy.optimize")
    interpolate_mod = sys.modules.get("scipy.interpolate")
    linalg_mod = sys.modules.get("scipy.linalg")
    if scipy_mod is None and stats_mod is None and special_mod is None and optimize_mod is None and interpolate_mod is None and linalg_mod is None:
        return

    # If the user's module imported only top-level ``scipy`` and refers to
    # ``scipy.stats`` or ``scipy.special`` lazily, load those public submodules
    # now.  This remains optional: it happens only after the user has already
    # imported SciPy, and AutoIA still declares no SciPy package dependency.
    if scipy_mod is not None:
        if stats_mod is None:
            try:
                stats_mod = importlib.import_module("scipy.stats")
            except Exception:
                stats_mod = None
        if special_mod is None:
            try:
                special_mod = importlib.import_module("scipy.special")
            except Exception:
                special_mod = None
        if optimize_mod is None:
            try:
                optimize_mod = importlib.import_module("scipy.optimize")
            except Exception:
                optimize_mod = None
        if interpolate_mod is None:
            try:
                interpolate_mod = importlib.import_module("scipy.interpolate")
            except Exception:
                interpolate_mod = None
        if linalg_mod is None:
            try:
                linalg_mod = importlib.import_module("scipy.linalg")
            except Exception:
                linalg_mod = None

    norm_proxy = None
    Normal_proxy = None
    real_norm = None
    real_Normal = None
    if stats_mod is not None:
        real_norm = getattr(stats_mod, "norm", None)
        if real_norm is not None:
            norm_proxy = _NormProxy(real_norm)
        real_Normal = getattr(stats_mod, "Normal", None)
        if real_Normal is not None:
            Normal_proxy = _NormalClassProxy(real_Normal)

    erf_dispatch = None
    ndtr_dispatch = None
    real_erf = None
    real_ndtr = None
    if special_mod is not None:
        real_erf = getattr(special_mod, "erf", None)
        real_ndtr = getattr(special_mod, "ndtr", None)
        if real_erf is not None:
            erf_dispatch = _make_special_erf(real_erf)
        if real_ndtr is not None:
            ndtr_dispatch = _make_special_ndtr(real_ndtr)

    stats_proxy = (
        _StatsProxy(stats_mod, norm_proxy, Normal_proxy)
        if stats_mod is not None and norm_proxy is not None
        else None
    )
    special_proxy = (
        _SpecialProxy(special_mod, erf_dispatch, ndtr_dispatch)
        if special_mod is not None and erf_dispatch is not None and ndtr_dispatch is not None
        else None
    )
    real_root_scalar = getattr(optimize_mod, "root_scalar", None) if optimize_mod is not None else None
    real_fsolve = getattr(optimize_mod, "fsolve", None) if optimize_mod is not None else None
    real_linprog = getattr(optimize_mod, "linprog", None) if optimize_mod is not None else None
    root_scalar_dispatch = _make_optimize_root_scalar(real_root_scalar) if real_root_scalar is not None else None
    fsolve_dispatch = _make_optimize_fsolve(real_fsolve) if real_fsolve is not None else None
    linprog_dispatch = _make_optimize_linprog(real_linprog) if real_linprog is not None else None
    optimize_proxy = (
        _OptimizeProxy(optimize_mod, root_scalar_dispatch, fsolve_dispatch, linprog_dispatch)
        if optimize_mod is not None and root_scalar_dispatch is not None and fsolve_dispatch is not None and linprog_dispatch is not None
        else None
    )

    real_rgi = getattr(interpolate_mod, "RegularGridInterpolator", None) if interpolate_mod is not None else None
    rgi_class_proxy = _RegularGridInterpolatorClassProxy(real_rgi) if real_rgi is not None else None
    interpolate_proxy = (
        _InterpolateProxy(interpolate_mod, rgi_class_proxy)
        if interpolate_mod is not None and rgi_class_proxy is not None
        else None
    )

    real_eigh = getattr(linalg_mod, "eigh", None) if linalg_mod is not None else None
    eigh_dispatch = _make_linalg_eigh(real_eigh) if real_eigh is not None else None
    linalg_proxy = (
        _LinalgProxy(linalg_mod, eigh_dispatch)
        if linalg_mod is not None and eigh_dispatch is not None
        else None
    )

    scipy_proxy = (
        _ScipyProxy(scipy_mod, stats_proxy, special_proxy, optimize_proxy, interpolate_proxy, linalg_proxy)
        if scipy_mod is not None
        else None
    )

    for name, value in list(shadow.items()):
        if scipy_mod is not None and value is scipy_mod and scipy_proxy is not None:
            shadow[name] = scipy_proxy
            continue
        if stats_mod is not None and value is stats_mod and stats_proxy is not None:
            shadow[name] = stats_proxy
            continue
        if special_mod is not None and value is special_mod and special_proxy is not None:
            shadow[name] = special_proxy
            continue
        if optimize_mod is not None and value is optimize_mod and optimize_proxy is not None:
            shadow[name] = optimize_proxy
            continue
        if interpolate_mod is not None and value is interpolate_mod and interpolate_proxy is not None:
            shadow[name] = interpolate_proxy
            continue
        if linalg_mod is not None and value is linalg_mod and linalg_proxy is not None:
            shadow[name] = linalg_proxy
            continue
        if real_rgi is not None and value is real_rgi and rgi_class_proxy is not None:
            shadow[name] = rgi_class_proxy
            continue
        if real_eigh is not None and value is real_eigh and eigh_dispatch is not None:
            shadow[name] = eigh_dispatch
            continue
        if real_rgi is not None:
            try:
                is_rgi_instance = isinstance(value, real_rgi)
            except TypeError:
                is_rgi_instance = False
            if is_rgi_instance:
                from .interpolation import RegularGridInterpolatorProxy
                shadow[name] = RegularGridInterpolatorProxy(value)
                continue
        if real_root_scalar is not None and value is real_root_scalar and root_scalar_dispatch is not None:
            shadow[name] = root_scalar_dispatch
            continue
        if real_fsolve is not None and value is real_fsolve and fsolve_dispatch is not None:
            shadow[name] = fsolve_dispatch
            continue
        if real_linprog is not None and value is real_linprog and linprog_dispatch is not None:
            shadow[name] = linprog_dispatch
            continue
        if real_norm is not None and value is real_norm and norm_proxy is not None:
            shadow[name] = norm_proxy
            continue
        if real_Normal is not None and value is real_Normal and Normal_proxy is not None:
            shadow[name] = Normal_proxy
            continue
        if real_erf is not None and value is real_erf and erf_dispatch is not None:
            shadow[name] = erf_dispatch
            continue
        if real_ndtr is not None and value is real_ndtr and ndtr_dispatch is not None:
            shadow[name] = ndtr_dispatch
            continue

        # Direct aliases such as ``pdf = norm.pdf`` and ``cdf = norm.cdf``.
        bound_self = getattr(value, "__self__", None)
        bound_name = getattr(value, "__name__", None)
        if real_norm is not None and bound_self is real_norm and bound_name in {"pdf", "cdf"}:
            shadow[name] = getattr(norm_proxy, bound_name)
            continue

        # Frozen ``rv = norm(loc=..., scale=...)`` objects and method aliases.
        frozen_self = bound_self if bound_self is not None else value
        if real_norm is not None:
            frozen_proxy = _frozen_norm_proxy_for(frozen_self, real_norm)
            if frozen_proxy is not None:
                if bound_self is not None and bound_name in {"pdf", "cdf"}:
                    shadow[name] = getattr(frozen_proxy, bound_name)
                elif bound_self is None:
                    shadow[name] = frozen_proxy
                continue

        # New ``scipy.stats.Normal`` instances and direct pdf/cdf method aliases.
        if real_Normal is not None:
            candidate = bound_self if bound_self is not None else value
            try:
                is_normal_instance = isinstance(candidate, real_Normal)
            except TypeError:
                is_normal_instance = False
            if is_normal_instance:
                instance_proxy = _NormalInstanceProxy(
                    candidate, float(candidate.mu), float(candidate.sigma)
                )
                if bound_self is not None and bound_name in {"pdf", "cdf"}:
                    shadow[name] = getattr(instance_proxy, bound_name)
                elif bound_self is None:
                    shadow[name] = instance_proxy


__all__ = ["apply_scipy_shadow_replacements"]
