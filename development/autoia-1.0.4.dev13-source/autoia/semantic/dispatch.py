"""Local semantic-dispatch execution for ordinary Python engineering functions.

This module replaces the research prototype's process-global monkeypatch of
``numpy.linalg.solve``.  It executes a Python function against a *shadow copy*
of its global namespace in which NumPy references are replaced by lightweight
proxies.  The original function object, its original globals dictionary, and
NumPy itself are never mutated.

Scope of the current reference dispatcher
----------------------------------------
The local dispatch is intentionally conservative and auditable.  It handles:

* module-level ``import numpy as np`` (including aliases),
* module-level ``import numpy.linalg as la`` (including aliases),
* module-level ``from numpy.linalg import solve`` (including aliases),
* module-level NumPy or :mod:`math` ``sqrt``, ``exp``, ``log``, ``log10``, ``sin`` and
  ``cos`` calls and direct aliases of those functions,
* ``numpy.power`` / ``math.pow`` with a validated base and one constant real
  exponent, and
* ordinary Python helper functions defined in the same global namespace as the
  model. Those helpers are cloned to use the same shadow globals,
* imported pure-Python helper functions. These are wrapped locally and prepared
  lazily only when called with AutoIA semantic values,
* pure-Python functions and user objects captured in closure cells; their reached
  methods are prepared lazily without publishing module globals, and
* common NumPy array construction, reductions, dot/matmul, and vector/Frobenius
  norm forms used by Step 4.

A local ``import numpy`` executed *inside* a function body and compiled-extension
call paths remain outside the current shadow-namespace dispatcher. Imported
pure-Python helpers are supported without mutating their source modules or globals.
"""

from __future__ import annotations

from functools import partial
import math
import types
from contextvars import ContextVar
from typing import Any, Callable

import numpy as np

from ..arrays import (
    contains_validated as array_contains_validated,
    validated_sum, validated_prod, validated_mean,
    validated_dot, validated_matmul, validated_inner, validated_outer,
    validated_trace, validated_norm,
)
from .scipy_dispatch import apply_scipy_shadow_replacements
from .spd_matrix import validated_slogdet, validated_det



_DEFERRED_LINALG_NAMES = {"eig", "eigvals", "eigh", "eigvalsh"}


def _make_spd_linalg(name: str, real_func):
    """Validated SPD-only NumPy linalg semantic with numeric fallthrough."""
    def traced(*args, **kwargs):
        if any(array_contains_validated(arg) for arg in args):
            if name == "slogdet":
                return validated_slogdet(*args, **kwargs)
            if name == "det":
                return validated_det(*args, **kwargs)
        return real_func(*args, **kwargs)
    traced.__name__ = getattr(real_func, "__name__", name)
    traced.__doc__ = getattr(real_func, "__doc__", None)
    return traced


def _make_deferred_linalg(name: str, real_func):
    """Fail explicitly when a deferred NumPy linalg primitive sees validated data.

    Ordinary numeric calls still go directly to NumPy.  This is a diagnostics
    boundary, not a semantic implementation of the operation.
    """
    def traced(*args, **kwargs):
        if any(array_contains_validated(arg) for arg in args):
            raise NotImplementedError(
                f"AutoIA Version-1 does not support np.linalg.{name} on "
                "validated uncertain values. Eigenvalue/eigenvector semantic "
                "primitives are currently deferred."
            )
        return real_func(*args, **kwargs)
    traced.__name__ = getattr(real_func, "__name__", name)
    traced.__doc__ = getattr(real_func, "__doc__", None)
    return traced


class _LinalgProxy:
    """Read-only proxy for ``numpy.linalg`` with local validated dispatch."""

    __slots__ = ("_target", "_solve", "_norm", "_deferred", "_spd")

    def __init__(self, target, solve: Callable[..., Any], norm: Callable[..., Any]):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_solve", solve)
        object.__setattr__(self, "_norm", norm)
        object.__setattr__(self, "_deferred", {
            name: _make_deferred_linalg(name, getattr(target, name))
            for name in _DEFERRED_LINALG_NAMES
        })
        object.__setattr__(self, "_spd", {
            name: _make_spd_linalg(name, getattr(target, name))
            for name in ("slogdet", "det")
        })

    def __getattr__(self, name: str):
        if name == "solve":
            return object.__getattribute__(self, "_solve")
        if name == "norm":
            return object.__getattribute__(self, "_norm")
        spd = object.__getattribute__(self, "_spd")
        if name in spd:
            return spd[name]
        deferred = object.__getattribute__(self, "_deferred")
        if name in deferred:
            return deferred[name]
        return getattr(object.__getattribute__(self, "_target"), name)

    def __setattr__(self, name, value):  # pragma: no cover - defensive
        raise AttributeError("AutoIA NumPy dispatch proxies are read-only.")

    def __dir__(self):
        return sorted(
            set(dir(object.__getattribute__(self, "_target")))
            | {"solve", "norm"}
            | _DEFERRED_LINALG_NAMES
            | {"slogdet", "det"}
        )




def _make_object_constructor(name: str, real_func):
    """Create an array constructor that defaults to dtype=object in traced code.

    This enables common engineering assembly patterns such as ``A=np.zeros(...);
    A[i,j]=uncertain_value`` without mutating real NumPy or requiring the user to
    add ``dtype=object`` manually.  An explicit user ``dtype`` is respected.
    """
    def traced(*args, **kwargs):
        if "dtype" not in kwargs or kwargs.get("dtype") is None:
            kwargs = dict(kwargs)
            kwargs["dtype"] = object
        return real_func(*args, **kwargs)
    traced.__name__ = getattr(real_func, "__name__", name)
    traced.__doc__ = getattr(real_func, "__doc__", None)
    return traced


def _make_validated_array_function(name: str, real_func, validated_func):
    """Dispatch a NumPy array operation only when validated values are present."""
    def traced(*args, **kwargs):
        # ``out`` arrays can themselves contain validated values, but the current
        # contract rejects validated out= semantics inside the validated helper.
        if any(array_contains_validated(arg) for arg in args):
            return validated_func(*args, **kwargs)
        return real_func(*args, **kwargs)
    traced.__name__ = getattr(real_func, "__name__", name)
    traced.__doc__ = getattr(real_func, "__doc__", None)
    return traced


def _make_traced_norm(real_norm):
    def traced(x, ord=None, axis=None, keepdims=False):
        if array_contains_validated(x):
            return validated_norm(x, ord=ord, axis=axis, keepdims=keepdims)
        return real_norm(x, ord=ord, axis=axis, keepdims=keepdims)
    traced.__name__ = getattr(real_norm, "__name__", "norm")
    traced.__doc__ = getattr(real_norm, "__doc__", None)
    return traced

_ELEMENTARY_NUMPY_NAMES = (
    "sqrt", "exp", "log", "log10", "log1p",
    "sin", "cos", "sinh", "cosh", "tanh", "arctan", "atan", "abs", "absolute",
)
_ELEMENTARY_MATH_NAMES = (
    "sqrt", "exp", "log", "log10", "log1p",
    "sin", "cos", "sinh", "cosh", "tanh", "atan", "erf",
)


def _make_traced_unary(name: str, real_func):
    """Create a local NumPy-like unary callable for validated values."""
    def traced(x, *args, **kwargs):
        from ..enclosures import ValidatedValue

        dual_unary = getattr(x, "__autoia_unary__", None)
        if callable(dual_unary):
            if args or kwargs:
                raise NotImplementedError(
                    f"AutoIA validated np.{name} currently supports only the "
                    "ordinary single-argument form."
                )
            return dual_unary(name)

        if isinstance(x, ValidatedValue):
            if args or kwargs:
                raise NotImplementedError(
                    f"AutoIA validated np.{name} currently supports only the "
                    "ordinary single-argument form."
                )
            return x.unary(name)

        obj = np.asarray(x, dtype=object)
        has_validated = any(isinstance(v, ValidatedValue) for v in obj.flat)
        if not has_validated:
            return real_func(x, *args, **kwargs)
        if args or kwargs:
            raise NotImplementedError(
                f"AutoIA validated np.{name} currently supports only the "
                "ordinary single-argument form."
            )
        out = np.empty(obj.shape, dtype=object)
        for idx in np.ndindex(obj.shape):
            v = obj[idx]
            out[idx] = v.unary(name) if isinstance(v, ValidatedValue) else real_func(v)
        return out.item() if out.ndim == 0 else out

    traced.__name__ = getattr(real_func, "__name__", name)
    traced.__doc__ = getattr(real_func, "__doc__", None)
    return traced


def _make_traced_power(real_func, *, allow_object_array: bool):
    """Create a local power callable for validated bases and constant exponents."""
    def traced(base, exponent, *args, **kwargs):
        from ..enclosures import ValidatedValue

        if getattr(base, "__autoia_semantic_dual__", False):
            if args or kwargs:
                raise NotImplementedError(
                    "Validated power currently supports only the ordinary two-argument form."
                )
            return base ** exponent

        if isinstance(exponent, ValidatedValue) or getattr(exponent, "__autoia_semantic_dual__", False):
            raise NotImplementedError(
                "AutoIA does not yet support uncertain exponents; the exponent "
                "must be a constant real scalar."
            )
        if isinstance(base, ValidatedValue):
            if args or kwargs:
                raise NotImplementedError(
                    "Validated power currently supports only the ordinary two-argument form."
                )
            return base ** exponent

        if allow_object_array:
            obj = np.asarray(base, dtype=object)
            has_validated = any(isinstance(v, ValidatedValue) for v in obj.flat)
            if has_validated:
                if args or kwargs:
                    raise NotImplementedError(
                        "Validated np.power currently supports only the ordinary "
                        "two-argument form."
                    )
                exp_obj = np.asarray(exponent, dtype=object)
                if exp_obj.ndim != 0:
                    raise NotImplementedError(
                        "AutoIA validated np.power currently requires one constant "
                        "scalar exponent; exponent arrays are deferred to Step 4."
                    )
                p = exp_obj.item()
                out = np.empty(obj.shape, dtype=object)
                for idx in np.ndindex(obj.shape):
                    v = obj[idx]
                    out[idx] = v ** p if isinstance(v, ValidatedValue) else real_func(v, p)
                return out.item() if out.ndim == 0 else out

        return real_func(base, exponent, *args, **kwargs)

    traced.__name__ = getattr(real_func, "__name__", "power")
    traced.__doc__ = getattr(real_func, "__doc__", None)
    return traced


class _NumpyProxy:
    """Read-only NumPy proxy with local validated-operation dispatch."""

    __slots__ = ("_target", "_linalg", "_unary", "_power", "_array_dispatch")

    def __init__(self, target, linalg_proxy, unary_dispatch, power_dispatch, array_dispatch):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_linalg", linalg_proxy)
        object.__setattr__(self, "_unary", dict(unary_dispatch))
        object.__setattr__(self, "_power", power_dispatch)
        object.__setattr__(self, "_array_dispatch", dict(array_dispatch))

    def __getattr__(self, name: str):
        if name == "linalg":
            return object.__getattribute__(self, "_linalg")
        unary = object.__getattribute__(self, "_unary")
        if name in unary:
            return unary[name]
        if name == "power":
            return object.__getattribute__(self, "_power")
        arr = object.__getattribute__(self, "_array_dispatch")
        if name in arr:
            return arr[name]
        return getattr(object.__getattribute__(self, "_target"), name)

    def __setattr__(self, name, value):  # pragma: no cover - defensive
        raise AttributeError("AutoIA NumPy dispatch proxies are read-only.")

    def __dir__(self):
        return sorted(
            set(dir(object.__getattribute__(self, "_target")))
            | set(_ELEMENTARY_NUMPY_NAMES)
            | {"power"}
            | set(object.__getattribute__(self, "_array_dispatch"))
        )



class _MathProxy:
    """Read-only proxy for :mod:`math` with validated unary dispatch."""

    __slots__ = ("_target", "_unary", "_power")

    def __init__(self, target, unary_dispatch, power_dispatch):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_unary", dict(unary_dispatch))
        object.__setattr__(self, "_power", power_dispatch)

    def __getattr__(self, name: str):
        unary = object.__getattribute__(self, "_unary")
        if name in unary:
            return unary[name]
        if name == "pow":
            return object.__getattribute__(self, "_power")
        return getattr(object.__getattribute__(self, "_target"), name)

    def __setattr__(self, name, value):  # pragma: no cover - defensive
        raise AttributeError("AutoIA math dispatch proxies are read-only.")

    def __dir__(self):
        return sorted(set(dir(object.__getattribute__(self, "_target"))) | set(_ELEMENTARY_MATH_NAMES) | {"pow"})

def _contains_dispatch_value(value: Any) -> bool:
    """Return whether ``value`` carries AutoIA semantic state.

    This deliberately recognizes both ordinary ``ValidatedValue`` instances and
    the internal implicit-differentiation dual values. The latter matter when an
    imported pure-Python helper is called while certifying a nonlinear solve.
    """
    from ..enclosures import ValidatedValue

    if isinstance(value, ValidatedValue) or getattr(value, "__autoia_semantic_dual__", False):
        return True
    if isinstance(value, dict):
        return any(_contains_dispatch_value(k) or _contains_dispatch_value(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return any(_contains_dispatch_value(v) for v in value)
    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return False
        return any(_contains_dispatch_value(v) for v in value.flat)
    return False


def _function_closure_contains_dispatch_value(fn: types.FunctionType) -> bool:
    closure = getattr(fn, "__closure__", None)
    if not closure:
        return False
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:  # empty cell
            continue
        if _contains_dispatch_value(value):
            return True
    return False


def _make_imported_python_helper(fn: types.FunctionType, traced_solve: Callable[..., Any]):
    """Create a non-mutating lazy dispatcher for an imported Python helper.

    Ordinary numeric calls preserve the original function exactly. If AutoIA
    semantic values reach the helper, a private prepared clone of that helper is
    created and cached. Its own module globals receive the same local NumPy/SciPy
    shadow replacements, so helper modules can contain root solves and same-module
    residual functions without requiring user rewrites.
    """
    prepared = None

    def traced(*args, **kwargs):
        nonlocal prepared
        active = (
            any(_contains_dispatch_value(v) for v in args)
            or any(_contains_dispatch_value(v) for v in kwargs.values())
            or _function_closure_contains_dispatch_value(fn)
        )
        if not active:
            return fn(*args, **kwargs)
        if prepared is None:
            prepared = prepare_model_for_dispatch(fn, traced_solve)
        return prepared(*args, **kwargs)

    traced.__name__ = getattr(fn, "__name__", "autoia_imported_helper")
    traced.__doc__ = getattr(fn, "__doc__", None)
    traced.__module__ = getattr(fn, "__module__", None)
    traced.__qualname__ = getattr(fn, "__qualname__", traced.__name__)
    traced.__autoia_original_function__ = getattr(fn, "__autoia_original_function__", fn)
    traced.__autoia_imported_helper_dispatch__ = True
    return traced


class _PythonModuleProxy:
    """Read-only local proxy for an imported pure-Python helper module.

    Attribute access is delegated unchanged except for pure-Python functions.
    Those functions receive the same lazy, non-mutating dispatch wrapper used
    for directly imported helpers. This lets ordinary code call
    ``module.helper(...)`` without preparing or mutating the source module.
    """

    __slots__ = ("_target", "_traced_solve", "_cache")

    def __init__(self, target: types.ModuleType, traced_solve: Callable[..., Any]):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_traced_solve", traced_solve)
        object.__setattr__(self, "_cache", {})

    def __getattr__(self, name: str):
        target = object.__getattribute__(self, "_target")
        value = getattr(target, name)
        if not isinstance(value, types.FunctionType):
            return value
        cache = object.__getattribute__(self, "_cache")
        if name not in cache or getattr(cache[name], "__autoia_original_function__", None) is not value:
            cache[name] = _make_imported_python_helper(
                value, object.__getattribute__(self, "_traced_solve")
            )
        return cache[name]

    def __setattr__(self, name, value):  # pragma: no cover - defensive
        raise AttributeError("AutoIA imported-module dispatch proxies are read-only.")

    def __dir__(self):
        return dir(object.__getattribute__(self, "_target"))




class _PythonObjectProxy:
    """Local proxy for a source-level Python object captured by a closure.

    The proxy leaves data attributes on the original object untouched.  Pure-Python
    bound methods are prepared lazily when AutoIA semantic values reach them, and
    are bound to the proxy so nested ``self.other_method(...)`` calls pass through
    the same local dispatch boundary.  This is intentionally limited to ordinary
    source-level user classes; NumPy/SciPy/Torch/AutoIA objects retain their
    dedicated semantics.
    """

    __slots__ = ("_target", "_traced_solve", "_cache")

    def __init__(self, target: Any, traced_solve: Callable[..., Any]):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_traced_solve", traced_solve)
        object.__setattr__(self, "_cache", {})

    def __getattr__(self, name: str):
        target = object.__getattribute__(self, "_target")
        value = getattr(target, name)
        cache = object.__getattribute__(self, "_cache")
        traced_solve = object.__getattribute__(self, "_traced_solve")

        if isinstance(value, types.MethodType) and isinstance(value.__func__, types.FunctionType):
            key = (name, "method", value.__func__)
            if key not in cache:
                cache[key] = _make_bound_python_helper(value, self, traced_solve)
            return cache[key]

        # Functions stored directly on a captured object (``obj.g = response``)
        # are plain FunctionType values rather than bound methods.  Give them
        # the same private, lazy semantic preparation as a captured function.
        if isinstance(value, types.FunctionType):
            key = (name, "function", value)
            if key not in cache:
                cache[key] = _make_imported_python_helper(value, traced_solve)
            return cache[key]

        return value

    def __setattr__(self, name, value):
        # Preserve ordinary object semantics for code that intentionally mutates
        # state during execution.  AutoIA does not introduce extra global state.
        setattr(object.__getattribute__(self, "_target"), name, value)

    def __dir__(self):
        return dir(object.__getattribute__(self, "_target"))


def _is_proxyable_python_object(value: Any) -> bool:
    if isinstance(value, (types.ModuleType, types.FunctionType, types.MethodType, type)):
        return False
    if isinstance(value, (str, bytes, bytearray, int, float, complex, bool, tuple, list, dict, set, frozenset, np.ndarray, np.generic)):
        return False
    cls = type(value)
    module = str(getattr(cls, "__module__", ""))
    denied = ("builtins", "numpy", "scipy", "torch", "gpytorch", "botorch", "sklearn", "autoia")
    if module == "builtins" or any(module == p or module.startswith(p + ".") for p in denied[1:]):
        return False
    # Require at least one ordinary Python method in the class hierarchy OR a
    # pure-Python callable stored directly on the instance.
    for base in getattr(cls, "__mro__", (cls,)):
        for attr in getattr(base, "__dict__", {}).values():
            if isinstance(attr, types.FunctionType):
                return True
    try:
        instance_values = vars(value).values()
    except TypeError:
        instance_values = ()
    for attr in instance_values:
        if isinstance(attr, (types.FunctionType, types.MethodType, partial)):
            return True
    return False


def _make_bound_python_helper(
    bound_method: types.MethodType,
    owner_proxy: _PythonObjectProxy,
    traced_solve: Callable[..., Any],
):
    """Lazy local dispatcher for a pure-Python bound method."""
    prepared = None

    def traced(*args, **kwargs):
        nonlocal prepared
        active = (
            any(_contains_dispatch_value(v) for v in args)
            or any(_contains_dispatch_value(v) for v in kwargs.values())
            or _function_closure_contains_dispatch_value(bound_method.__func__)
        )
        if not active:
            return bound_method(*args, **kwargs)
        if prepared is None:
            _, clone_func = _shadow_globals_for_function(bound_method.__func__, traced_solve)
            prepared = types.MethodType(clone_func, owner_proxy)
        return prepared(*args, **kwargs)

    traced.__name__ = getattr(bound_method, "__name__", "autoia_bound_helper")
    traced.__doc__ = getattr(bound_method, "__doc__", None)
    traced.__qualname__ = getattr(bound_method, "__qualname__", traced.__name__)
    traced.__autoia_original_function__ = getattr(bound_method, "__func__", bound_method)
    traced.__autoia_imported_helper_dispatch__ = True
    return traced

def _is_proxyable_python_module(value: Any) -> bool:
    """Return whether an imported module is safe for local helper dispatch.

    Dedicated NumPy / SciPy / math semantics are handled elsewhere. For other
    modules, only source-backed Python modules are proxied. Compiled extension
    modules remain untouched.
    """
    if not isinstance(value, types.ModuleType):
        return False
    name = str(getattr(value, "__name__", ""))
    if name == "math" or name == "numpy" or name.startswith("numpy."):
        return False
    if name == "scipy" or name.startswith("scipy."):
        return False
    if name == "torch" or name.startswith("torch."):
        return False
    if name == "autoia" or name.startswith("autoia."):
        return False
    filename = getattr(value, "__file__", None)
    if not filename:
        return False
    filename = str(filename).lower()
    return filename.endswith((".py", ".pyc"))


def _copy_function_metadata(dst: types.FunctionType, src: types.FunctionType) -> None:
    """Copy non-code metadata without altering the executable semantics."""
    dst.__kwdefaults__ = src.__kwdefaults__
    dst.__annotations__ = dict(getattr(src, "__annotations__", {}))
    dst.__dict__.update(getattr(src, "__dict__", {}))
    dst.__module__ = src.__module__
    dst.__qualname__ = src.__qualname__
    dst.__doc__ = src.__doc__


def _clone_function_with_globals(
    fn: types.FunctionType,
    shadow_globals: dict[str, Any],
    *, closure=None,
) -> types.FunctionType:
    clone = types.FunctionType(
        fn.__code__,
        shadow_globals,
        name=fn.__name__,
        argdefs=fn.__defaults__,
        closure=fn.__closure__ if closure is None else closure,
    )
    _copy_function_metadata(clone, fn)
    clone.__autoia_original_function__ = getattr(fn, "__autoia_original_function__", fn)
    clone.__autoia_dispatch_prepared__ = True
    return clone


def _make_cell(value):
    def _cell_factory():
        return value
    return _cell_factory.__closure__[0]


def _dispatch_wrapped_closure(
    fn: types.FunctionType,
    traced_solve: Callable[..., Any],
):
    """Return closure cells with local semantic dispatch where appropriate.

    Historically only fitted-GP objects captured in closure cells were wrapped.
    Ordinary function factories such as ``fn = response; def model(...): fn(...)``
    therefore bypassed the private shadow namespace.  The same happened for
    captured user objects such as ``prob.g(...)``.  This routine extends the
    non-mutating shadow model to those common Python closure patterns.
    """
    closure = fn.__closure__
    if not closure:
        return closure
    from ..integrations.gp import (
        maybe_wrap_gp_object, maybe_make_torch_proxy, maybe_make_gp_construction_proxy,
    )
    changed = False
    cells = []
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            cells.append(cell)
            continue

        wrapped = maybe_wrap_gp_object(value)
        if wrapped is value:
            wrapped = maybe_make_gp_construction_proxy(value)
        if wrapped is value:
            wrapped = maybe_make_torch_proxy(value)

        # Pure-Python function captured by a factory/closure.  Prepare a private
        # clone directly.  A lazy helper wrapper here is tempting, but recognized
        # SciPy solvers may themselves receive the captured function as a callback;
        # wrapping that wrapper again during callback preparation creates a
        # recursive dispatch chain.  A direct clone preserves ordinary numeric
        # fallthrough through its local proxies and composes cleanly with solver
        # callback preparation.
        if wrapped is value and isinstance(value, types.FunctionType):
            _, wrapped = _shadow_globals_for_function(value, traced_solve)

        # A directly captured bound method gets a proxy owner so nested method
        # calls can also cross the dispatch boundary.
        if wrapped is value and isinstance(value, types.MethodType) and isinstance(value.__func__, types.FunctionType):
            owner = _PythonObjectProxy(value.__self__, traced_solve)
            wrapped = _make_bound_python_helper(value, owner, traced_solve)

        if wrapped is value and _is_proxyable_python_module(value):
            wrapped = _PythonModuleProxy(value, traced_solve)

        if wrapped is value and _is_proxyable_python_object(value):
            wrapped = _PythonObjectProxy(value, traced_solve)

        if wrapped is value:
            cells.append(cell)
        else:
            cells.append(_make_cell(wrapped))
            changed = True
    return tuple(cells) if changed else closure


def _shadow_globals_for_function(
    fn: types.FunctionType,
    traced_solve: Callable[..., Any],
) -> tuple[dict[str, Any], types.FunctionType]:
    """Create a non-mutating execution namespace for ``fn``.

    All ordinary Python functions sharing exactly the same original globals
    dictionary are cloned in a second pass so helper-to-helper calls remain in
    the shadow namespace as well.
    """
    original_globals = fn.__globals__
    original_solve = np.linalg.solve

    norm_dispatch = _make_traced_norm(np.linalg.norm)
    linalg_proxy = _LinalgProxy(np.linalg, traced_solve, norm_dispatch)
    unary_dispatch = {
        name: _make_traced_unary("arctan" if name == "atan" else name, getattr(np, name))
        for name in _ELEMENTARY_NUMPY_NAMES
    }
    numpy_power_dispatch = _make_traced_power(np.power, allow_object_array=True)

    array_dispatch = {
        # Constructors: object dtype by default so ordinary matrix/vector
        # assembly can later receive validated scalar assignments.
        name: _make_object_constructor(name, getattr(np, name))
        for name in (
            "array", "asarray", "zeros", "ones", "empty", "full",
            "zeros_like", "ones_like", "empty_like", "full_like",
            "eye", "identity",
        )
    }
    array_dispatch.update({
        "sum": _make_validated_array_function("sum", np.sum, validated_sum),
        "prod": _make_validated_array_function("prod", np.prod, validated_prod),
        "mean": _make_validated_array_function("mean", np.mean, validated_mean),
        "dot": _make_validated_array_function("dot", np.dot, validated_dot),
        "matmul": _make_validated_array_function("matmul", np.matmul, validated_matmul),
        "inner": _make_validated_array_function("inner", np.inner, validated_inner),
        "outer": _make_validated_array_function("outer", np.outer, validated_outer),
        "trace": _make_validated_array_function("trace", np.trace, validated_trace),
    })
    numpy_proxy = _NumpyProxy(
        np, linalg_proxy, unary_dispatch, numpy_power_dispatch, array_dispatch
    )
    math_unary_dispatch = {
        name: _make_traced_unary("arctan" if name == "atan" else name, getattr(math, name))
        for name in _ELEMENTARY_MATH_NAMES
    }
    math_power_dispatch = _make_traced_power(math.pow, allow_object_array=False)
    math_proxy = _MathProxy(math, math_unary_dispatch, math_power_dispatch)

    shadow = dict(original_globals)

    # Narrow optional fitted-GP semantics.  A recognized fitted GP object is
    # wrapped only in this private execution namespace; numeric calls still
    # delegate to the original model.  The torch module receives a query-only
    # proxy so ordinary ``torch.tensor`` / ``torch.stack`` construction can carry
    # AutoIA values to that recognized GP without implementing arbitrary Torch.
    from ..integrations.gp import (
        maybe_wrap_gp_object, maybe_make_torch_proxy, maybe_make_gp_construction_proxy,
    )
    for _name, _value in list(shadow.items()):
        _wrapped = maybe_wrap_gp_object(_value)
        if _wrapped is not _value:
            shadow[_name] = _wrapped
            continue
        _gp_construct = maybe_make_gp_construction_proxy(_value)
        if _gp_construct is not _value:
            shadow[_name] = _gp_construct
            continue
        _torch_proxy = maybe_make_torch_proxy(_value)
        if _torch_proxy is not _value:
            shadow[_name] = _torch_proxy

    # Replace common NumPy/math import styles without touching the originals.
    real_unary = {getattr(np, name): unary_dispatch[name] for name in _ELEMENTARY_NUMPY_NAMES}
    real_unary.update({
        getattr(math, name): math_unary_dispatch[name]
        for name in _ELEMENTARY_MATH_NAMES
    })
    real_special = [
        (np.power, numpy_power_dispatch),
        (math.pow, math_power_dispatch),
        (np.linalg.norm, norm_dispatch),
    ]
    real_special.extend(
        (getattr(np.linalg, name), getattr(linalg_proxy, name))
        for name in ("slogdet", "det")
    )
    real_special.extend(
        (getattr(np.linalg, name), getattr(linalg_proxy, name))
        for name in sorted(_DEFERRED_LINALG_NAMES)
    )
    real_special.extend(
        (getattr(np, name), traced) for name, traced in array_dispatch.items()
    )
    for name, value in list(shadow.items()):
        if value is np:
            shadow[name] = numpy_proxy
        elif value is math:
            shadow[name] = math_proxy
        elif value is np.linalg:
            shadow[name] = linalg_proxy
        elif value is original_solve:
            shadow[name] = traced_solve
        else:
            matched = False
            for real_func, traced_func in real_special:
                if value is real_func:
                    shadow[name] = traced_func
                    matched = True
                    break
            if matched:
                continue
            for real_func, traced_func in real_unary.items():
                if value is real_func:
                    shadow[name] = traced_func
                    break

    # Replace common SciPy normal/special-function import styles in the same
    # private shadow namespace. SciPy itself is never mutated and is not an
    # AutoIA package dependency.
    apply_scipy_shadow_replacements(shadow)

    # Proxy imported pure-Python *modules* locally as well. This covers the very
    # common engineering pattern ``import mymodule; mymodule.helper(...)``. The
    # source module is not mutated; only function attributes actually reached by
    # the proof receive lazy local preparation. Dedicated NumPy/SciPy/math
    # proxies above remain authoritative.
    for name, value in list(original_globals.items()):
        if _is_proxyable_python_module(value) and shadow.get(name) is value:
            shadow[name] = _PythonModuleProxy(value, traced_solve)

    # Clone same-namespace Python helpers so they resolve NumPy through the same
    # shadow dictionary. Imported pure-Python helpers get a lazy local wrapper:
    # numeric calls remain untouched, while calls carrying AutoIA semantic values
    # prepare the helper in its own private shadow namespace. Recognized NumPy /
    # SciPy replacements above are never overwritten.
    for name, value in list(original_globals.items()):
        if not isinstance(value, types.FunctionType):
            continue
        if value.__globals__ is original_globals:
            shadow[name] = _clone_function_with_globals(value, shadow, closure=_dispatch_wrapped_closure(value, traced_solve))
        elif shadow.get(name) is value:
            shadow[name] = _make_imported_python_helper(value, traced_solve)

    cloned_model = _clone_function_with_globals(fn, shadow, closure=_dispatch_wrapped_closure(fn, traced_solve))
    # Ensure recursive calls through the function's own global name use the same
    # clone when that name exists in the original module namespace.
    if original_globals.get(fn.__name__) is fn:
        shadow[fn.__name__] = cloned_model
    return shadow, cloned_model


def prepare_model_for_dispatch(model, traced_solve: Callable[..., Any]):
    """Return a callable whose NumPy solve references dispatch locally.

    No global object is modified.  Python functions and bound Python methods are
    supported directly; ``functools.partial`` is handled recursively.  A wrapper
    may provide ``__autoia_prepare_dispatch__`` to preserve wrapper semantics
    while preparing its underlying user function.  Other callable objects are
    returned unchanged because the current dispatcher cannot safely rewrite the globals of
    arbitrary ``__call__`` implementations.
    """
    prepare_hook = getattr(model, "__autoia_prepare_dispatch__", None)
    if callable(prepare_hook):
        return prepare_hook(traced_solve)

    if isinstance(model, types.FunctionType):
        _, clone = _shadow_globals_for_function(model, traced_solve)
        return clone

    if isinstance(model, types.MethodType) and isinstance(model.__func__, types.FunctionType):
        _, clone_func = _shadow_globals_for_function(model.__func__, traced_solve)
        return types.MethodType(clone_func, model.__self__)

    if isinstance(model, partial):
        new_func = prepare_model_for_dispatch(model.func, traced_solve)
        return partial(new_func, *model.args, **(model.keywords or {}))

    return model



_ACTIVE_PREPARED_SOLVE_DISPATCHER: ContextVar[object | None] = ContextVar(
    "autoia_active_prepared_solve_dispatcher", default=None
)


def _prepared_traced_solve(A, b, *args, **kwargs):
    """Stable solve hook used by a prepared model.

    The engineering function is cloned only once with this callable installed in
    its private shadow namespace.  Each certification sets the active semantic
    solve dispatcher context-locally before execution, so prepared models remain
    safe for nested calls and concurrent threads without global monkeypatching.
    """
    dispatcher = _ACTIVE_PREPARED_SOLVE_DISPATCHER.get()
    if dispatcher is None:
        raise RuntimeError(
            "Prepared AutoIA model executed outside an active semantic solve context."
        )
    return dispatcher.traced_solve(A, b, *args, **kwargs)


class PreparedDispatchModel:
    """One-time local-dispatch preparation of an engineering callable."""

    __slots__ = ("original_model", "prepared_model")

    def __init__(self, model):
        self.original_model = model
        self.prepared_model = prepare_model_for_dispatch(model, _prepared_traced_solve)

    def execute(self, p, solve_dispatcher):
        token = _ACTIVE_PREPARED_SOLVE_DISPATCHER.set(solve_dispatcher)
        try:
            return self.prepared_model(p)
        finally:
            _ACTIVE_PREPARED_SOLVE_DISPATCHER.reset(token)


def prepare_dispatch_model(model):
    """Prepare local NumPy/SciPy semantic dispatch once for repeated execution."""
    if isinstance(model, PreparedDispatchModel):
        return model
    return PreparedDispatchModel(model)

def execute_model_with_dispatch(model, p, solve_dispatcher):
    """Execute ``model(p)`` with local semantic ``solve`` dispatch.

    ``solve_dispatcher`` is expected to provide ``traced_solve(A, b)``.  The
    dispatcher owns any certificates; this function owns only local execution
    namespace preparation.
    """
    if isinstance(model, PreparedDispatchModel):
        return model.execute(p, solve_dispatcher)
    prepared = prepare_model_for_dispatch(model, solve_dispatcher.traced_solve)
    return prepared(p)


__all__ = [
    "prepare_model_for_dispatch",
    "prepare_dispatch_model",
    "PreparedDispatchModel",
    "execute_model_with_dispatch",
]
