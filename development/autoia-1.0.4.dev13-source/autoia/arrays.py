"""Validated NumPy-style array operations for AutoIA.

AutoIA deliberately uses ordinary NumPy ``dtype=object`` arrays as the current
array container.  The uncertain scalar semantics remain owned by
:class:`autoia.enclosures.ValidatedValue`; this module supplies a small, explicit
set of array/reduction operations so supported behavior does not depend on
accidental NumPy object-array dispatch details.

The functions here are *composition* operations, not a new trusted numerical
kernel.  Whenever an element is a ``ValidatedValue``, arithmetic ultimately
routes through that value's validated scalar operations.
"""

from __future__ import annotations

import numpy as np

from .enclosures.domain import DomainRefinementRequired

from .enclosures import ValidatedValue, IntervalEnclosure
from .trusted import point_interval, real_power_interval, sqrt_interval, ElementaryDomainError


def contains_validated(obj) -> bool:
    """Return whether ``obj`` contains at least one :class:`ValidatedValue`."""
    if isinstance(obj, ValidatedValue):
        return True
    try:
        arr = np.asarray(obj, dtype=object)
    except Exception:
        return False
    return any(isinstance(v, ValidatedValue) for v in arr.flat)


def as_object_array(obj):
    """Return an object ndarray without copying unless NumPy requires it."""
    return np.asarray(obj, dtype=object)


def _reject_out(out, opname: str) -> None:
    if out is not None:
        raise NotImplementedError(
            f"AutoIA validated np.{opname} does not yet support the 'out' argument."
        )


def _validate_reduction_keywords(*, dtype, out, where, opname: str) -> None:
    _reject_out(out, opname)
    if dtype is not None and np.dtype(dtype) != np.dtype(object):
        raise NotImplementedError(
            f"AutoIA validated np.{opname} currently supports dtype=None or dtype=object only."
        )
    if not (where is True or (np.ndim(where) == 0 and bool(where))):
        raise NotImplementedError(
            f"AutoIA validated np.{opname} does not yet support a nontrivial 'where' mask."
        )


def validated_sum(a, axis=None, dtype=None, out=None, keepdims=False,
                  initial=None, where=True):
    """NumPy-like sum for arrays containing validated scalars."""
    _validate_reduction_keywords(dtype=dtype, out=out, where=where, opname="sum")
    obj = as_object_array(a)
    kwargs = dict(axis=axis, dtype=object, keepdims=keepdims)
    if initial is not None:
        kwargs["initial"] = initial
    return np.sum(obj, **kwargs)


def validated_prod(a, axis=None, dtype=None, out=None, keepdims=False,
                   initial=None, where=True):
    """NumPy-like product for arrays containing validated scalars."""
    _validate_reduction_keywords(dtype=dtype, out=out, where=where, opname="prod")
    obj = as_object_array(a)
    kwargs = dict(axis=axis, dtype=object, keepdims=keepdims)
    if initial is not None:
        kwargs["initial"] = initial
    return np.prod(obj, **kwargs)


def validated_mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    """NumPy-like arithmetic mean for arrays containing validated scalars."""
    _validate_reduction_keywords(dtype=dtype, out=out, where=where, opname="mean")
    obj = as_object_array(a)
    return np.mean(obj, axis=axis, dtype=object, keepdims=keepdims)


def _reject_linear_out(out, opname: str) -> None:
    if out is not None:
        raise NotImplementedError(
            f"AutoIA validated np.{opname} does not yet support the 'out' argument."
        )


def validated_dot(a, b, out=None):
    _reject_linear_out(out, "dot")
    return np.dot(as_object_array(a), as_object_array(b))


def validated_matmul(a, b, out=None):
    _reject_linear_out(out, "matmul")
    return np.matmul(as_object_array(a), as_object_array(b))


def validated_inner(a, b):
    return np.inner(as_object_array(a), as_object_array(b))


def validated_outer(a, b, out=None):
    _reject_linear_out(out, "outer")
    return np.outer(as_object_array(a), as_object_array(b))


def validated_trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    _reject_out(out, "trace")
    if dtype is not None and np.dtype(dtype) != np.dtype(object):
        raise NotImplementedError(
            "AutoIA validated np.trace currently supports dtype=None or dtype=object only."
        )
    return np.trace(
        as_object_array(a), offset=offset, axis1=axis1, axis2=axis2, dtype=object
    )


def _sqrt_validated_result(value):
    """Apply the already-certified scalar sqrt to a scalar/array result."""
    if isinstance(value, ValidatedValue):
        return value.unary("sqrt")
    arr = np.asarray(value, dtype=object)
    has = any(isinstance(v, ValidatedValue) for v in arr.flat)
    if not has:
        return np.sqrt(value)
    out = np.empty(arr.shape, dtype=object)
    for idx in np.ndindex(arr.shape):
        v = arr[idx]
        out[idx] = v.unary("sqrt") if isinstance(v, ValidatedValue) else np.sqrt(v)
    return out.item() if out.ndim == 0 else out


def validated_norm(x, ord=None, axis=None, keepdims=False):
    """Validated common vector/Frobenius norms.

    Supported validated cases are deliberately those expressible as

        sqrt(sum(x_i**2)).

    Specifically:

    * ``ord=None`` for vectors and matrices (Euclidean/Frobenius semantics),
    * ``ord=2`` for vector norms (including a single ``axis``), and
    * ``ord='fro'`` or ``ord='f'`` for matrix/Frobenius norms.

    Matrix spectral 2-norms, 1/inf norms, nuclear norms, and general p-norms are
    deferred because they require additional nonsmooth/eigenvalue semantics.
    """
    obj = as_object_array(x)

    if not contains_validated(obj):
        return np.linalg.norm(np.asarray(x), ord=ord, axis=axis, keepdims=keepdims)

    if ord is None:
        # NumPy: vector 2-norm for 1D, Frobenius for 2D; for explicit axes the
        # same sum-of-squares formula is the natural supported subset.
        pass
    elif ord in ("fro", "f"):
        if axis is None and obj.ndim != 2:
            raise ValueError("Frobenius norm requires a matrix or a two-axis selection.")
        if axis is not None:
            axes = (axis,) if np.isscalar(axis) else tuple(axis)
            if len(axes) != 2:
                raise ValueError("Frobenius norm with axis requires exactly two axes.")
    elif ord == 2:
        if axis is None and obj.ndim != 1:
            raise NotImplementedError(
                "AutoIA validated matrix spectral 2-norm is not yet supported; "
                "use ord='fro' for the Frobenius norm."
            )
        if axis is not None and not np.isscalar(axis):
            axes = tuple(axis)
            if len(axes) != 1:
                raise NotImplementedError(
                    "AutoIA validated ord=2 currently supports vector norms only."
                )
    else:
        raise NotImplementedError(
            "AutoIA validated np.linalg.norm currently supports ord=None, vector ord=2, "
            "and matrix ord='fro'."
        )

    squared = obj * obj
    total = validated_sum(squared, axis=axis, keepdims=keepdims)
    try:
        return _sqrt_validated_result(total)
    except (ElementaryDomainError, DomainRefinementRequired):
        # A norm is mathematically nonnegative even when an intermediate
        # polynomial/interval range for sum(x_i^2) includes a small artificial
        # negative tail.  For the scalar whole-array norm we can use the norm's
        # own semantics to build a safe interval image from natural same-variable
        # square ranges, rather than clipping an unrelated sqrt argument.
        if axis is not None:
            raise
        prototype = next(v for v in obj.flat if isinstance(v, ValidatedValue))
        backend = prototype.backend
        total_I = point_interval(0.0)
        for v in obj.flat:
            if isinstance(v, ValidatedValue):
                Xi = v.to_interval().interval
            else:
                Xi = point_interval(float(v))
            sq = real_power_interval(Xi, 2.0, backend)
            # The exact square is nonnegative by semantics.  Outward rounding can
            # produce a predecessor below +0; intersecting with [0,+inf) is a
            # theorem-backed tightening for a square, not arbitrary clipping.
            sq = type(sq)(max(0.0, sq.lo), sq.hi)
            total_I = backend.add(total_I, sq)
            total_I = type(total_I)(max(0.0, total_I.lo), total_I.hi)
        image = sqrt_interval(total_I, backend)
        proto_I = prototype.to_interval()
        rep = IntervalEnclosure(
            image, proto_I.h, proto_I.order, proto_I.backend, proto_I.range_method
        )
        return ValidatedValue(rep)


__all__ = [
    "contains_validated",
    "as_object_array",
    "validated_sum",
    "validated_prod",
    "validated_mean",
    "validated_dot",
    "validated_matmul",
    "validated_inner",
    "validated_outer",
    "validated_trace",
    "validated_norm",
]
