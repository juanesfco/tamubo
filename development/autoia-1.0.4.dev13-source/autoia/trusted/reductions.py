"""Proof-critical reductions and small linear-algebra kernels.

AutoIA retains two rigorous reduction implementations:

``reference``
    The historical scalar oracle.  Every product and addition is performed by
    the outward-rounded scalar backend in a Python loop.

``vectorized``
    A Step-9/9b performance backend. Point, point×interval, and interval×interval
    products are formed by NumPy float64 elementwise arithmetic and widened
    componentwise with ``nextafter``. The resulting intervals are summed in a
    pairwise tree; every tree addition is again widened outward. Matrix kernels
    batch many independent trees. Thus the fast path is still a direct interval
    computation, not an a-posteriori BLAS error estimate.

``auto`` (the rigorous backend default)
    Uses the reference oracle for short reductions and the vectorized tree once
    the number of terms reaches ``backend.vectorized_reduction_min_terms``.

The slow scalar functions remain available through ``reduction_method='reference'``
for regression/oracle testing.
"""

from __future__ import annotations

import numpy as np

from .interval import Interval, point_interval, abs_upper_interval
from .array_ops import rigorous_interval_subtract_arrays, rigorous_interval_divide_point_arrays


def _record_reduction(backend, kind: str, mode: str, terms: int) -> None:
    recorder = getattr(backend, "_record_reduction", None)
    if recorder is not None:
        recorder(kind, mode, int(terms))


def _reduction_mode(backend, terms: int) -> str:
    mode = str(getattr(backend, "reduction_method", "reference"))
    if mode == "auto":
        threshold = int(getattr(backend, "vectorized_reduction_min_terms", 16))
        return "vectorized" if int(terms) >= threshold else "reference"
    if mode not in {"reference", "vectorized"}:
        raise ValueError(f"Unknown trusted reduction method {mode!r}.")
    return mode


def _batched_reduction_mode(backend, terms: int, count: int) -> str:
    """Choose a reduction mode for many independent same-length sums.

    The historical scalar/vector threshold is expressed in terms per individual
    dot product.  A lowered family kernel can amortize vectorized setup across
    many short reductions, so ``auto`` uses the vectorized tree when the total
    number of proof products in the batch reaches the same threshold.
    Explicit ``reference`` and ``vectorized`` selections are never overridden.
    """
    configured = str(getattr(backend, "reduction_method", "reference"))
    if configured != "auto":
        return _reduction_mode(backend, terms)
    threshold = int(getattr(backend, "vectorized_reduction_min_terms", 16))
    return "vectorized" if int(terms) * int(count) >= threshold else "reference"


def _nextafter_array(values, toward):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.nextafter(np.asarray(values, dtype=np.float64), np.float64(toward))


def _vectorized_point_product_interval_arrays(a, b):
    """Outward product endpoint arrays for broadcast-compatible point arrays."""
    a, b = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    )
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        raise FloatingPointError("Fast trusted point reductions require finite inputs.")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        p = np.multiply(a, b, dtype=np.float64)
    if np.any(~np.isfinite(p)):
        raise FloatingPointError("Overflow in vectorized trusted point multiplication.")
    lo = _nextafter_array(p, -np.inf)
    hi = _nextafter_array(p, np.inf)
    zero = (a == 0.0) | (b == 0.0)
    lo[zero] = 0.0; hi[zero] = 0.0
    a_one = a == 1.0; b_one = b == 1.0
    a_mone = a == -1.0; b_mone = b == -1.0
    lo[a_one] = b[a_one]; hi[a_one] = b[a_one]
    lo[b_one] = a[b_one]; hi[b_one] = a[b_one]
    lo[a_mone] = -b[a_mone]; hi[a_mone] = -b[a_mone]
    lo[b_mone] = -a[b_mone]; hi[b_mone] = -a[b_mone]
    return lo, hi


def _pairwise_interval_sum_axis_last(lo, hi):
    """Reduce many independent interval sums along the final array axis."""
    lo = np.asarray(lo, dtype=np.float64).copy()
    hi = np.asarray(hi, dtype=np.float64).copy()
    if lo.shape != hi.shape:
        raise ValueError("Interval endpoint shapes do not match.")
    if lo.ndim == 0:
        return lo, hi
    if lo.shape[-1] == 0:
        shape = lo.shape[:-1]
        return np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.float64)
    while lo.shape[-1] > 1:
        n = lo.shape[-1]
        pairs = n // 2
        left_lo = lo[..., : 2 * pairs : 2]
        right_lo = lo[..., 1 : 2 * pairs : 2]
        left_hi = hi[..., : 2 * pairs : 2]
        right_hi = hi[..., 1 : 2 * pairs : 2]
        with np.errstate(over="ignore", invalid="ignore"):
            sum_lo = np.add(left_lo, right_lo, dtype=np.float64)
            sum_hi = np.add(left_hi, right_hi, dtype=np.float64)
        if np.any(~np.isfinite(sum_lo)) or np.any(~np.isfinite(sum_hi)):
            raise FloatingPointError("Overflow in vectorized trusted interval reduction.")
        new_lo = _nextafter_array(sum_lo, -np.inf)
        new_hi = _nextafter_array(sum_hi, np.inf)
        left_zero = (left_lo == 0.0) & (left_hi == 0.0)
        right_zero = (right_lo == 0.0) & (right_hi == 0.0)
        new_lo[left_zero] = right_lo[left_zero]; new_hi[left_zero] = right_hi[left_zero]
        new_lo[right_zero] = left_lo[right_zero]; new_hi[right_zero] = left_hi[right_zero]
        if n % 2:
            new_lo = np.concatenate((new_lo, lo[..., -1:]), axis=-1)
            new_hi = np.concatenate((new_hi, hi[..., -1:]), axis=-1)
        lo, hi = new_lo, new_hi
    return lo[..., 0], hi[..., 0]


def _pairwise_nonnegative_sum_upper_axis_last(values):
    """Upward pairwise reduction of many nonnegative sums along the last axis."""
    values = np.asarray(values, dtype=np.float64).copy()
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Expected finite nonnegative values.")
    if values.ndim == 0:
        return values
    if values.shape[-1] == 0:
        return np.zeros(values.shape[:-1], dtype=np.float64)
    while values.shape[-1] > 1:
        n = values.shape[-1]; pairs = n // 2
        left = values[..., : 2 * pairs : 2]
        right = values[..., 1 : 2 * pairs : 2]
        with np.errstate(over="ignore", invalid="ignore"):
            s = np.add(left, right, dtype=np.float64)
        if np.any(~np.isfinite(s)):
            raise FloatingPointError("Overflow in vectorized trusted nonnegative sum.")
        up = _nextafter_array(s, np.inf)
        up[left == 0.0] = right[left == 0.0]
        up[right == 0.0] = left[right == 0.0]
        if n % 2:
            up = np.concatenate((up, values[..., -1:]), axis=-1)
        values = up
    return values[..., 0]


def _record_logical_reductions(backend, kind, mode, terms, count):
    recorder = getattr(backend, "_record_reduction_many", None)
    if recorder is not None:
        recorder(kind, mode, int(terms), int(count))
        return
    for _ in range(int(count)):
        _record_reduction(backend, kind, mode, terms)


def _interval_endpoint_arrays_shaped(values):
    values = np.asarray(values, dtype=object)
    lo = np.empty(values.shape, dtype=np.float64)
    hi = np.empty(values.shape, dtype=np.float64)
    for idx in np.ndindex(values.shape):
        value = values[idx]
        if not isinstance(value, Interval):
            raise TypeError("Trusted interval reductions require Interval entries.")
        lo[idx] = value.lo; hi[idx] = value.hi
    return lo, hi


def _interval_objects_from_endpoints(lo, hi):
    lo = np.asarray(lo, dtype=np.float64); hi = np.asarray(hi, dtype=np.float64)
    if lo.shape != hi.shape:
        raise ValueError("Endpoint shapes do not match.")
    out = np.empty(lo.shape, dtype=object)
    for idx in np.ndindex(lo.shape):
        out[idx] = Interval(float(lo[idx]), float(hi[idx]))
    return out


def _vectorized_point_product_intervals(a, b):
    """Outward intervals for exact products of stored finite binary64 points."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Dot-product shapes do not match.")
    return _vectorized_point_product_interval_arrays(a, b)


def _interval_endpoint_arrays(values):
    """Extract binary64 lower/upper endpoint arrays from Interval objects."""
    values = np.asarray(values, dtype=object).reshape(-1)
    lo = np.empty(values.size, dtype=np.float64)
    hi = np.empty(values.size, dtype=np.float64)
    for i, value in enumerate(values):
        if not isinstance(value, Interval):
            raise TypeError("Trusted interval reductions require Interval entries.")
        lo[i] = value.lo
        hi[i] = value.hi
    return lo, hi


def _vectorized_point_interval_product_interval_arrays(a, x_lo, x_hi):
    """Outward products for broadcast-compatible point and interval arrays."""
    a, x_lo, x_hi = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64),
        np.asarray(x_lo, dtype=np.float64),
        np.asarray(x_hi, dtype=np.float64),
    )
    if (np.any(~np.isfinite(a)) or np.any(~np.isfinite(x_lo)) or
            np.any(~np.isfinite(x_hi))):
        raise FloatingPointError("Fast trusted point-interval products require finite endpoints.")
    nonnegative = a >= 0.0
    chosen_lo = np.where(nonnegative, x_lo, x_hi)
    chosen_hi = np.where(nonnegative, x_hi, x_lo)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        p_lo = np.multiply(a, chosen_lo, dtype=np.float64)
        p_hi = np.multiply(a, chosen_hi, dtype=np.float64)
    if np.any(~np.isfinite(p_lo)) or np.any(~np.isfinite(p_hi)):
        raise FloatingPointError("Overflow in vectorized trusted point-interval multiplication.")
    lo = _nextafter_array(p_lo, -np.inf); hi = _nextafter_array(p_hi, np.inf)
    zero = a == 0.0
    lo[zero] = 0.0; hi[zero] = 0.0
    one = a == 1.0
    lo[one] = x_lo[one]; hi[one] = x_hi[one]
    mone = a == -1.0
    lo[mone] = -x_hi[mone]; hi[mone] = -x_lo[mone]
    return lo, hi


def _vectorized_point_interval_product_intervals(a, x_lo, x_hi):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    x_lo = np.asarray(x_lo, dtype=np.float64).reshape(-1)
    x_hi = np.asarray(x_hi, dtype=np.float64).reshape(-1)
    if a.shape != x_lo.shape or a.shape != x_hi.shape:
        raise ValueError("Point/interval product shapes do not match.")
    return _vectorized_point_interval_product_interval_arrays(a, x_lo, x_hi)


def _outward_product_candidate(a, b):
    """Return downward/upward arrays enclosing exact finite point products."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        p = np.multiply(a, b, dtype=np.float64)
    if np.any(~np.isfinite(p)):
        raise FloatingPointError("Overflow in vectorized trusted interval multiplication.")
    return _nextafter_array(p, -np.inf), _nextafter_array(p, np.inf)


def _vectorized_interval_product_intervals(a_lo, a_hi, b_lo, b_hi):
    """Outward natural interval products for broadcast-compatible finite arrays."""
    a_lo, a_hi, b_lo, b_hi = np.broadcast_arrays(
        np.asarray(a_lo, dtype=np.float64), np.asarray(a_hi, dtype=np.float64),
        np.asarray(b_lo, dtype=np.float64), np.asarray(b_hi, dtype=np.float64),
    )
    if any(np.any(~np.isfinite(v)) for v in (a_lo, a_hi, b_lo, b_hi)):
        raise FloatingPointError("Fast trusted interval products require finite endpoints.")
    lows = []; highs = []
    for a, b in ((a_lo, b_lo), (a_lo, b_hi), (a_hi, b_lo), (a_hi, b_hi)):
        l, h = _outward_product_candidate(a, b); lows.append(l); highs.append(h)
    lo = np.minimum.reduce(lows); hi = np.maximum.reduce(highs)
    a_zero = (a_lo == 0.0) & (a_hi == 0.0); b_zero = (b_lo == 0.0) & (b_hi == 0.0)
    z = a_zero | b_zero; lo[z] = 0.0; hi[z] = 0.0
    a_one = (a_lo == 1.0) & (a_hi == 1.0); b_one = (b_lo == 1.0) & (b_hi == 1.0)
    lo[a_one] = b_lo[a_one]; hi[a_one] = b_hi[a_one]
    lo[b_one] = a_lo[b_one]; hi[b_one] = a_hi[b_one]
    a_mone = (a_lo == -1.0) & (a_hi == -1.0); b_mone = (b_lo == -1.0) & (b_hi == -1.0)
    lo[a_mone] = -b_hi[a_mone]; hi[a_mone] = -b_lo[a_mone]
    lo[b_mone] = -a_hi[b_mone]; hi[b_mone] = -a_lo[b_mone]
    return lo, hi


def _pairwise_interval_sum(lo, hi) -> Interval:
    """Outward pairwise reduction of same-shape interval endpoint arrays."""
    lo = np.asarray(lo, dtype=np.float64).reshape(-1).copy()
    hi = np.asarray(hi, dtype=np.float64).reshape(-1).copy()
    if lo.shape != hi.shape:
        raise ValueError("Interval endpoint shapes do not match.")
    if lo.size == 0:
        return Interval(0.0, 0.0)

    while lo.size > 1:
        pairs = lo.size // 2
        left_lo = lo[: 2 * pairs : 2]
        right_lo = lo[1 : 2 * pairs : 2]
        left_hi = hi[: 2 * pairs : 2]
        right_hi = hi[1 : 2 * pairs : 2]

        with np.errstate(over="ignore", invalid="ignore"):
            sum_lo = np.add(left_lo, right_lo, dtype=np.float64)
            sum_hi = np.add(left_hi, right_hi, dtype=np.float64)
        if np.any(~np.isfinite(sum_lo)) or np.any(~np.isfinite(sum_hi)):
            raise FloatingPointError("Overflow in vectorized trusted interval reduction.")

        new_lo = _nextafter_array(sum_lo, -np.inf)
        new_hi = _nextafter_array(sum_hi, np.inf)

        # Exact zero-interval addition is an identity in the scalar oracle.
        left_zero = (left_lo == 0.0) & (left_hi == 0.0)
        right_zero = (right_lo == 0.0) & (right_hi == 0.0)
        new_lo[left_zero] = right_lo[left_zero]
        new_hi[left_zero] = right_hi[left_zero]
        new_lo[right_zero] = left_lo[right_zero]
        new_hi[right_zero] = left_hi[right_zero]

        if lo.size % 2:
            new_lo = np.concatenate((new_lo, lo[-1:]))
            new_hi = np.concatenate((new_hi, hi[-1:]))
        lo, hi = new_lo, new_hi

    return Interval(float(lo[0]), float(hi[0]))


def _pairwise_nonnegative_sum_upper(values) -> float:
    """Upward pairwise sum for finite stored nonnegative binary64 values."""
    values = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Expected finite nonnegative values.")
    if values.size == 0:
        return 0.0
    while values.size > 1:
        pairs = values.size // 2
        left = values[: 2 * pairs : 2]
        right = values[1 : 2 * pairs : 2]
        with np.errstate(over="ignore", invalid="ignore"):
            s = np.add(left, right, dtype=np.float64)
        if np.any(~np.isfinite(s)):
            raise FloatingPointError("Overflow in vectorized trusted nonnegative sum.")
        up = _nextafter_array(s, np.inf)
        up[left == 0.0] = right[left == 0.0]
        up[right == 0.0] = left[right == 0.0]
        if values.size % 2:
            up = np.concatenate((up, values[-1:]))
        values = up
    return float(values[0])



def rigorous_point_interval_sum_axis_last(a, x_lo, x_hi, backend):
    """Batch rigorous sums ``sum_t a_t * X_t`` along the final axis.

    ``a`` contains stored binary64 point values. ``x_lo`` and ``x_hi`` are
    lower/upper endpoint arrays for interval factors and may be broadcast
    against ``a``.  The returned pair ``(lo, hi)`` contains endpoint arrays
    with the final reduction axis removed.

    This is a proof-kernel primitive for numeric lowered representations.  In
    vectorized mode every elementwise product is widened outward and every
    pairwise tree addition is widened outward.  The scalar backend remains the
    reference path and is used automatically for short reductions.
    """
    a, x_lo, x_hi = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64),
        np.asarray(x_lo, dtype=np.float64),
        np.asarray(x_hi, dtype=np.float64),
    )
    if a.ndim == 0:
        raise ValueError("A reduction axis is required.")
    terms = int(a.shape[-1])
    count = int(np.prod(a.shape[:-1], dtype=np.int64)) if a.shape[:-1] else 1
    mode = _batched_reduction_mode(backend, terms, count)

    if mode == "vectorized":
        try:
            plo, phi = _vectorized_point_interval_product_interval_arrays(
                a, x_lo, x_hi
            )
            lo, hi = _pairwise_interval_sum_axis_last(plo, phi)
            _record_logical_reductions(
                backend, "dot-point-interval", mode, terms, count
            )
            return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        except FloatingPointError:
            # Match the existing trusted-reduction contract: unsupported
            # exceptional/extended cases fall back to the mature scalar path.
            pass

    out_shape = a.shape[:-1]
    lo = np.empty(out_shape, dtype=np.float64)
    hi = np.empty(out_shape, dtype=np.float64)
    for idx in np.ndindex(out_shape):
        total = Interval(0.0, 0.0)
        for t in range(terms):
            total = backend.add(
                total,
                backend.mul(
                    point_interval(float(a[idx + (t,)])),
                    Interval(
                        float(x_lo[idx + (t,)]),
                        float(x_hi[idx + (t,)]),
                    ),
                ),
            )
        lo[idx] = total.lo
        hi[idx] = total.hi
    _record_logical_reductions(
        backend, "dot-point-interval", "reference", terms, count
    )
    return lo, hi


def rigorous_interval_product_sum_axis_last(a_lo, a_hi, b_lo, b_hi, backend):
    """Batch rigorous natural interval dot products along the final axis.

    The four endpoint arrays are broadcast to a common shape.  For every
    leading index this returns an enclosure of

        ``sum_t [a_lo,a_hi]_t * [b_lo,b_hi]_t``.

    This is the interval×interval analogue of
    :func:`rigorous_point_interval_sum_axis_last`.
    """
    a_lo, a_hi, b_lo, b_hi = np.broadcast_arrays(
        np.asarray(a_lo, dtype=np.float64),
        np.asarray(a_hi, dtype=np.float64),
        np.asarray(b_lo, dtype=np.float64),
        np.asarray(b_hi, dtype=np.float64),
    )
    if a_lo.ndim == 0:
        raise ValueError("A reduction axis is required.")
    terms = int(a_lo.shape[-1])
    count = int(np.prod(a_lo.shape[:-1], dtype=np.int64)) if a_lo.shape[:-1] else 1
    mode = _batched_reduction_mode(backend, terms, count)

    if mode == "vectorized":
        try:
            plo, phi = _vectorized_interval_product_intervals(
                a_lo, a_hi, b_lo, b_hi
            )
            lo, hi = _pairwise_interval_sum_axis_last(plo, phi)
            _record_logical_reductions(
                backend, "dot-interval", mode, terms, count
            )
            return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        except FloatingPointError:
            pass

    out_shape = a_lo.shape[:-1]
    lo = np.empty(out_shape, dtype=np.float64)
    hi = np.empty(out_shape, dtype=np.float64)
    for idx in np.ndindex(out_shape):
        total = Interval(0.0, 0.0)
        for t in range(terms):
            total = backend.add(
                total,
                backend.mul(
                    Interval(
                        float(a_lo[idx + (t,)]),
                        float(a_hi[idx + (t,)]),
                    ),
                    Interval(
                        float(b_lo[idx + (t,)]),
                        float(b_hi[idx + (t,)]),
                    ),
                ),
            )
        lo[idx] = total.lo
        hi[idx] = total.hi
    _record_logical_reductions(
        backend, "dot-interval", "reference", terms, count
    )
    return lo, hi


def rigorous_interval_subtract_point_arrays(x_lo, x_hi, point, backend):
    """Return endpoint arrays enclosing ``[x_lo,x_hi] - point``.

    The operation is componentwise, not a reduction.  For the rigorous
    binary64 backend finite arrays use direct float64 subtraction followed by
    one outward ``nextafter`` per endpoint, with the exact zero-subtraction
    identity retained.  Other/exceptional cases fall back to scalar backend
    operations.
    """
    x_lo, x_hi, point = np.broadcast_arrays(
        np.asarray(x_lo, dtype=np.float64),
        np.asarray(x_hi, dtype=np.float64),
        np.asarray(point, dtype=np.float64),
    )
    finite = (
        np.all(np.isfinite(x_lo))
        and np.all(np.isfinite(x_hi))
        and np.all(np.isfinite(point))
    )
    if bool(getattr(backend, "machine_rigorous", False)) and finite:
        with np.errstate(over="ignore", invalid="ignore"):
            lo_rn = np.subtract(x_lo, point, dtype=np.float64)
            hi_rn = np.subtract(x_hi, point, dtype=np.float64)
        if np.any(~np.isfinite(lo_rn)) or np.any(~np.isfinite(hi_rn)):
            raise FloatingPointError("Overflow in trusted interval-point subtraction.")
        lo = _nextafter_array(lo_rn, -np.inf)
        hi = _nextafter_array(hi_rn, np.inf)
        zero = point == 0.0
        lo[zero] = x_lo[zero]
        hi[zero] = x_hi[zero]
        return lo, hi

    lo = np.empty(x_lo.shape, dtype=np.float64)
    hi = np.empty(x_hi.shape, dtype=np.float64)
    for idx in np.ndindex(x_lo.shape):
        value = backend.sub(
            Interval(float(x_lo[idx]), float(x_hi[idx])),
            point_interval(float(point[idx])),
        )
        lo[idx] = value.lo
        hi[idx] = value.hi
    return lo, hi


def rigorous_triangular_solve_point_interval_many(L, rhs_lo, rhs_hi, backend):
    """Rigorously solve ``L x = rhs`` for many interval right-hand sides.

    Parameters
    ----------
    L : array_like, shape (n, n)
        Stored binary64 point-valued lower-triangular matrix.  Every diagonal
        entry must be nonzero.
    rhs_lo, rhs_hi : array_like, shape (..., n)
        Endpoint arrays for one or many interval right-hand sides.  Arbitrary
        leading dimensions are preserved.
    backend : trusted backend
        Normally :class:`RigorousBinary64Backend`.

    Returns
    -------
    (x_lo, x_hi) : tuple of ndarray
        Endpoint arrays enclosing the exact-real natural interval forward
        substitution result for every leading batch index.

    Notes
    -----
    Only the triangular row index is traversed in Python.  For each row, all
    already-known point-times-interval products across the entire batch are
    reduced by :func:`rigorous_point_interval_sum_axis_last`.  The historical
    scalar arithmetic therefore remains the oracle while large batches use the
    same outward-rounded reduction kernel as the rest of AutoIA.
    """
    L = np.asarray(L, dtype=np.float64)
    rhs_lo, rhs_hi = np.broadcast_arrays(
        np.asarray(rhs_lo, dtype=np.float64), np.asarray(rhs_hi, dtype=np.float64)
    )
    if rhs_lo.ndim == 0:
        raise ValueError("rhs endpoint arrays require a final solve dimension.")
    n = int(rhs_lo.shape[-1])
    if L.shape != (n, n):
        raise ValueError("L must be square and match the final rhs dimension.")
    if np.any(~np.isfinite(L)) or np.any(~np.isfinite(rhs_lo)) or np.any(~np.isfinite(rhs_hi)):
        raise ValueError("Trusted batched triangular solve requires finite stored inputs.")
    if np.any(rhs_lo > rhs_hi):
        raise ValueError("rhs lower endpoints exceed upper endpoints.")
    if np.any(np.diag(L) == 0.0):
        raise ZeroDivisionError("Triangular solve has a zero diagonal entry.")
    if np.any(np.triu(L, k=1) != 0.0):
        raise ValueError("L must be lower triangular.")

    out_lo = np.empty(rhs_lo.shape, dtype=np.float64)
    out_hi = np.empty(rhs_hi.shape, dtype=np.float64)
    for j in range(n):
        if j == 0:
            numer_lo = rhs_lo[..., 0]
            numer_hi = rhs_hi[..., 0]
        else:
            coeff = np.broadcast_to(L[j, :j], rhs_lo.shape[:-1] + (j,))
            sum_lo, sum_hi = rigorous_point_interval_sum_axis_last(
                coeff, out_lo[..., :j], out_hi[..., :j], backend
            )
            numer_lo, numer_hi = rigorous_interval_subtract_arrays(
                rhs_lo[..., j], rhs_hi[..., j], sum_lo, sum_hi, backend
            )
        row_lo, row_hi = rigorous_interval_divide_point_arrays(
            numer_lo, numer_hi, np.float64(L[j, j]), backend
        )
        out_lo[..., j] = row_lo
        out_hi[..., j] = row_hi
    return out_lo, out_hi

def interval_sum(values, backend) -> Interval:
    # General Interval objects remain on the scalar oracle because the main
    # performance hot spots use specialized point/nonnegative kernels below.
    total = Interval(0.0, 0.0)
    for value in values:
        total = backend.add(total, value)
    return total


def _rigorous_dot_point_reference(a, b, backend) -> Interval:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Dot-product shapes do not match.")
    s = Interval(0.0, 0.0)
    for ai, bi in zip(a, b):
        prod = backend.mul(point_interval(ai), point_interval(bi))
        s = backend.add(s, prod)
    return s


def _rigorous_dot_point_vectorized(a, b) -> Interval:
    lo, hi = _vectorized_point_product_intervals(a, b)
    return _pairwise_interval_sum(lo, hi)


def rigorous_dot_point(a, b, backend) -> Interval:
    """Enclose the exact-real dot product of stored point binary64 vectors."""
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Dot-product shapes do not match.")
    mode = _reduction_mode(backend, a.size)
    _record_reduction(backend, "dot-point", mode, a.size)
    if mode == "vectorized":
        return _rigorous_dot_point_vectorized(a, b)
    return _rigorous_dot_point_reference(a, b, backend)


def rigorous_matmul_point(A, B, backend):
    """Interval enclosure of exact-real ``A @ B`` for stored point matrices."""
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError("Invalid matrix shapes for multiplication.")
    m, k = A.shape; n = B.shape[1]
    mode = _reduction_mode(backend, k)
    if mode == "vectorized":
        try:
            out = np.empty((m, n), dtype=object)
            # Bound temporary product arrays to roughly two million entries.
            max_terms = 2_000_000
            block_rows = max(1, min(m, max_terms // max(1, n * k)))
            BT = B.T
            for i0 in range(0, m, block_rows):
                i1 = min(m, i0 + block_rows)
                lo, hi = _vectorized_point_product_interval_arrays(
                    A[i0:i1, None, :], BT[None, :, :]
                )
                slo, shi = _pairwise_interval_sum_axis_last(lo, hi)
                out[i0:i1, :] = _interval_objects_from_endpoints(slo, shi)
            _record_logical_reductions(backend, "dot-point", mode, k, m*n)
            return out
        except FloatingPointError:
            pass
    out = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            out[i, j] = rigorous_dot_point(A[i, :], B[:, j], backend)
    return out


def rigorous_matvec_point(A, x, backend):
    """Interval enclosure of exact-real ``A @ x`` for stored point data."""
    A = np.asarray(A, dtype=float); x = np.asarray(x, dtype=float)
    if A.ndim != 2 or x.ndim != 1 or A.shape[1] != x.shape[0]:
        raise ValueError("Invalid matrix/vector shapes.")
    m, k = A.shape; mode = _reduction_mode(backend, k)
    if mode == "vectorized":
        try:
            lo, hi = _vectorized_point_product_interval_arrays(A, x[None, :])
            slo, shi = _pairwise_interval_sum_axis_last(lo, hi)
            _record_logical_reductions(backend, "dot-point", mode, k, m)
            return _interval_objects_from_endpoints(slo, shi)
        except FloatingPointError:
            pass
    out = np.empty(m, dtype=object)
    for i in range(m): out[i] = rigorous_dot_point(A[i, :], x, backend)
    return out


def _rigorous_dot_point_interval_reference(a, xI, backend) -> Interval:
    a = np.asarray(a, dtype=float).reshape(-1)
    xI = np.asarray(xI, dtype=object).reshape(-1)
    if a.shape != xI.shape:
        raise ValueError("Point/interval dot-product shapes do not match.")
    total = Interval(0.0, 0.0)
    for ai, xi in zip(a, xI):
        total = backend.add(total, backend.mul(point_interval(ai), xi))
    return total


def _rigorous_dot_point_interval_vectorized(a, xI) -> Interval:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    x_lo, x_hi = _interval_endpoint_arrays(xI)
    lo, hi = _vectorized_point_interval_product_intervals(a, x_lo, x_hi)
    return _pairwise_interval_sum(lo, hi)


def rigorous_dot_point_interval(a, xI, backend) -> Interval:
    """Enclose ``sum_i a_i x_i`` for point ``a_i`` and interval ``x_i``."""
    a = np.asarray(a, dtype=float).reshape(-1)
    xI = np.asarray(xI, dtype=object).reshape(-1)
    if a.shape != xI.shape:
        raise ValueError("Point/interval dot-product shapes do not match.")
    mode = _reduction_mode(backend, a.size)
    _record_reduction(backend, "dot-point-interval", mode, a.size)
    if mode == "vectorized":
        try:
            return _rigorous_dot_point_interval_vectorized(a, xI)
        except FloatingPointError:
            # Extended endpoints and unsupported exceptional products remain on
            # the mature scalar semantics.  True finite overflow still fails
            # closed there as before.
            return _rigorous_dot_point_interval_reference(a, xI, backend)
    return _rigorous_dot_point_interval_reference(a, xI, backend)


def _rigorous_dot_interval_reference(aI, bI, backend) -> Interval:
    aI = np.asarray(aI, dtype=object).reshape(-1)
    bI = np.asarray(bI, dtype=object).reshape(-1)
    if aI.shape != bI.shape:
        raise ValueError("Interval dot-product shapes do not match.")
    total = Interval(0.0, 0.0)
    for ai, bi in zip(aI, bI):
        total = backend.add(total, backend.mul(ai, bi))
    return total


def _rigorous_dot_interval_vectorized(aI, bI) -> Interval:
    a_lo, a_hi = _interval_endpoint_arrays(aI)
    b_lo, b_hi = _interval_endpoint_arrays(bI)
    lo, hi = _vectorized_interval_product_intervals(a_lo, a_hi, b_lo, b_hi)
    return _pairwise_interval_sum(lo, hi)


def rigorous_dot_interval(aI, bI, backend) -> Interval:
    """Enclose the natural interval dot product of two interval vectors."""
    aI = np.asarray(aI, dtype=object).reshape(-1)
    bI = np.asarray(bI, dtype=object).reshape(-1)
    if aI.shape != bI.shape:
        raise ValueError("Interval dot-product shapes do not match.")
    mode = _reduction_mode(backend, aI.size)
    _record_reduction(backend, "dot-interval", mode, aI.size)
    if mode == "vectorized":
        try:
            return _rigorous_dot_interval_vectorized(aI, bI)
        except FloatingPointError:
            return _rigorous_dot_interval_reference(aI, bI, backend)
    return _rigorous_dot_interval_reference(aI, bI, backend)


def rigorous_matvec_point_interval(A, xI, backend):
    """Enclose ``A @ x`` when A is point-valued and x has interval entries."""
    A = np.asarray(A, dtype=float); xI = np.asarray(xI, dtype=object)
    if A.ndim != 2 or xI.ndim != 1 or A.shape[1] != xI.shape[0]:
        raise ValueError("Invalid matrix/interval-vector shapes.")
    m, k = A.shape; mode = _reduction_mode(backend, k)
    if mode == "vectorized":
        try:
            xl, xh = _interval_endpoint_arrays_shaped(xI)
            lo, hi = _vectorized_point_interval_product_interval_arrays(A, xl[None, :], xh[None, :])
            slo, shi = _pairwise_interval_sum_axis_last(lo, hi)
            _record_logical_reductions(backend, "dot-point-interval", mode, k, m)
            return _interval_objects_from_endpoints(slo, shi)
        except FloatingPointError:
            pass
    out = np.empty(m, dtype=object)
    for i in range(m): out[i] = rigorous_dot_point_interval(A[i, :], xI, backend)
    return out


def rigorous_matmul_point_interval(A, BI, backend):
    """Enclose ``A @ B`` when A is point-valued and B has interval entries."""
    A = np.asarray(A, dtype=float); BI = np.asarray(BI, dtype=object)
    if A.ndim != 2 or BI.ndim != 2 or A.shape[1] != BI.shape[0]:
        raise ValueError("Invalid matrix/interval-matrix shapes.")
    m, k = A.shape; n = BI.shape[1]; mode = _reduction_mode(backend, k)
    if mode == "vectorized":
        try:
            bl, bh = _interval_endpoint_arrays_shaped(BI); blt = bl.T; bht = bh.T
            out = np.empty((m, n), dtype=object)
            max_terms = 2_000_000
            block_rows = max(1, min(m, max_terms // max(1, n*k)))
            for i0 in range(0, m, block_rows):
                i1 = min(m, i0+block_rows)
                lo, hi = _vectorized_point_interval_product_interval_arrays(
                    A[i0:i1, None, :], blt[None, :, :], bht[None, :, :]
                )
                slo, shi = _pairwise_interval_sum_axis_last(lo, hi)
                out[i0:i1, :] = _interval_objects_from_endpoints(slo, shi)
            _record_logical_reductions(backend, "dot-point-interval", mode, k, m*n)
            return out
        except FloatingPointError:
            pass
    out = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n): out[i, j] = rigorous_dot_point_interval(A[i, :], BI[:, j], backend)
    return out


def interval_matvec(AI, xI, backend):
    """Interval matrix times interval vector, with outward-rounded operations."""
    AI = np.asarray(AI, dtype=object); xI = np.asarray(xI, dtype=object)
    if AI.ndim != 2 or xI.ndim != 1 or AI.shape[1] != xI.shape[0]:
        raise ValueError("Invalid interval matrix/vector shapes.")
    m, k = AI.shape; mode = _reduction_mode(backend, k)
    if mode == "vectorized":
        try:
            al, ah = _interval_endpoint_arrays_shaped(AI); xl, xh = _interval_endpoint_arrays_shaped(xI)
            lo, hi = _vectorized_interval_product_intervals(al, ah, xl[None, :], xh[None, :])
            slo, shi = _pairwise_interval_sum_axis_last(lo, hi)
            _record_logical_reductions(backend, "dot-interval", mode, k, m)
            return _interval_objects_from_endpoints(slo, shi)
        except FloatingPointError:
            pass
    out = np.empty(m, dtype=object)
    for i in range(m): out[i] = rigorous_dot_interval(AI[i, :], xI, backend)
    return out


def interval_matmul(AI, BI, backend):
    """Natural interval matrix product with rigorous dot reductions."""
    AI = np.asarray(AI, dtype=object); BI = np.asarray(BI, dtype=object)
    if AI.ndim != 2 or BI.ndim != 2 or AI.shape[1] != BI.shape[0]:
        raise ValueError("Invalid interval matrix shapes.")
    m, k = AI.shape; n = BI.shape[1]; mode = _reduction_mode(backend, k)
    if mode == "vectorized":
        try:
            al, ah = _interval_endpoint_arrays_shaped(AI); bl, bh = _interval_endpoint_arrays_shaped(BI)
            blt, bht = bl.T, bh.T; out = np.empty((m,n), dtype=object)
            max_terms = 1_000_000
            block_rows = max(1, min(m, max_terms // max(1,n*k)))
            for i0 in range(0,m,block_rows):
                i1=min(m,i0+block_rows)
                lo, hi = _vectorized_interval_product_intervals(
                    al[i0:i1,None,:], ah[i0:i1,None,:], blt[None,:,:], bht[None,:,:]
                )
                slo, shi = _pairwise_interval_sum_axis_last(lo,hi)
                out[i0:i1,:] = _interval_objects_from_endpoints(slo,shi)
            _record_logical_reductions(backend, "dot-interval", mode, k, m*n)
            return out
        except FloatingPointError:
            pass
    out=np.empty((m,n),dtype=object)
    for i in range(m):
        for j in range(n): out[i,j]=rigorous_dot_interval(AI[i,:],BI[:,j],backend)
    return out


def _nonnegative_dot_upper_reference(a, b, backend) -> float:
    s = 0.0
    for ai, bi in zip(a, b):
        if ai < 0.0 or bi < 0.0:
            raise ValueError("nonnegative_dot_upper received a negative input.")
        prod_hi = backend.mul(point_interval(ai), point_interval(bi)).hi
        s = backend.add(point_interval(s), point_interval(prod_hi)).hi
    return float(s)


def _nonnegative_dot_upper_vectorized(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(a < 0.0) or np.any(b < 0.0):
        raise ValueError("nonnegative_dot_upper requires finite nonnegative inputs.")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        p = np.multiply(a, b, dtype=np.float64)
    if np.any(~np.isfinite(p)):
        raise FloatingPointError("Overflow in vectorized trusted nonnegative product.")
    hi = _nextafter_array(p, np.inf)
    zero = (a == 0.0) | (b == 0.0)
    hi[zero] = 0.0
    hi[a == 1.0] = b[a == 1.0]
    hi[b == 1.0] = a[b == 1.0]
    return _pairwise_nonnegative_sum_upper(hi)


def nonnegative_dot_upper(a, b, backend) -> float:
    """Upward bound for ``sum_i a_i b_i`` for stored nonnegative data."""
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Dot-product shapes do not match.")
    mode = _reduction_mode(backend, a.size)
    _record_reduction(backend, "dot-nonnegative", mode, a.size)
    if mode == "vectorized":
        return _nonnegative_dot_upper_vectorized(a, b)
    return _nonnegative_dot_upper_reference(a, b, backend)


def nonnegative_matvec_upper(A, x, backend):
    A = np.asarray(A, dtype=float); x = np.asarray(x, dtype=float)
    if np.any(A < 0.0) or np.any(x < 0.0): raise ValueError("nonnegative_matvec_upper requires nonnegative inputs.")
    if A.ndim != 2 or x.ndim != 1 or A.shape[1] != len(x): raise ValueError("Invalid matrix/vector shapes.")
    m,k=A.shape; mode=_reduction_mode(backend,k)
    if mode=="vectorized":
        with np.errstate(over="ignore",under="ignore",invalid="ignore"):
            p=np.multiply(A,x[None,:],dtype=np.float64)
        if np.any(~np.isfinite(p)): raise FloatingPointError("Overflow in vectorized trusted nonnegative product.")
        hi=_nextafter_array(p,np.inf); zero=(A==0.0)|(x[None,:]==0.0); hi[zero]=0.0
        hi[A==1.0]=np.broadcast_to(x[None,:],A.shape)[A==1.0]
        sums=_pairwise_nonnegative_sum_upper_axis_last(hi)
        _record_logical_reductions(backend,"dot-nonnegative",mode,k,m)
        return np.asarray(sums,dtype=float)
    return np.array([nonnegative_dot_upper(A[i],x,backend) for i in range(m)],dtype=float)


def nonnegative_matmul_upper(A, B, backend):
    A=np.asarray(A,dtype=float); B=np.asarray(B,dtype=float)
    if np.any(A<0.0) or np.any(B<0.0): raise ValueError("nonnegative_matmul_upper requires nonnegative inputs.")
    if A.ndim!=2 or B.ndim!=2 or A.shape[1]!=B.shape[0]: raise ValueError("Invalid matrix shapes.")
    m,k=A.shape; n=B.shape[1]; mode=_reduction_mode(backend,k)
    if mode=="vectorized":
        BT=B.T; out=np.empty((m,n),dtype=float); max_terms=2_000_000
        block_rows=max(1,min(m,max_terms//max(1,n*k)))
        for i0 in range(0,m,block_rows):
            i1=min(m,i0+block_rows)
            Ab=A[i0:i1,None,:]
            with np.errstate(over="ignore",under="ignore",invalid="ignore"):
                p=np.multiply(Ab,BT[None,:,:],dtype=np.float64)
            if np.any(~np.isfinite(p)): raise FloatingPointError("Overflow in vectorized trusted nonnegative product.")
            hi=_nextafter_array(p,np.inf); zero=(Ab==0.0)|(BT[None,:,:]==0.0); hi[zero]=0.0
            out[i0:i1,:]=_pairwise_nonnegative_sum_upper_axis_last(hi)
        _record_logical_reductions(backend,"dot-nonnegative",mode,k,m*n)
        return out
    out=np.empty((m,n),dtype=float)
    for i in range(m):
        for j in range(n): out[i,j]=nonnegative_dot_upper(A[i,:],B[:,j],backend)
    return out


def row_sum_norm_upper(B, backend) -> float:
    """Rigorous upper bound on ``||B||_infinity`` for nonnegative point B."""
    B = np.asarray(B, dtype=float)
    if np.any(B < 0.0):
        raise ValueError("row_sum_norm_upper requires B >= 0.")
    rho = 0.0
    for i in range(B.shape[0]):
        mode = _reduction_mode(backend, B.shape[1])
        _record_reduction(backend, "row-sum", mode, B.shape[1])
        if mode == "vectorized":
            s = _pairwise_nonnegative_sum_upper(B[i])
        else:
            s = 0.0
            for value in B[i]:
                s = backend.add(point_interval(s), point_interval(value)).hi
        rho = max(rho, s)
    return float(rho)


def matrix_product_abs_upper(C, A, backend):
    """Entrywise upper bound on ``|C A|`` for stored point matrices."""
    CA = rigorous_matmul_point(C, A, backend)
    out = np.empty(CA.shape)
    for idx in np.ndindex(CA.shape):
        out[idx] = abs_upper_interval(CA[idx])
    return out


def identity_minus_product_abs_upper(C, A0, backend):
    """Entrywise upper bound on ``|I - C A0|`` for stored point matrices."""
    CA = rigorous_matmul_point(C, A0, backend)
    n = A0.shape[0]
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            delta = 1.0 if i == j else 0.0
            Dij = backend.sub(point_interval(delta), CA[i, j])
            out[i, j] = abs_upper_interval(Dij)
    return out


def add_nonnegative_matrices_upper(A, B, backend):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape or np.any(A < 0.0) or np.any(B < 0.0):
        raise ValueError("Expected same-shape nonnegative matrices.")
    out = np.empty(A.shape)
    for idx in np.ndindex(A.shape):
        out[idx] = backend.add(point_interval(A[idx]), point_interval(B[idx])).hi
    return out


def scale_nonnegative_matrix_upper(s, A, backend):
    s = float(s)
    A = np.asarray(A, dtype=float)
    if s < 0.0 or np.any(A < 0.0):
        raise ValueError("Expected nonnegative scale and matrix.")
    out = np.empty(A.shape)
    for idx in np.ndindex(A.shape):
        out[idx] = backend.mul(point_interval(s), point_interval(A[idx])).hi
    return out


__all__ = [
    "interval_sum",
    "rigorous_point_interval_sum_axis_last",
    "rigorous_interval_product_sum_axis_last",
    "rigorous_interval_subtract_point_arrays",
    "rigorous_triangular_solve_point_interval_many",
    "rigorous_dot_point",
    "rigorous_matmul_point",
    "rigorous_matvec_point",
    "rigorous_dot_point_interval",
    "rigorous_dot_interval",
    "rigorous_matvec_point_interval",
    "rigorous_matmul_point_interval",
    "interval_matvec",
    "interval_matmul",
    "nonnegative_dot_upper",
    "nonnegative_matvec_upper",
    "nonnegative_matmul_upper",
    "row_sum_norm_upper",
    "matrix_product_abs_upper",
    "identity_minus_product_abs_upper",
    "add_nonnegative_matrices_upper",
    "scale_nonnegative_matrix_upper",
]
