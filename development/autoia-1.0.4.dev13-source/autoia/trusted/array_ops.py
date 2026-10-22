"""Proof-critical elementwise interval arithmetic on numeric endpoint arrays.

These kernels are the elementwise counterpart to :mod:`autoia.trusted.reductions`.
They preserve arbitrary NumPy leading shapes and return separate float64 lower
and upper endpoint arrays.  The rigorous binary64 path forms the ordinary
float64 candidate operation and immediately expands it with ``nextafter``.

The functions are intentionally small and backend-agnostic.  Exceptional or
non-rigorous cases fall back to the scalar backend so the mature scalar
arithmetic remains the oracle.
"""
from __future__ import annotations

import numpy as np

from .interval import Interval, point_interval


def _nextafter(values, toward):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.nextafter(np.asarray(values, dtype=np.float64), np.float64(toward))


def _finite_rigorous(backend, *arrays):
    return bool(getattr(backend, "machine_rigorous", False)) and all(
        np.all(np.isfinite(np.asarray(a, dtype=np.float64))) for a in arrays
    )


def rigorous_interval_add_arrays(a_lo, a_hi, b_lo, b_hi, backend):
    a_lo, a_hi, b_lo, b_hi = np.broadcast_arrays(
        np.asarray(a_lo, dtype=np.float64), np.asarray(a_hi, dtype=np.float64),
        np.asarray(b_lo, dtype=np.float64), np.asarray(b_hi, dtype=np.float64),
    )
    if _finite_rigorous(backend, a_lo, a_hi, b_lo, b_hi):
        with np.errstate(over="ignore", invalid="ignore"):
            lr = np.add(a_lo, b_lo, dtype=np.float64)
            hr = np.add(a_hi, b_hi, dtype=np.float64)
        if np.any(~np.isfinite(lr)) or np.any(~np.isfinite(hr)):
            raise FloatingPointError("Overflow in trusted interval-array addition.")
        lo, hi = _nextafter(lr, -np.inf), _nextafter(hr, np.inf)
        az = (a_lo == 0.0) & (a_hi == 0.0)
        bz = (b_lo == 0.0) & (b_hi == 0.0)
        lo = np.where(az, b_lo, lo); hi = np.where(az, b_hi, hi)
        lo = np.where(bz, a_lo, lo); hi = np.where(bz, a_hi, hi)
        return lo, hi
    lo = np.empty(a_lo.shape); hi = np.empty(a_hi.shape)
    for idx in np.ndindex(a_lo.shape):
        v = backend.add(Interval(float(a_lo[idx]), float(a_hi[idx])), Interval(float(b_lo[idx]), float(b_hi[idx])))
        lo[idx], hi[idx] = v.lo, v.hi
    return lo, hi


def rigorous_interval_subtract_arrays(a_lo, a_hi, b_lo, b_hi, backend):
    a_lo, a_hi, b_lo, b_hi = np.broadcast_arrays(
        np.asarray(a_lo, dtype=np.float64), np.asarray(a_hi, dtype=np.float64),
        np.asarray(b_lo, dtype=np.float64), np.asarray(b_hi, dtype=np.float64),
    )
    if _finite_rigorous(backend, a_lo, a_hi, b_lo, b_hi):
        with np.errstate(over="ignore", invalid="ignore"):
            lr = np.subtract(a_lo, b_hi, dtype=np.float64)
            hr = np.subtract(a_hi, b_lo, dtype=np.float64)
        if np.any(~np.isfinite(lr)) or np.any(~np.isfinite(hr)):
            raise FloatingPointError("Overflow in trusted interval-array subtraction.")
        lo, hi = _nextafter(lr, -np.inf), _nextafter(hr, np.inf)
        bz = (b_lo == 0.0) & (b_hi == 0.0)
        lo = np.where(bz, a_lo, lo); hi = np.where(bz, a_hi, hi)
        return lo, hi
    lo = np.empty(a_lo.shape); hi = np.empty(a_hi.shape)
    for idx in np.ndindex(a_lo.shape):
        v = backend.sub(Interval(float(a_lo[idx]), float(a_hi[idx])), Interval(float(b_lo[idx]), float(b_hi[idx])))
        lo[idx], hi[idx] = v.lo, v.hi
    return lo, hi


def rigorous_point_interval_multiply_arrays(point, x_lo, x_hi, backend):
    point, x_lo, x_hi = np.broadcast_arrays(
        np.asarray(point, dtype=np.float64), np.asarray(x_lo, dtype=np.float64), np.asarray(x_hi, dtype=np.float64)
    )
    if _finite_rigorous(backend, point, x_lo, x_hi):
        pos = point >= 0.0
        cl = np.where(pos, x_lo, x_hi); ch = np.where(pos, x_hi, x_lo)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            lr = np.multiply(point, cl, dtype=np.float64)
            hr = np.multiply(point, ch, dtype=np.float64)
        if np.any(~np.isfinite(lr)) or np.any(~np.isfinite(hr)):
            raise FloatingPointError("Overflow in trusted point-interval array multiplication.")
        lo, hi = _nextafter(lr, -np.inf), _nextafter(hr, np.inf)
        zero = point == 0.0
        lo = np.where(zero, 0.0, lo); hi = np.where(zero, 0.0, hi)
        one = point == 1.0
        lo = np.where(one, x_lo, lo); hi = np.where(one, x_hi, hi)
        mone = point == -1.0
        lo = np.where(mone, -x_hi, lo); hi = np.where(mone, -x_lo, hi)
        return lo, hi
    lo = np.empty(point.shape); hi = np.empty(point.shape)
    for idx in np.ndindex(point.shape):
        v = backend.mul(point_interval(float(point[idx])), Interval(float(x_lo[idx]), float(x_hi[idx])))
        lo[idx], hi[idx] = v.lo, v.hi
    return lo, hi


def rigorous_interval_multiply_arrays(a_lo, a_hi, b_lo, b_hi, backend):
    a_lo, a_hi, b_lo, b_hi = np.broadcast_arrays(
        np.asarray(a_lo, dtype=np.float64), np.asarray(a_hi, dtype=np.float64),
        np.asarray(b_lo, dtype=np.float64), np.asarray(b_hi, dtype=np.float64),
    )
    if _finite_rigorous(backend, a_lo, a_hi, b_lo, b_hi):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            vals = np.stack((a_lo*b_lo, a_lo*b_hi, a_hi*b_lo, a_hi*b_hi), axis=0).astype(np.float64, copy=False)
        if np.any(~np.isfinite(vals)):
            raise FloatingPointError("Overflow in trusted interval-array multiplication.")
        lows = _nextafter(vals, -np.inf); highs = _nextafter(vals, np.inf)
        lo, hi = np.min(lows, axis=0), np.max(highs, axis=0)
        az = (a_lo == 0.0) & (a_hi == 0.0)
        bz = (b_lo == 0.0) & (b_hi == 0.0)
        zero = az | bz
        lo = np.where(zero, 0.0, lo); hi = np.where(zero, 0.0, hi)
        return lo, hi
    lo = np.empty(a_lo.shape); hi = np.empty(a_hi.shape)
    for idx in np.ndindex(a_lo.shape):
        v = backend.mul(Interval(float(a_lo[idx]), float(a_hi[idx])), Interval(float(b_lo[idx]), float(b_hi[idx])))
        lo[idx], hi[idx] = v.lo, v.hi
    return lo, hi


def rigorous_interval_divide_point_arrays(x_lo, x_hi, divisor, backend):
    x_lo, x_hi, divisor = np.broadcast_arrays(
        np.asarray(x_lo, dtype=np.float64), np.asarray(x_hi, dtype=np.float64), np.asarray(divisor, dtype=np.float64)
    )
    if np.any(divisor == 0.0):
        raise ZeroDivisionError("Point divisor array contains zero.")
    if _finite_rigorous(backend, x_lo, x_hi, divisor):
        pos = divisor > 0.0
        cl = np.where(pos, x_lo, x_hi); ch = np.where(pos, x_hi, x_lo)
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            lr = np.divide(cl, divisor, dtype=np.float64); hr = np.divide(ch, divisor, dtype=np.float64)
        if np.any(~np.isfinite(lr)) or np.any(~np.isfinite(hr)):
            raise FloatingPointError("Overflow in trusted interval/point array division.")
        lo, hi = _nextafter(lr, -np.inf), _nextafter(hr, np.inf)
        one = divisor == 1.0
        lo = np.where(one, x_lo, lo); hi = np.where(one, x_hi, hi)
        mone = divisor == -1.0
        lo = np.where(mone, -x_hi, lo); hi = np.where(mone, -x_lo, hi)
        z = (x_lo == 0.0) & (x_hi == 0.0)
        lo = np.where(z, 0.0, lo); hi = np.where(z, 0.0, hi)
        return lo, hi
    lo = np.empty(x_lo.shape); hi = np.empty(x_hi.shape)
    for idx in np.ndindex(x_lo.shape):
        v = backend.div(Interval(float(x_lo[idx]), float(x_hi[idx])), point_interval(float(divisor[idx])))
        lo[idx], hi[idx] = v.lo, v.hi
    return lo, hi


def rigorous_interval_divide_arrays(a_lo, a_hi, b_lo, b_hi, backend):
    a_lo, a_hi, b_lo, b_hi = np.broadcast_arrays(
        np.asarray(a_lo, dtype=np.float64), np.asarray(a_hi, dtype=np.float64),
        np.asarray(b_lo, dtype=np.float64), np.asarray(b_hi, dtype=np.float64),
    )
    if np.any((b_lo <= 0.0) & (b_hi >= 0.0)):
        raise ZeroDivisionError("Interval divisor array contains zero.")
    if _finite_rigorous(backend, a_lo, a_hi, b_lo, b_hi):
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            vals = np.stack((a_lo/b_lo, a_lo/b_hi, a_hi/b_lo, a_hi/b_hi), axis=0).astype(np.float64, copy=False)
        if np.any(~np.isfinite(vals)):
            raise FloatingPointError("Overflow in trusted interval-array division.")
        lo, hi = np.min(_nextafter(vals, -np.inf), axis=0), np.max(_nextafter(vals, np.inf), axis=0)
        return lo, hi
    lo = np.empty(a_lo.shape); hi = np.empty(a_hi.shape)
    for idx in np.ndindex(a_lo.shape):
        v = backend.div(Interval(float(a_lo[idx]), float(a_hi[idx])), Interval(float(b_lo[idx]), float(b_hi[idx])))
        lo[idx], hi[idx] = v.lo, v.hi
    return lo, hi


__all__ = [
    "rigorous_interval_add_arrays", "rigorous_interval_subtract_arrays",
    "rigorous_point_interval_multiply_arrays", "rigorous_interval_multiply_arrays",
    "rigorous_interval_divide_point_arrays", "rigorous_interval_divide_arrays",
]
