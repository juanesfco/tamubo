"""
Taylor-enclosure bounds on the GP posterior mean and variance with AutoBound.

The interval bounds in ``bounds.py`` bound every kernel entry k_i(x) on its own
and then combine them, which ignores that all entries depend on the same x: with
mixed-sign alpha, mu(x) = k(x)^T alpha is overestimated by ~sum_i |alpha_i| (k_hi_i
- k_lo_i). AutoBound (https://github.com/google/autobound) instead bounds
mu(x) and Q(x) = k(x)^T (K + sigma_n^2 I)^{-1} k(x) as whole functions of x:

    f(x) in f(x0) + f'(x0) (x - x0) + ... + [I] (x - x0)^degree,  x in box,

with x0 the box center and [I] a sharp interval remainder. The polynomial is then
bounded over the box. The first-order term is exact, so the error shrinks like
width^degree instead of width, and the bounds get much tighter than the interval
ones once boxes are a few splits deep. Callers intersect both.

JAX is imported lazily so the rest of ``tamubo.exactbo`` works without it. Importing
this module enables float64 in JAX (``jax_enable_x64``) and, unless already set,
stops JAX from preallocating GPU memory, which would starve CuPy's pool.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from tamubo.utils import to_numpy

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

import autobound.jax as _ab  # noqa: E402
from autobound import enclosure_arithmetic as _ea  # noqa: E402
from autobound.jax import jax_bound as _jb  # noqa: E402

__all__ = ["taylor_mu_q_bounds"]

# Jitted shapes are fixed so XLA compiles only a few programs per run: chunks of
# _SMALL_BATCH boxes (early partitions) or of batch_size boxes, and N padded up to
# a multiple of _N_PAD so the programs survive BO iterations that add points.
_SMALL_BATCH = 512
_N_PAD = 16


def _broadcast_in_dim_pushforward(intermediate, shape, broadcast_dimensions, **unused_kwargs):
    """
    AutoBound's rule for ``lax.broadcast_in_dim``, fixed for non-constant operands.

    Taylor coefficient i of an enclosure carries i trailing copies of the input
    shape; autobound 0.1.7 does not extend ``broadcast_dimensions`` over them, so
    any broadcast of a variable (e.g. ``x - X_train``) fails.
    """
    enclosure = intermediate.enclosure
    x0 = enclosure[0][0] if isinstance(enclosure[0], tuple) else enclosure[0]
    x_shape = () if len(enclosure) == 1 else tuple(enclosure[1].shape[x0.ndim:])

    def broadcast(a, i):
        extra = tuple(range(len(shape), len(shape) + i * len(x_shape)))
        return jax.lax.broadcast_in_dim(
            a, tuple(shape) + i * x_shape, tuple(broadcast_dimensions) + extra
        )

    return tuple(
        tuple(broadcast(c, i) for c in coeff) if isinstance(coeff, tuple) else broadcast(coeff, i)
        for i, coeff in enumerate(enclosure)
    )


_jb._broadcast_in_dim_pushforward_fun = _broadcast_in_dim_pushforward


@lru_cache(maxsize=None)
def _range_fn(degree: int):
    """
    Jitted ``(boxes_L, boxes_U, gp arrays) -> (lo, hi)`` with lo/hi of shape (n, 2)
    bounding [mu(x), Q(x)] over each box. GP arrays are arguments, not closure
    constants, so one compilation serves every GP with the same (N, d) and chunk size.
    """

    def per_box(box_L, box_U, X_train, alpha, K_inv, length_scale, sigma_f_2):
        # AutoBound supports neither stack nor concatenate; combine outputs with basis vectors.
        e_mu = jnp.array([1.0, 0.0])
        e_q = jnp.array([0.0, 1.0])

        def mu_q(x):
            diff = (x - X_train) / length_scale
            k = sigma_f_2 * jnp.exp(-0.5 * jnp.sum(diff * diff, axis=1))  # (N,)
            return e_mu * (k @ alpha) + e_q * (k @ (K_inv @ k))

        x0 = 0.5 * (box_L + box_U)
        enclosure = _ab.taylor_bounds(mu_q, degree)(x0, (box_L, box_U)).coefficients
        # Bound the Taylor polynomial over the box: a degree-0 enclosure is an interval.
        lo, hi = _ea.enclose_enclosure(enclosure, (box_L - x0, box_U - x0), 0, jnp)[0]
        return lo, hi

    return jax.jit(jax.vmap(per_box, in_axes=(0, 0, None, None, None, None, None)))


def _to_jax(array, device):
    if type(array).__module__.split(".")[0] == "cupy":
        # Zero-copy; the array stays on its GPU.
        return jax.dlpack.from_dlpack(array)
    return jax.device_put(np.asarray(array), device)


def taylor_mu_q_bounds(
    bounds_L,
    bounds_U,
    X_train,
    alpha,
    K_inv,
    length_scale,
    sigma_f_2: float,
    *,
    xp,
    degree: int = 2,
    batch_size: int = 8192,
):
    """
    Bound mu(x) = k(x)^T alpha and Q(x) = k(x)^T K_inv k(x) over each box.

    Parameters
    ----------
    bounds_L, bounds_U : np.ndarray or cupy.ndarray, shape (n, d)
        Box lower/upper corners.
    X_train : array, shape (N, d)
        Training inputs.
    alpha : array, shape (N,)
        GP dual coefficients in the standardized target space.
    K_inv : array, shape (N, N)
        (K + sigma_n^2 I)^{-1} of the trained GP.
    length_scale : array, shape (d,)
        RBF length scales.
    sigma_f_2 : float
        Kernel variance.
    xp : module
        numpy or cupy; outputs are returned in this module.
    degree : int, default=2
        Taylor degree. Degree 3 is tighter but costs ~60x more at d=10.
    batch_size : int, default=8192
        Boxes per jitted call. Degree-2 memory is ~O(batch_size * N * d^2). Calls
        with at most 512 boxes use a 512-box chunk instead; chunks are padded, so
        each run compiles at most two programs.

    Returns
    -------
    (mu_lo, mu_hi, q_lo, q_hi) : arrays of shape (n,) in ``xp``.
    """
    n, d = int(bounds_L.shape[0]), int(bounds_L.shape[1])
    if xp is np:
        device = jax.devices("cpu")[0]
    else:
        device = jax.devices("gpu")[xp.cuda.Device().id]
        # Hand-off to JAX: make earlier CuPy kernels visible to JAX's stream.
        xp.cuda.get_current_stream().synchronize()

    # Pad the training set with copies of its first point whose alpha and K_inv
    # rows/columns are 0: their terms are exactly 0 in mu and Q, enclosures included.
    X_train, alpha, K_inv, length_scale = (
        np.asarray(to_numpy(a), dtype=np.float64) for a in (X_train, alpha, K_inv, length_scale)
    )
    N = X_train.shape[0]
    N_pad = -(-N // _N_PAD) * _N_PAD
    X_pad = np.concatenate([X_train, np.repeat(X_train[:1], N_pad - N, axis=0)])
    alpha_pad = np.zeros(N_pad)
    alpha_pad[:N] = alpha.reshape(-1)
    K_inv_pad = np.zeros((N_pad, N_pad))
    K_inv_pad[:N, :N] = K_inv
    gp_args = tuple(
        jax.device_put(a, device) for a in (X_pad, alpha_pad, K_inv_pad, length_scale)
    ) + (float(sigma_f_2),)
    fn = _range_fn(int(degree))

    chunk = _SMALL_BATCH if n <= _SMALL_BATCH else max(1, int(batch_size))
    lo_out = xp.empty((n, 2), dtype=xp.float64)
    hi_out = xp.empty((n, 2), dtype=xp.float64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        box_L = xp.ascontiguousarray(bounds_L[start:end], dtype=xp.float64)
        box_U = xp.ascontiguousarray(bounds_U[start:end], dtype=xp.float64)
        if end - start < chunk:
            # Pad with copies of the first box so the jitted shape never changes.
            pad = chunk - (end - start)
            box_L = xp.concatenate([box_L, xp.broadcast_to(box_L[:1], (pad, d))])
            box_U = xp.concatenate([box_U, xp.broadcast_to(box_U[:1], (pad, d))])
            if xp is not np:
                xp.cuda.get_current_stream().synchronize()
        lo, hi = fn(_to_jax(box_L, device), _to_jax(box_U, device), *gp_args)
        if xp is np:
            lo_out[start:end] = np.asarray(lo)[: end - start]
            hi_out[start:end] = np.asarray(hi)[: end - start]
        else:
            lo_out[start:end] = xp.from_dlpack(lo)[: end - start]
            hi_out[start:end] = xp.from_dlpack(hi)[: end - start]
    return lo_out[:, 0], hi_out[:, 0], lo_out[:, 1], hi_out[:, 1]
