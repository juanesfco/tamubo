"""Shared pieces of the subsystem scripts: backend, a toy GP, random boxes, timing, AutoBound."""

import time
from dataclasses import dataclass

import numpy as np


def get_xp(backend="auto"):
    """CuPy when a GPU is visible (or required with backend="cupy"), else NumPy."""
    if backend == "numpy":
        return np
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            return cp
    except Exception:
        if backend == "cupy":
            raise
    return np


def to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def sync(xp):
    if xp is not np:
        xp.cuda.Device().synchronize()


def best_time(fn, xp, repeats=5):
    """Fastest of `repeats` runs of fn() in seconds, after one warm-up run."""
    fn()
    sync(xp)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync(xp)
        times.append(time.perf_counter() - start)
    return min(times)


def add_common_args(parser, n=2**14, N=64, d=5):
    parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "cupy"])
    parser.add_argument("--n", type=int, default=n, help="number of boxes in the benchmark")
    parser.add_argument("--N", type=int, default=N, help="number of GP training points")
    parser.add_argument("--d", type=int, default=d, help="dimension")
    parser.add_argument("--seed", type=int, default=0)
    return parser


# ---------------------------------------------------------------------------------------
# Toy GP: RBF kernel with anisotropic length scales, trained in the standardized target
# space (like sklearn's normalize_y=True). Everything the bounds need is precomputed.
# ---------------------------------------------------------------------------------------

@dataclass
class GP:
    xp: object
    X: object            # (N, d) training inputs in [0, 1]^d
    y: object            # (N,) standardized targets
    length_scale: object # (d,)
    sigma_f_2: float     # kernel variance
    sigma_n_2: float     # noise variance
    L: object            # cholesky(K + sigma_n^2 I), lower
    L_inv: object        # L^{-1}
    K_inv: object        # (K + sigma_n^2 I)^{-1}
    alpha: object        # K_inv @ y
    lambda_max: float    # largest eigenvalue of K + sigma_n^2 I
    y_min: float         # best (lowest) standardized target

    def kernel(self, x, X=None):
        """k(x, X) for points x of shape (m, d): (m, N)."""
        xp = self.xp
        X = self.X if X is None else X
        diff = (x[:, None, :] - X[None, :, :]) / self.length_scale
        return self.sigma_f_2 * xp.exp(-0.5 * xp.sum(diff * diff, axis=2))

    def posterior(self, x):
        """Mean and latent std (no noise) at points x of shape (m, d)."""
        xp = self.xp
        k = self.kernel(x)
        mu = k @ self.alpha
        v = k @ self.L_inv.T  # rows are (L^{-1} k(x))^T
        var = self.sigma_f_2 - xp.sum(v * v, axis=1)
        return mu, xp.sqrt(xp.maximum(var, 1e-12))


def make_gp(xp, d=5, N=64, seed=0, sigma_n_2=1e-4):
    """GP fitted (fixed hyperparameters) to N random points of a smooth test function."""
    rng = np.random.default_rng(seed)
    X = rng.random((N, d))
    # Length scales from 0.15 to 1.5: the first dimensions matter, the last are nearly flat.
    length_scale = np.geomspace(0.15, 1.5, d)
    y = np.sum(np.sin(3.0 * X / length_scale), axis=1) + 0.5 * np.sum((X - 0.3) ** 2, axis=1)
    y = (y - y.mean()) / y.std()
    sigma_f_2 = 1.0

    diff = (X[:, None, :] - X[None, :, :]) / length_scale
    K = sigma_f_2 * np.exp(-0.5 * np.sum(diff * diff, axis=2)) + sigma_n_2 * np.eye(N)
    L = np.linalg.cholesky(K)
    L_inv = np.linalg.solve(L, np.eye(N))
    K_inv = L_inv.T @ L_inv
    alpha = K_inv @ y
    lambda_max = float(np.linalg.eigvalsh(K)[-1])

    a = lambda arr: xp.asarray(arr, dtype=xp.float64)
    return GP(xp, a(X), a(y), a(length_scale), sigma_f_2, sigma_n_2, a(L), a(L_inv), a(K_inv), a(alpha),
              lambda_max, float(y.min()))


# ---------------------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------------------

def random_boxes(xp, n, d, seed=0, depth=(1, 6)):
    """
    n random boxes inside [0, 1]^d with DIRECT-like widths 3^-k, k uniform in
    [depth[0], depth[1]] per dimension.
    """
    rng = np.random.default_rng(seed)
    width = 3.0 ** -rng.integers(depth[0], depth[1] + 1, size=(n, d))
    L = rng.random((n, d)) * (1.0 - width)
    return xp.asarray(L), xp.asarray(L + width)


def sample_in_boxes(xp, L, U, m, seed=0):
    """m uniform points per box, shape (n, m, d), plus the box center as point 0."""
    rng = np.random.default_rng(seed)
    n, d = L.shape
    t = xp.asarray(rng.random((n, m, d)))
    t[:, 0, :] = 0.5
    return L[:, None, :] + t * (U - L)[:, None, :]


# ---------------------------------------------------------------------------------------
# AutoBound
# ---------------------------------------------------------------------------------------

def load_autobound():
    """
    (jax, jnp, autobound.jax, autobound.enclosure_arithmetic).

    Imported through tamubo.exactbo.autobound_bounds, which enables float64 in JAX,
    stops JAX from preallocating GPU memory (it would starve CuPy's pool) and patches
    autobound 0.1.7's broadcast rule, which fails on e.g. x - X_train.
    """
    import tamubo.exactbo.autobound_bounds  # noqa: F401  (side effects above)
    import jax
    import jax.numpy as jnp
    import autobound.jax as ab
    from autobound import enclosure_arithmetic as ea
    return jax, jnp, ab, ea


def autobound_range(f, degree=2):
    """
    Jitted (L, U, *args) -> (lo, hi): for each box, an interval containing f(x, *args)
    for every x in the box, from AutoBound's degree-`degree` Taylor enclosure around the
    box center,

        f(x) in f(x0) + f'(x0)(x - x0) + ... + [I] (x - x0)^degree,

    whose polynomial is then bounded over the box. L, U have shape (n, d), or (n,) for
    scalar x; optional args are per-box arrays of shape (n, ...); f maps one x to a
    scalar or a vector, lo/hi have shape (n,) + f's shape.
    """
    jax, jnp, ab, ea = load_autobound()

    def per_box(box_L, box_U, *args):
        x0 = 0.5 * (box_L + box_U)
        coeffs = ab.taylor_bounds(lambda x: f(x, *args), degree)(x0, (box_L, box_U)).coefficients
        # A degree-0 enclosure of the polynomial over the box is an interval.
        return ea.enclose_enclosure(coeffs, (box_L - x0, box_U - x0), 0, jnp)[0]

    fn = jax.jit(jax.vmap(per_box))

    def run(L, U, *args):
        xp = get_module(L)
        # Pad the number of boxes to a power of two (copies of box 0): jit compiles once
        # per shape, so this caps the compilations at a few per run.
        n = L.shape[0]
        n_pad = max(256, 1 << (n - 1).bit_length())
        arrays = [xp.concatenate([a, xp.repeat(a[:1], n_pad - n, axis=0)]) for a in (L, U) + args]
        if xp is np:
            lo, hi = fn(*(jnp.asarray(a) for a in arrays))
            return np.asarray(lo)[:n], np.asarray(hi)[:n]
        # Zero-copy hand-off to JAX on the same GPU (fresh arrays are contiguous and aligned).
        xp.cuda.get_current_stream().synchronize()
        lo, hi = fn(*(jax.dlpack.from_dlpack(a) for a in arrays))
        return xp.from_dlpack(lo)[:n], xp.from_dlpack(hi)[:n]

    return run


def get_module(a):
    return np if isinstance(a, np.ndarray) else __import__("cupy")
