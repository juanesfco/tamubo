"""
Subsystem 3: kernel bounds from the distance bounds, with Google's AutoBound.

The RBF kernel is a function of the squared scaled distance r alone,

    k(r) = sigma_f^2 exp(-r / 2),   r in [r_lo, r_hi]   (subsystem 2),

so each kernel entry k(x, X_i) over a box lies in an interval of k over [r_lo, r_hi].
Two ways to get it:

  exact:     k is decreasing in r, so k in [k(r_hi), k(r_lo)] (what tamubo's
             rbf_k_bounds computes).
  AutoBound: a Taylor enclosure of k around r0 = (r_lo + r_hi) / 2,
             k(r) in k(r0) + k'(r0)(r - r0) + ... + [I](r - r0)^degree, bounded over
             [r_lo, r_hi]. Degree 0 is interval arithmetic (exp is monotone, so exact
             here); higher degrees bound the polynomial term by term and are slightly
             wider for a 1-D monotone function.

For a single entry the exact range cannot be beaten. AutoBound pays off one level up,
on mu(x) = k(x)^T alpha and Q(x) = k(x)^T K^{-1} k(x) as whole functions of x, where the
per-entry intervals ignore that every entry depends on the same x (subsystems 4 and 5).

Run: python development/exactbo_subsystems/s3_kernel_bounds.py [--backend numpy]
"""

import argparse

from common import (add_common_args, autobound_range, best_time, get_module, get_xp, load_autobound, make_gp,
                    random_boxes, sample_in_boxes, to_numpy)
from s2_distance_bounds import sq_distance_bounds


def kernel_bounds(d2_lo, d2_hi, sigma_f_2):
    """Exact range of sigma_f^2 exp(-r/2) over r in [d2_lo, d2_hi] (elementwise)."""
    xp = get_module(d2_lo)
    return sigma_f_2 * xp.exp(-0.5 * d2_hi), sigma_f_2 * xp.exp(-0.5 * d2_lo)


def autobound_kernel_bounds(sigma_f_2, degree):
    """(d2_lo, d2_hi) -> AutoBound range of sigma_f^2 exp(-r/2) over [d2_lo, d2_hi] (elementwise)."""
    _, jnp, _, _ = load_autobound()
    rng = autobound_range(lambda r: sigma_f_2 * jnp.exp(-0.5 * r), degree)

    def run(d2_lo, d2_hi):
        lo, hi = rng(d2_lo.reshape(-1), d2_hi.reshape(-1))
        return lo.reshape(d2_lo.shape), hi.reshape(d2_hi.shape)

    return run


def check(xp, gp, d, seed):
    print("== check ==")
    L, U = random_boxes(xp, 200, d, seed, depth=(0, 3))
    d2_lo, d2_hi = sq_distance_bounds(L, U, gp.X, gp.length_scale)
    k_lo, k_hi = kernel_bounds(d2_lo, d2_hi, gp.sigma_f_2)

    pts = sample_in_boxes(xp, L, U, 256, seed)
    k = gp.kernel(pts.reshape(-1, d)).reshape(pts.shape[0], pts.shape[1], -1)  # (n, m, N)
    ok = bool(xp.all(k >= k_lo[:, None] - 1e-15) and xp.all(k <= k_hi[:, None] + 1e-15))
    print(f"  sampled kernel values inside the exact bounds: {ok}")

    for degree in (0, 1, 2, 3):
        a_lo, a_hi = autobound_kernel_bounds(gp.sigma_f_2, degree)(d2_lo, d2_hi)
        # AutoBound does not round outward: allow a few ulps.
        valid = bool(xp.all(a_lo <= k_lo + 1e-12) and xp.all(a_hi >= k_hi - 1e-12))
        excess = (a_hi - a_lo) - (k_hi - k_lo)
        print(f"  AutoBound degree {degree}: contains the exact range (up to 1e-12): {valid}, "
              f"width excess median {float(xp.median(excess)):.2e} max {float(xp.max(excess)):.2e}")

    try:
        from tamubo.exactbo.bounds import rbf_k_bounds
    except ImportError:
        print("  tamubo not importable, skipping comparison")
        return
    n, N = L.shape[0], gp.X.shape[0]
    backend = "numpy" if get_module(L).__name__ == "numpy" else "cupy"
    ref = [rbf_k_bounds(L, U, gp.X[i], n, d, gp.sigma_f_2, gp.length_scale, backend=backend) for i in range(N)]
    err = max(float(xp.max(xp.abs(xp.stack([r[j] for r in ref], axis=1) - b))) for j, b in enumerate((k_lo, k_hi)))
    print(f"  max difference to tamubo rbf_k_bounds: {err:.1e}")


def bench(xp, gp, n, d, seed):
    print("== benchmark ==")
    L, U = random_boxes(xp, n, d, seed)
    N = gp.X.shape[0]
    d2_lo, d2_hi = sq_distance_bounds(L, U, gp.X, gp.length_scale)
    t = best_time(lambda: kernel_bounds(d2_lo, d2_hi, gp.sigma_f_2), xp)
    print(f"  exact:              {n} x {N} entries in {t * 1e3:8.2f} ms ({n * N / t / 1e9:.2f} G entries/s)")
    for degree in (0, 2):
        fn = autobound_kernel_bounds(gp.sigma_f_2, degree)
        t = best_time(lambda: fn(d2_lo, d2_hi), xp)
        print(f"  AutoBound degree {degree}: {n} x {N} entries in {t * 1e3:8.2f} ms ({n * N / t / 1e9:.2f} G entries/s)")


if __name__ == "__main__":
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    xp = get_xp(args.backend)
    print(f"backend={xp.__name__} n={args.n} N={args.N} d={args.d}")
    gp = make_gp(xp, args.d, args.N, args.seed)
    check(xp, gp, args.d, args.seed)
    bench(xp, gp, args.n, args.d, args.seed)
