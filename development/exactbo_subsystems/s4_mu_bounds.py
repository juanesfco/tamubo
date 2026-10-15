"""
Subsystem 4: bounds on the GP posterior mean over a box.

    mu(x) = k(x)^T alpha,   alpha = (K + sigma_n^2 I)^{-1} y

Two bounds, intersected:

  interval:  mu is linear in the kernel entries, each in [k_lo_i, k_hi_i] (subsystem 3),
             so split alpha by sign:
               mu_lo = k_lo . alpha+ + k_hi . alpha-,   mu_hi = k_hi . alpha+ + k_lo . alpha-
             Exact for the kernel box, but the kernel box is not exact for mu: every entry
             depends on the same x, so with mixed-sign alpha this overestimates mu by
             ~ sum_i |alpha_i| (k_hi_i - k_lo_i), which only shrinks linearly with the width.
  AutoBound: degree-2 Taylor enclosure of mu(x) as one function of x around the box
             center; the remainder shrinks like width^2 (width^degree).

On large boxes the interval bound can win, on small ones AutoBound wins by orders of
magnitude, hence the intersection.

Run: python development/exactbo_subsystems/s4_mu_bounds.py [--backend numpy]
"""

import argparse

import numpy as np

from common import (add_common_args, autobound_range, best_time, get_module, get_xp, load_autobound, make_gp,
                    random_boxes, sample_in_boxes, to_numpy)
from s2_distance_bounds import sq_distance_bounds
from s3_kernel_bounds import kernel_bounds


def mu_bounds_interval(k_lo, k_hi, alpha):
    """Sign split of alpha over the kernel-entry intervals. k_lo, k_hi: (n, N)."""
    xp = get_module(k_lo)
    a_pos, a_neg = xp.maximum(alpha, 0.0), xp.minimum(alpha, 0.0)
    return k_lo @ a_pos + k_hi @ a_neg, k_hi @ a_pos + k_lo @ a_neg


def gp_jax(gp):
    """The GP arrays as JAX arrays, and k(x) for one point x of shape (d,)."""
    _, jnp, _, _ = load_autobound()
    X, ls, alpha, K_inv = (jnp.asarray(to_numpy(a)) for a in (gp.X, gp.length_scale, gp.alpha, gp.K_inv))

    def k(x):
        diff = (x - X) / ls
        return gp.sigma_f_2 * jnp.exp(-0.5 * jnp.sum(diff * diff, axis=1))  # (N,)

    return k, alpha, K_inv


def mu_bounds_autobound(gp, degree=2):
    """(L, U) -> AutoBound range of mu(x) over each box."""
    k, alpha, _ = gp_jax(gp)
    return autobound_range(lambda x: k(x) @ alpha, degree)


def mu_bounds(gp, L, U, ab=None):
    """Intersection of both bounds; pass ab = mu_bounds_autobound(gp) to reuse its compilation."""
    xp = get_module(L)
    k_lo, k_hi = kernel_bounds(*sq_distance_bounds(L, U, gp.X, gp.length_scale), gp.sigma_f_2)
    lo, hi = mu_bounds_interval(k_lo, k_hi, gp.alpha)
    if ab is not None:
        a_lo, a_hi = ab(L, U)
        lo, hi = xp.maximum(lo, a_lo), xp.minimum(hi, a_hi)
    return lo, hi


def check(xp, gp, d, seed):
    print("== check ==")
    ab = mu_bounds_autobound(gp)
    print("  overestimate = bound width / sampled range of mu (>= 1; 1 is tight), median over 200 boxes")
    print(f"  {'width':>10} {'interval':>10} {'AutoBound':>10} {'both':>10}  valid")
    for depth in range(0, 9, 2):
        L, U = random_boxes(xp, 200, d, seed, depth=(depth, depth))
        pts = sample_in_boxes(xp, L, U, 512, seed)
        mu, _ = gp.posterior(pts.reshape(-1, d))
        mu = mu.reshape(pts.shape[:2])
        s_lo, s_hi = mu.min(axis=1), mu.max(axis=1)
        k_lo, k_hi = kernel_bounds(*sq_distance_bounds(L, U, gp.X, gp.length_scale), gp.sigma_f_2)
        i_lo, i_hi = mu_bounds_interval(k_lo, k_hi, gp.alpha)
        a_lo, a_hi = ab(L, U)
        b_lo, b_hi = xp.maximum(i_lo, a_lo), xp.minimum(i_hi, a_hi)
        tol = 1e-9 * (1 + xp.abs(s_hi))
        valid = bool(xp.all(b_lo <= s_lo + tol) and xp.all(b_hi >= s_hi - tol))
        ratio = lambda lo, hi: float(xp.median((hi - lo) / (s_hi - s_lo)))
        print(f"  {'3^-%d' % depth:>10} {ratio(i_lo, i_hi):10.3g} {ratio(a_lo, a_hi):10.3g} "
              f"{ratio(b_lo, b_hi):10.3g}  {valid}")

    try:
        from tamubo.exactbo.bounds import mu_bounds as ref_mu_bounds
        from tamubo.exactbo.autobound_bounds import taylor_mu_q_bounds
    except ImportError:
        print("  tamubo not importable, skipping comparison")
        return
    L, U = random_boxes(xp, 200, d, seed, depth=(0, 4))
    n, N = L.shape[0], gp.X.shape[0]
    k_lo, k_hi = kernel_bounds(*sq_distance_bounds(L, U, gp.X, gp.length_scale), gp.sigma_f_2)
    backend = "numpy" if xp is np else "cupy"
    ref = ref_mu_bounds(gp.alpha, k_lo, k_hi, n, N, scaled_output=True, backend=backend)
    ref_ab = taylor_mu_q_bounds(L, U, gp.X, gp.alpha, gp.K_inv, gp.length_scale, gp.sigma_f_2, xp=xp)[:2]
    err = max(float(xp.max(xp.abs(a - b))) for a, b in zip(mu_bounds_interval(k_lo, k_hi, gp.alpha), ref))
    err_ab = max(float(xp.max(xp.abs(a - b))) for a, b in zip(ab(L, U), ref_ab))
    print(f"  max difference to tamubo mu_bounds: {err:.1e}, to taylor_mu_q_bounds: {err_ab:.1e}")


def bench(xp, gp, n, d, seed):
    print("== benchmark ==")
    L, U = random_boxes(xp, n, d, seed)
    k_lo, k_hi = kernel_bounds(*sq_distance_bounds(L, U, gp.X, gp.length_scale), gp.sigma_f_2)
    t = best_time(lambda: mu_bounds_interval(k_lo, k_hi, gp.alpha), xp)
    print(f"  interval (given kernel bounds): {n} boxes in {t * 1e3:8.2f} ms ({n / t / 1e6:.2f} M boxes/s)")
    ab = mu_bounds_autobound(gp)
    t = best_time(lambda: ab(L, U), xp, repeats=3)
    print(f"  AutoBound degree 2:             {n} boxes in {t * 1e3:8.2f} ms ({n / t / 1e6:.3f} M boxes/s)")


if __name__ == "__main__":
    args = add_common_args(argparse.ArgumentParser(description=__doc__), n=2**13).parse_args()
    xp = get_xp(args.backend)
    print(f"backend={xp.__name__} n={args.n} N={args.N} d={args.d}")
    gp = make_gp(xp, args.d, args.N, args.seed)
    check(xp, gp, args.d, args.seed)
    bench(xp, gp, args.n, args.d, args.seed)
