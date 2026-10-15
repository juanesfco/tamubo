"""
Subsystem 5: bounds on the GP posterior (latent) standard deviation over a box.

    sigma^2(x) = sigma_f^2 - Q(x),   Q(x) = k(x)^T (K + sigma_n^2 I)^{-1} k(x) = ||L^{-1} k(x)||^2

sigma_hi needs a LOWER bound on Q and sigma_lo an UPPER bound. Bounds on Q, intersected:

  (A) sign split: v = L^{-1} k is linear in k, so each v_j is bounded exactly over the
      kernel box by splitting L^{-1} by sign; then Q_lo = sum_j min v_j^2 (0 if v_j can
      be 0), Q_hi = sum_j max v_j^2.
  (B) eigenvalue: Q >= ||k||^2 / lambda_max(K + sigma_n^2 I) >= ||k_lo||^2 / lambda_max.
  (C) Cauchy-Schwarz: for any w, (k^T w)^2 <= Q * w^T K w. With w = K^{-1} k(x0) at the
      box center, w^T K w = Q(x0) and Q(x) >= (k(x)^T w)^2 / Q(x0), exact at x0. Only the
      LINEAR function s(x) = k(x)^T w has to be bounded from below (sign split of w, like
      the mean, plus AutoBound). This is what makes sigma_hi tight.
  (D) AutoBound: degree-2 Taylor enclosures of Q(x) and s(x) as functions of x.

Run: python development/exactbo_subsystems/s5_sigma_bounds.py [--backend numpy]
"""

import argparse

import numpy as np

from common import (add_common_args, autobound_range, best_time, get_module, get_xp, make_gp, random_boxes,
                    sample_in_boxes)
from s2_distance_bounds import sq_distance_bounds
from s3_kernel_bounds import kernel_bounds
from s4_mu_bounds import gp_jax


def q_bounds_sign_split(k_lo, k_hi, L_inv):
    """(A): (Q_lo, Q_hi) from the exact range of v = L^{-1} k over the kernel box."""
    xp = get_module(k_lo)
    P, M = xp.maximum(L_inv, 0.0).T, xp.minimum(L_inv, 0.0).T
    v_lo = k_lo @ P + k_hi @ M
    v_hi = k_hi @ P + k_lo @ M
    q_hi = xp.sum(xp.maximum(v_lo ** 2, v_hi ** 2), axis=1)
    q_lo = xp.sum(xp.maximum(v_lo, 0.0) ** 2 + xp.minimum(v_hi, 0.0) ** 2, axis=1)
    return q_lo, q_hi


def q_lower_eig(k_lo, lambda_max):
    """(B)"""
    xp = get_module(k_lo)
    return xp.sum(k_lo * k_lo, axis=1) / lambda_max


def cs_weights(gp, L, U):
    """(C): w = K^{-1} k(x0) per box (n, N) and Q(x0) = k(x0)^T w (n,)."""
    xp = get_module(L)
    k0 = gp.kernel(0.5 * (L + U))
    w = k0 @ gp.K_inv
    return w, xp.sum(k0 * w, axis=1)


def q_lower_cs(k_lo, k_hi, w, q0, s_lo=None):
    """(C): Q >= max(s_lo, 0)^2 / Q(x0), with s_lo a lower bound on k(x)^T w."""
    xp = get_module(k_lo)
    s = xp.sum(xp.maximum(w, 0.0) * k_lo + xp.minimum(w, 0.0) * k_hi, axis=1)
    if s_lo is not None:
        s = xp.maximum(s, s_lo)
    return xp.maximum(s, 0.0) ** 2 / q0


def qs_bounds_autobound(gp, degree=2):
    """(D): (L, U, w) -> AutoBound ranges of [Q(x), k(x)^T w], each (n, 2)."""
    k, _, K_inv = gp_jax(gp)
    e_q, e_s = np.array([1.0, 0.0]), np.array([0.0, 1.0])

    def q_and_s(x, w):
        kx = k(x)
        # AutoBound supports neither stack nor concatenate: combine outputs with basis vectors.
        return e_q * (kx @ (K_inv @ kx)) + e_s * (kx @ w)

    return autobound_range(q_and_s, degree)


def sigma_bounds(gp, L, U, ab=None, parts="ABCD"):
    """(sig_lo, sig_hi) per box, from the Q bounds in `parts`; ab = qs_bounds_autobound(gp) for D."""
    xp = get_module(L)
    k_lo, k_hi = kernel_bounds(*sq_distance_bounds(L, U, gp.X, gp.length_scale), gp.sigma_f_2)
    q_lo, q_hi = q_bounds_sign_split(k_lo, k_hi, gp.L_inv)
    if "B" in parts:
        q_lo = xp.maximum(q_lo, q_lower_eig(k_lo, gp.lambda_max))
    s_lo = None
    if "D" in parts:
        w, q0 = cs_weights(gp, L, U)
        lo, hi = ab(L, U, w)
        q_lo, q_hi, s_lo = xp.maximum(q_lo, lo[:, 0]), xp.minimum(q_hi, hi[:, 0]), lo[:, 1]
    if "C" in parts:
        w, q0 = cs_weights(gp, L, U)
        q_lo = xp.maximum(q_lo, q_lower_cs(k_lo, k_hi, w, q0, s_lo))
    sig = lambda q: xp.sqrt(xp.maximum(gp.sigma_f_2 - q, 1e-12))
    return sig(q_hi), sig(q_lo)


def check(xp, gp, d, seed):
    print("== check ==")
    ab = qs_bounds_autobound(gp)
    variants = ["A", "AB", "ABC", "ABCD"]
    print("  sigma_hi - sampled max sigma (0 is tight; sigma_f = 1), median over 200 boxes")
    print(f"  {'width':>8} " + " ".join(f"{v:>9}" for v in variants) + "  valid  (sigma_lo: sampled min - bound, ABCD)")
    for depth in range(0, 9, 2):
        L, U = random_boxes(xp, 200, d, seed, depth=(depth, depth))
        pts = sample_in_boxes(xp, L, U, 512, seed)
        _, sig = gp.posterior(pts.reshape(-1, d))
        sig = sig.reshape(pts.shape[:2])
        s_lo, s_hi = sig.min(axis=1), sig.max(axis=1)
        gaps, valid = [], True
        for parts in variants:
            lo, hi = sigma_bounds(gp, L, U, ab, parts)
            valid &= bool(xp.all(lo <= s_lo + 1e-7) and xp.all(hi >= s_hi - 1e-7))
            gaps.append(float(xp.median(hi - s_hi)))
        print(f"  {'3^-%d' % depth:>8} " + " ".join(f"{g:9.2e}" for g in gaps)
              + f"  {valid!s:5}  {float(xp.median(s_lo - lo)):.2e}")

    try:
        from tamubo.exactbo.bounds import sigma_bounds as ref_sigma_bounds, q_cs_weights, q_cs_lower
    except ImportError:
        print("  tamubo not importable, skipping comparison")
        return
    L, U = random_boxes(xp, 200, d, seed, depth=(0, 4))
    n, N = L.shape[0], gp.X.shape[0]
    backend = "numpy" if xp is np else "cupy"
    k_lo, k_hi = kernel_bounds(*sq_distance_bounds(L, U, gp.X, gp.length_scale), gp.sigma_f_2)
    w, q0 = q_cs_weights(L, U, gp.X, gp.length_scale, gp.sigma_f_2, gp.K_inv, backend=backend)
    q_extra = (q_cs_lower(k_lo, k_hi, w, q0, backend=backend), xp.full(n, xp.inf))
    ref = ref_sigma_bounds(k_lo, k_hi, gp.L, n, N, gp.sigma_f_2, scaled_output=True, backend=backend,
                           L_inv=gp.L_inv, lambda_max=gp.lambda_max, q_bounds=q_extra)
    err = max(float(xp.max(xp.abs(a - b))) for a, b in zip(sigma_bounds(gp, L, U, parts="ABC"), ref))
    print(f"  max difference to tamubo sigma_bounds (A+B+C): {err:.1e}")


def bench(xp, gp, n, d, seed):
    print("== benchmark ==")
    L, U = random_boxes(xp, n, d, seed)
    k_lo, k_hi = kernel_bounds(*sq_distance_bounds(L, U, gp.X, gp.length_scale), gp.sigma_f_2)
    w, q0 = cs_weights(gp, L, U)
    ab = qs_bounds_autobound(gp)
    for name, fn in [("(A) sign split, 2 GEMMs", lambda: q_bounds_sign_split(k_lo, k_hi, gp.L_inv)),
                     ("(B) eigenvalue", lambda: q_lower_eig(k_lo, gp.lambda_max)),
                     ("(C) Cauchy-Schwarz weights", lambda: cs_weights(gp, L, U)),
                     ("(C) Cauchy-Schwarz bound", lambda: q_lower_cs(k_lo, k_hi, w, q0)),
                     ("(D) AutoBound Q and s", lambda: ab(L, U, w))]:
        t = best_time(fn, xp, repeats=3)
        print(f"  {name:28}: {n} boxes in {t * 1e3:8.2f} ms ({n / t / 1e6:.3f} M boxes/s)")


if __name__ == "__main__":
    args = add_common_args(argparse.ArgumentParser(description=__doc__), n=2**13).parse_args()
    xp = get_xp(args.backend)
    print(f"backend={xp.__name__} n={args.n} N={args.N} d={args.d}")
    gp = make_gp(xp, args.d, args.N, args.seed)
    check(xp, gp, args.d, args.seed)
    bench(xp, gp, args.n, args.d, args.seed)
