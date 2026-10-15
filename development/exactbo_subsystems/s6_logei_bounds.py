"""
Subsystem 6: log EI and its bounds over a box.

EI for minimization, with z = (y_min - mu) / sigma:

    EI(mu, sigma) = sigma * h(z),   h(z) = phi(z) + z Phi(z)

Evaluating h directly cancels catastrophically for z << 0 (both terms ~ phi(z)/|z|) and
underflows to 0 below z ~ -38, which in high dimension is most of the space far from
the data. log EI = log sigma + log h(z) is evaluated stably after LogEI (Ament et al.,
NeurIPS 2023):

    z > -1:           log(phi(z) + z Phi(z))                         (no cancellation)
    -100 <= z <= -1:  log phi(z) + log(1 - |z| sqrt(pi/2) erfcx(|z|/sqrt 2))
    z < -100:         log phi(z) - 2 log|z| + log(1 - 3/z^2 + 15/z^4 - 105/z^6 + 945/z^8)

Bounds: EI is decreasing in mu (dEI/dmu = -Phi(z)) and increasing in sigma
(dEI/dsigma = phi(z)), so over the rectangle [mu_lo, mu_hi] x [sig_lo, sig_hi]
(subsystems 4 and 5) its exact range is

    [EI(mu_hi, sig_lo), EI(mu_lo, sig_hi)],

and log is monotone, so the same corners bound log EI. The search only uses the upper
bound: a box is pruned when it cannot beat the incumbent by more than epsilon.

Run: python development/exactbo_subsystems/s6_logei_bounds.py [--backend numpy]
"""

import argparse
import math

import numpy as np

from common import add_common_args, best_time, get_module, get_xp, make_gp, random_boxes, sample_in_boxes
from s4_mu_bounds import mu_bounds, mu_bounds_autobound
from s5_sigma_bounds import qs_bounds_autobound, sigma_bounds


def _special(xp):
    if xp is np:
        from scipy.special import erfcx, ndtr
    else:
        from cupyx.scipy.special import erfcx, ndtr
    return ndtr, erfcx


def log_h(z):
    xp = get_module(z)
    ndtr, erfcx = _special(xp)
    mid, far = z > -1.0, z < -100.0
    # Each branch is evaluated on a safe placeholder where it does not apply.
    zm = xp.where(mid, z, 0.0)
    out_mid = xp.log(xp.exp(-0.5 * zm * zm) / math.sqrt(2 * math.pi) + zm * ndtr(zm))
    a = -xp.where(mid | far, -1.0, z)  # |z| in [1, 100]
    x = xp.log(a * erfcx(a / math.sqrt(2))) + 0.5 * math.log(math.pi / 2)  # log(|z| sqrt(pi/2) erfcx), < 0
    log1mexp = xp.where(x > -math.log(2), xp.log(-xp.expm1(x)), xp.log1p(-xp.exp(x)))
    out_tail = -0.5 * a * a - 0.5 * math.log(2 * math.pi) + log1mexp
    zf = xp.where(far, z, -100.0)
    r = 1.0 / (zf * zf)
    out_far = (-0.5 * zf * zf - 0.5 * math.log(2 * math.pi) - 2.0 * xp.log(-zf)
               + xp.log1p(r * (-3.0 + r * (15.0 + r * (-105.0 + r * 945.0)))))
    return xp.where(mid, out_mid, xp.where(far, out_far, out_tail))


def log_ei(mu, sigma, y_min, pad=1e-12):
    """log EI; for sigma < pad the limit log(max(y_min - mu, 0))."""
    xp = get_module(mu)
    tiny = sigma < pad
    s = xp.where(tiny, 1.0, sigma)
    with np.errstate(divide="ignore"):
        return xp.where(tiny, xp.log(xp.maximum(y_min - mu, 0.0)), xp.log(s) + log_h((y_min - mu) / s))


def logei_bounds(mu_lo, mu_hi, sig_lo, sig_hi, y_min):
    return log_ei(mu_hi, sig_lo, y_min), log_ei(mu_lo, sig_hi, y_min)


def box_logei_bounds(gp, L, U, ab_mu=None, ab_qs=None):
    """The full chain for boxes: distances -> kernel -> mu, sigma -> log EI."""
    mu_lo, mu_hi = mu_bounds(gp, L, U, ab_mu)
    sig_lo, sig_hi = sigma_bounds(gp, L, U, ab_qs, "ABCD" if ab_qs is not None else "ABC")
    return logei_bounds(mu_lo, mu_hi, sig_lo, sig_hi, gp.y_min)


def check(xp, gp, d, seed):
    print("== check ==")
    # 1. log h against 50-digit arithmetic.
    import mpmath
    mpmath.mp.dps = 50
    z = np.concatenate([-np.geomspace(1e4, 1e-3, 400), np.linspace(0, 10, 50)])
    exact = np.array([float(mpmath.log(mpmath.npdf(t) + t * mpmath.ncdf(t))) for t in z])
    mine = np.asarray(log_h(xp.asarray(z))).astype(float) if xp is np else log_h(xp.asarray(z)).get()
    with np.errstate(divide="ignore"):
        from scipy.stats import norm
        naive = np.log(norm.pdf(z) + z * norm.cdf(z))
    rel = lambda v: np.abs(v - exact) / np.abs(exact).clip(1e-300)
    finite = np.isfinite(naive)
    print(f"  log h, z in [-1e4, 10]: stable max rel error {rel(mine).max():.1e}; naive log(phi + z Phi): "
          f"max rel error {rel(naive)[finite].max():.1e} where finite, "
          f"-inf below z = {z[~finite].max():.1f}")

    # 2. The corners bound log EI over the (mu, sigma) rectangle.
    rng = np.random.default_rng(seed)
    mu_lo = xp.asarray(rng.normal(0, 3, 1000))
    mu_hi = mu_lo + xp.asarray(rng.exponential(1.0, 1000))
    sig_lo = xp.asarray(rng.exponential(0.3, 1000))
    sig_hi = sig_lo + xp.asarray(rng.exponential(0.3, 1000))
    lo, hi = logei_bounds(mu_lo, mu_hi, sig_lo, sig_hi, 0.0)
    t = xp.asarray(rng.random((1000, 64, 2)))
    v = log_ei(mu_lo[:, None] + t[..., 0] * (mu_hi - mu_lo)[:, None],
               sig_lo[:, None] + t[..., 1] * (sig_hi - sig_lo)[:, None], 0.0)
    print(f"  rectangle: sampled log EI inside the corner bounds: "
          f"{bool(xp.all(v >= lo[:, None] - 1e-9) and xp.all(v <= hi[:, None] + 1e-9))}")

    # 3. The full chain on boxes, against log EI sampled in each box.
    ab_mu, ab_qs = mu_bounds_autobound(gp), qs_bounds_autobound(gp)
    print("  boxes: upper bound - sampled max log EI (nats; 0 is tight), median over 200 boxes")
    print(f"  {'width':>8} {'interval':>10} {'+AutoBound':>11}  valid")
    for depth in range(0, 9, 2):
        L, U = random_boxes(xp, 200, d, seed, depth=(depth, depth))
        pts = sample_in_boxes(xp, L, U, 512, seed)
        mu, sig = gp.posterior(pts.reshape(-1, d))
        sampled = log_ei(mu, sig, gp.y_min).reshape(pts.shape[:2]).max(axis=1)
        gaps, valid = [], True
        for abs_ in ((None, None), (ab_mu, ab_qs)):
            ub = box_logei_bounds(gp, L, U, *abs_)[1]
            valid &= bool(xp.all(ub >= sampled - 1e-7))
            gaps.append(float(xp.median(ub - sampled)))
        print(f"  {'3^-%d' % depth:>8} {gaps[0]:10.3g} {gaps[1]:11.3g}  {valid}")

    try:
        from tamubo.exactbo.bounds import ei_bounds
    except ImportError:
        print("  tamubo not importable, skipping comparison")
        return
    ref = ei_bounds(mu_lo, mu_hi, sig_lo, sig_hi, 1000, 0.0, log=True, backend="numpy" if xp is np else "cupy")
    print(f"  max difference to tamubo ei_bounds(log=True): "
          f"{max(float(xp.max(xp.abs(a - b))) for a, b in zip((lo, hi), ref)):.1e}")


def bench(xp, gp, n, d, seed):
    print("== benchmark ==")
    z = xp.asarray(np.random.default_rng(seed).uniform(-200, 5, n))
    t = best_time(lambda: log_h(z), xp)
    print(f"  log h:                        {n} values in {t * 1e3:8.2f} ms ({n / t / 1e6:.2f} M/s)")
    L, U = random_boxes(xp, n, d, seed)
    t = best_time(lambda: box_logei_bounds(gp, L, U), xp, repeats=3)
    print(f"  box chain, interval (A+B+C):  {n} boxes in {t * 1e3:8.2f} ms ({n / t / 1e6:.3f} M boxes/s)")
    ab_mu, ab_qs = mu_bounds_autobound(gp), qs_bounds_autobound(gp)
    t = best_time(lambda: box_logei_bounds(gp, L, U, ab_mu, ab_qs), xp, repeats=3)
    print(f"  box chain, + AutoBound:       {n} boxes in {t * 1e3:8.2f} ms ({n / t / 1e6:.3f} M boxes/s)")


if __name__ == "__main__":
    args = add_common_args(argparse.ArgumentParser(description=__doc__), n=2**13).parse_args()
    xp = get_xp(args.backend)
    print(f"backend={xp.__name__} n={args.n} N={args.N} d={args.d}")
    gp = make_gp(xp, args.d, args.N, args.seed)
    check(xp, gp, args.d, args.seed)
    bench(xp, gp, args.n, args.d, args.seed)
