"""
Soundness and tightness tests for the ExactBO box bounds.

Every bound is checked against the GP posterior sampled densely inside random
DIRECT-like boxes: a valid bound must contain every sampled value. Uses its own
small 4-d GP, so it runs in seconds (the AutoBound tests are skipped without
``autobound``):

    python -m pytest experiments/exactbo/experiment3/test_bounds.py
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_triangular
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from tamubo.acquisition_functions import expected_improvement, log_expected_improvement
from tamubo.acquisition_functions.ei import log_h
from tamubo.exactbo.bounds import ei_bounds, mu_bounds, rbf_k_bounds, sigma_bounds
from tamubo.utils import has_cupy, to_numpy

BACKENDS = ["numpy"] + (["cupy"] if has_cupy() else [])
TOL = 1e-9


def _xp(backend):
    if backend == "cupy":
        import cupy

        return cupy
    return np


@pytest.fixture(scope="module")
def gp_problem():
    """A 4-d GP with mixed-sign alpha (the case that makes interval mu bounds loose)."""
    rng = np.random.default_rng(0)
    d, n0 = 4, 24
    X = rng.random((n0, d))
    y = np.sin(6 * X[:, 0]) + np.cos(4 * X[:, 1]) * X[:, 2] + 0.5 * X[:, 3] ** 2
    kernel = ConstantKernel(1.0) * RBF(np.full(d, 0.3)) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True).fit(X, y)
    prm = gp.kernel_.get_params()
    return dict(
        X=X,
        alpha=np.asarray(gp.alpha_).ravel(),
        L=np.asarray(gp.L_),
        ls=np.asarray(prm["k1__k2__length_scale"], dtype=float),
        sf2=float(prm["k1__k1__constant_value"]),
        ymin=float(np.min(gp.y_train_)),
    )


def _random_boxes(n, d, depth, rng):
    """Boxes with sides 3^-k (DIRECT trisection depth ~depth) at random lattice positions."""
    k = depth // d + (rng.random((n, d)) < (depth % d) / d).astype(int)
    w = 3.0 ** (-k)
    lo = np.floor(rng.random((n, d)) / w) * w
    return lo, lo + w


def _sampled_posterior(P, lo, hi, m, rng):
    """Latent posterior mean/std at m uniform points plus the center of each box: (n, m+1)."""
    u = np.concatenate([rng.random((lo.shape[0], m, lo.shape[1])), np.full((lo.shape[0], 1, lo.shape[1]), 0.5)], 1)
    pts = (lo[:, None, :] + u * (hi - lo)[:, None, :]).reshape(-1, lo.shape[1])
    k = P["sf2"] * np.exp(-0.5 * (((pts[:, None, :] - P["X"][None]) / P["ls"]) ** 2).sum(-1))
    v = solve_triangular(P["L"], k.T, lower=True)
    mu = (k @ P["alpha"]).reshape(lo.shape[0], -1)
    sd = np.sqrt(np.maximum(P["sf2"] - (v * v).sum(0), 1e-12)).reshape(lo.shape[0], -1)
    return mu, sd


def _interval_bounds(P, lo, hi, backend, **sigma_kwargs):
    xp = _xp(backend)
    n, d = lo.shape
    N = P["X"].shape[0]
    L_, U_ = xp.asarray(lo), xp.asarray(hi)
    K_lo = xp.empty((n, N))
    K_hi = xp.empty((n, N))
    for i in range(N):
        K_lo[:, i], K_hi[:, i] = rbf_k_bounds(L_, U_, xp.asarray(P["X"][i]), n, d, P["sf2"], xp.asarray(P["ls"]), backend=backend)
    mu_lo, mu_hi = mu_bounds(xp.asarray(P["alpha"]), K_lo, K_hi, n, N, scaled_output=True, backend=backend)
    sig_lo, sig_hi = sigma_bounds(K_lo, K_hi, xp.asarray(P["L"]), n, N, P["sf2"], scaled_output=True, backend=backend, **sigma_kwargs)
    return mu_lo, mu_hi, sig_lo, sig_hi


def _ei(mu, sd, ymin):
    z = (ymin - mu) / sd
    return (ymin - mu) * norm.cdf(z) + sd * norm.pdf(z)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("depth", [0, 3, 8, 16, 30])
def test_interval_bounds_are_sound(gp_problem, backend, depth):
    P = gp_problem
    rng = np.random.default_rng(depth)
    lo, hi = _random_boxes(48, P["X"].shape[1], depth, rng)
    mu_s, sd_s = _sampled_posterior(P, lo, hi, 400, rng)
    mu_lo, mu_hi, sig_lo, sig_hi = _interval_bounds(P, lo, hi, backend)
    for log in (False, True):
        ei_lo, ei_hi = ei_bounds(mu_lo, mu_hi, sig_lo, sig_hi, 48, P["ymin"], backend=backend, log=log)
        ei_lo, ei_hi = to_numpy(ei_lo), to_numpy(ei_hi)
        ei_s = _ei(mu_s, sd_s, P["ymin"])
        if log:
            ei_lo, ei_hi = np.exp(ei_lo), np.exp(ei_hi)
        assert np.all(ei_hi >= ei_s.max(1) - TOL)
        assert np.all(ei_lo <= ei_s.min(1) + TOL)
    assert np.all(to_numpy(mu_lo) <= mu_s.min(1) + TOL) and np.all(to_numpy(mu_hi) >= mu_s.max(1) - TOL)
    assert np.all(to_numpy(sig_lo) <= sd_s.min(1) + TOL) and np.all(to_numpy(sig_hi) >= sd_s.max(1) - TOL)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sigma_bound_not_looser_than_forward_substitution(gp_problem, backend):
    """Bound (A) is the exact range of each v_j, so it contains no more than interval forward substitution."""
    P = gp_problem
    rng = np.random.default_rng(1)
    lo, hi = _random_boxes(64, P["X"].shape[1], 10, rng)
    xp = _xp(backend)
    n, N = 64, P["X"].shape[0]
    K_lo = xp.empty((n, N))
    K_hi = xp.empty((n, N))
    for i in range(N):
        K_lo[:, i], K_hi[:, i] = rbf_k_bounds(xp.asarray(lo), xp.asarray(hi), xp.asarray(P["X"][i]), n, lo.shape[1], P["sf2"], xp.asarray(P["ls"]), backend=backend)
    K_lo, K_hi = to_numpy(K_lo), to_numpy(K_hi)

    # Reference: interval forward substitution L v = k.
    L = P["L"]
    q_lo_fs = np.zeros(n)
    for b in range(n):
        v_lo, v_hi = np.zeros(N), np.zeros(N)
        for j in range(N):
            s_lo = sum(min(L[j, k] * v_lo[k], L[j, k] * v_hi[k]) for k in range(j))
            s_hi = sum(max(L[j, k] * v_lo[k], L[j, k] * v_hi[k]) for k in range(j))
            v_lo[j] = (K_lo[b, j] - s_hi) / L[j, j]
            v_hi[j] = (K_hi[b, j] - s_lo) / L[j, j]
        q_lo_fs[b] = np.sum(np.where((v_lo < 0) & (v_hi > 0), 0.0, np.minimum(v_lo**2, v_hi**2)))
    sig_hi_fs = np.sqrt(np.maximum(P["sf2"] - q_lo_fs, 1e-12))

    _, sig_hi = sigma_bounds(xp.asarray(K_lo), xp.asarray(K_hi), xp.asarray(L), n, N, P["sf2"], scaled_output=True, backend=backend)
    assert np.all(to_numpy(sig_hi) <= sig_hi_fs + 1e-12)


def test_ei_bound_is_exact_on_corners():
    """With point intervals the bound collapses to EI itself."""
    mu = np.linspace(-3, 3, 41)
    sd = np.linspace(0.05, 2, 41)
    ei_lo, ei_hi = ei_bounds(mu, mu, sd, sd, 41, 0.0, backend="numpy")
    ref = expected_improvement(mu, sd, 0.0, backend="numpy")
    np.testing.assert_allclose(ei_lo, ref, rtol=1e-12, atol=1e-300)
    np.testing.assert_allclose(ei_hi, ref, rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("backend", BACKENDS)
def test_log_h_is_stable_and_monotone(backend):
    z = np.concatenate([-np.logspace(-6, 12, 500), np.linspace(-5, 5, 101), np.logspace(-6, 2, 50)])
    z.sort()
    v = to_numpy(log_h(z, backend=backend))
    assert np.all(np.isfinite(v))
    assert np.all(np.diff(v) >= 0)
    # Accurate where the naive formula is (z > -5).
    m = z > -5
    naive = np.log(norm.pdf(z[m]) + z[m] * norm.cdf(z[m]))
    np.testing.assert_allclose(v[m], naive, rtol=1e-10, atol=1e-12)
    # Tail: log h(z) = log phi(z) - 2 log|z| + log(1 - 3/z^2 + ...), continuous across branches.
    zt = np.array([-1e6])
    np.testing.assert_allclose(to_numpy(log_h(zt, backend=backend)), norm.logpdf(zt) - 2 * np.log(-zt) - 3e-12, rtol=1e-15)
    # No jump at the branch switches: the change across them matches the slope -z - 2/z.
    for edge in (-1.0, -100.0):
        za, zb = edge * (1 + 1e-9), edge * (1 - 1e-9)
        a, b = to_numpy(log_h(np.array([za, zb]), backend=backend))
        slope = norm.cdf(edge) / (norm.pdf(edge) + edge * norm.cdf(edge)) if edge > -50 else -edge - 2 / edge  # Phi / h
        assert abs((b - a) - slope * (zb - za)) < 1e-6 * abs(slope * (zb - za))


def test_log_ei_matches_ei_and_survives_underflow():
    mu = np.linspace(-2, 1, 31)
    sd = np.full(31, 0.3)
    np.testing.assert_allclose(
        np.exp(log_expected_improvement(mu, sd, 0.0, backend="numpy")),
        expected_improvement(mu, sd, 0.0, backend="numpy"),
        rtol=1e-11,
    )
    # z = -40: EI underflows to 0, log EI stays finite and ordered.
    far = log_expected_improvement(np.array([12.0, 13.0]), np.array([0.3, 0.3]), 0.0, backend="numpy")
    assert np.all(np.isfinite(far)) and far[0] > far[1]
    assert expected_improvement(np.array([12.0]), np.array([0.3]), 0.0, backend="numpy")[0] == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("depth", [0, 8, 20, 40])
def test_autobound_bounds_are_sound_and_tighten_mu(gp_problem, backend, depth):
    pytest.importorskip("autobound")
    from tamubo.exactbo.autobound_bounds import taylor_mu_q_bounds

    P = gp_problem
    xp = _xp(backend)
    rng = np.random.default_rng(100 + depth)
    lo, hi = _random_boxes(40, P["X"].shape[1], depth, rng)
    mu_s, sd_s = _sampled_posterior(P, lo, hi, 400, rng)
    q_s = P["sf2"] - sd_s**2
    K_inv = np.linalg.inv(P["L"] @ P["L"].T)
    mu_lo, mu_hi, q_lo, q_hi = (
        to_numpy(a)
        for a in taylor_mu_q_bounds(xp.asarray(lo), xp.asarray(hi), P["X"], P["alpha"], K_inv, P["ls"], P["sf2"], xp=xp)
    )
    assert np.all(mu_lo <= mu_s.min(1) + TOL) and np.all(mu_hi >= mu_s.max(1) - TOL)
    assert np.all(q_lo <= q_s.min(1) + TOL) and np.all(q_hi >= q_s.max(1) - TOL)

    # Intersected with the interval bounds, the result is still sound and never looser.
    i_mu_lo, i_mu_hi, _, i_sig_hi = (to_numpy(a) for a in _interval_bounds(P, lo, hi, backend))
    _, sig_hi = _interval_bounds(P, lo, hi, backend, q_bounds=(xp.asarray(q_lo), xp.asarray(q_hi)))[2:]
    sig_hi = to_numpy(sig_hi)
    assert np.all(sig_hi <= i_sig_hi + 1e-12) and np.all(sig_hi >= sd_s.max(1) - TOL)
    if depth >= 20:
        # Deep boxes: the Taylor enclosure is much tighter than per-entry intervals.
        assert np.median(mu_s.min(1) - mu_lo) < 0.5 * np.median(mu_s.min(1) - i_mu_lo)
