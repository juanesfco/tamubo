"""Shared helpers for experiment 3: the experiment-2 GP, test boxes and a reference posterior."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

# Allow running these scripts directly from the repository without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
# Problems, initial design and GP come from experiment 2.
sys.path.insert(0, str(REPO_ROOT / "experiments" / "exactbo" / "experiment2"))

from problems import load_problem  # noqa: E402
from run_experiment import _build_default_gp, _resolve_initial_design, _set_seed  # noqa: E402
from tamubo.utils import _normalize_problem_to_unit_cube  # noqa: E402


def make_gp(problem: str = "problem10d", n0: int = 32, seed: int = 0):
    """Fit the experiment-2 GP on the unit cube, as ExactBO sees it on its first iteration."""
    _set_seed(seed)
    spec = load_problem(problem)
    X0 = _resolve_initial_design(n0, default_X0=spec.X0, bounds=spec.bounds)
    X, bounds, objective, _ = _normalize_problem_to_unit_cube(X0, spec.bounds, spec.objective)
    gp = _build_default_gp(X.shape[1])
    gp.fit(X, objective(X))
    return gp, X, bounds


def gp_arrays(gp, X) -> dict:
    """Trained-GP quantities in the standardized target space."""
    params = gp.kernel_.get_params()
    return dict(
        X=np.asarray(X, dtype=float),
        alpha=np.asarray(gp.alpha_, dtype=float).ravel(),
        L=np.asarray(gp.L_, dtype=float),
        ls=np.asarray(params["k1__k2__length_scale"], dtype=float),
        sf2=float(params["k1__k1__constant_value"]),
        ymin=float(np.min(gp.y_train_)),
    )


def random_boxes(n: int, d: int, depth: int, rng):
    """
    DIRECT-like boxes with `depth` trisections spread over the dimensions: each side
    is 3^-k with k in {depth // d, depth // d + 1}, at a random lattice position.
    Partition p of ExactBO produces boxes of depth p to p * d.
    """
    k = depth // d + (rng.random((n, d)) < (depth % d) / d).astype(int)
    w = 3.0 ** (-k)
    lo = np.floor(rng.random((n, d)) / w) * w
    return lo, lo + w


def sampled_posterior(P: dict, lo, hi, m: int, rng):
    """Latent posterior mean/std at m uniform points plus the center of each box: (n, m + 1)."""
    n, d = lo.shape
    u = np.concatenate([rng.random((n, m, d)), np.full((n, 1, d), 0.5)], axis=1)
    pts = (lo[:, None, :] + u * (hi - lo)[:, None, :]).reshape(-1, d)
    mu = np.empty(pts.shape[0])
    var = np.empty(pts.shape[0])
    for s in range(0, pts.shape[0], 100_000):
        x = pts[s:s + 100_000]
        k = P["sf2"] * np.exp(-0.5 * (((x[:, None, :] - P["X"][None]) / P["ls"]) ** 2).sum(-1))
        v = solve_triangular(P["L"], k.T, lower=True)
        mu[s:s + 100_000] = k @ P["alpha"]
        var[s:s + 100_000] = P["sf2"] - (v * v).sum(0)
    sd = np.sqrt(np.maximum(var, 1e-12))
    return mu.reshape(n, m + 1), sd.reshape(n, m + 1)
