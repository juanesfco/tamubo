"""
How far is ExactBO's incumbent from the true maximum EI?

Rebuilds the GP of one BO iteration from an experiment-2 log (the printed
training data), maximizes EI (or log EI) in ExactBO's standardized space with
multi-start L-BFGS-B, and compares it with the incumbent ExactBO had reached
in that iteration's partitions (center sampling only raises it slowly).

    python max_ei_probe.py ../experiment2/logs/exp2-677272.out 20 --acquisition ei
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiment3"))

from common import load_problem, _build_default_gp  # noqa: E402

from tamubo.acquisition_functions import expected_improvement, log_expected_improvement  # noqa: E402


def parse_iteration(log_path: Path, iteration: int):
    """Training data and per-partition incumbents of one iteration in an experiment-2 log."""
    text = log_path.read_text()
    blocks = re.split(r"^Iteration (\d+)/\d+\n", text, flags=re.M)
    block = dict(zip(map(int, blocks[1::2]), blocks[2::2]))[iteration]
    number = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    y = np.array([float(v) for v in re.findall(number, re.search(r"y: \[(.*?)\]", block, re.S).group(1))])
    X_text = re.search(r"X: \[(.*?)\], \ny:", block, re.S).group(1)
    X = np.array([float(v) for v in re.findall(number, X_text)]).reshape(len(y), -1)
    kernel = re.search(r"GP kernel after fitting: (.*)", block).group(1)
    incumbents = [float(v) for v in re.findall(r"Incumbent: ([-0-9.e]+)", block)]
    return X, y, kernel, incumbents


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("iteration", type=int)
    ap.add_argument("--problem", default="problem10d")
    ap.add_argument("--acquisition", default="logei", choices=["logei", "ei"])
    ap.add_argument("--starts", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X_phys, y, kernel_logged, incumbents = parse_iteration(args.log, args.iteration)
    bounds = np.asarray(load_problem(args.problem).bounds, dtype=float)
    X = (X_phys - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    d = X.shape[1]
    gp = _build_default_gp(d)
    gp.fit(X, y)
    print(f"N = {len(y)}\nlogged kernel: {kernel_logged}\nrefit kernel:  {gp.kernel_}")

    sigma_n_2 = gp.kernel_.get_params()["k2__noise_level"]
    y_mean, y_std = float(np.ravel(gp._y_train_mean)[0]), float(np.ravel(gp._y_train_std)[0])
    y_min_scaled = float(np.min(gp.y_train_))
    acq = log_expected_improvement if args.acquisition == "logei" else expected_improvement

    def score(x):
        # As ExactBO samples: standardized posterior, latent (noise-free) sigma.
        mu, sd = gp.predict(np.atleast_2d(x), return_std=True)
        mu, sd = (mu - y_mean) / y_std, sd / y_std
        return acq(mu, np.sqrt(np.clip(sd**2 - sigma_n_2, 1e-12, None)), y_min_scaled, backend="numpy")

    rng = np.random.default_rng(args.seed)
    starts = np.vstack([rng.random((args.starts, d)), np.full((1, d), 0.5)])
    best_x, best = None, -np.inf
    for x0 in starts:
        res = minimize(lambda x: -float(score(x)[0]), x0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * d)
        if -res.fun > best:
            best_x, best = res.x, -res.fun
    print(f"ExactBO incumbent by partition (last 5): {incumbents[-5:]}")
    print(f"max {args.acquisition} found by L-BFGS-B ({len(starts)} starts): {best:.6f} at {np.round(best_x, 4)}")
    if incumbents:
        gap = best - incumbents[-1]
        print(f"gap to the last incumbent: {gap:.6f}" + (f" ({np.exp(gap):.2f}x in EI)" if args.acquisition == "logei"
                                                          else f" ({best / incumbents[-1]:.2f}x)"))


if __name__ == "__main__":
    main()
