# `tamubo.exactbo`

`tamubo.exactbo` implements Exact Bayesian Optimization with a DIRECT-style
partition search.

## Public API

```python
from tamubo.exactbo import (
    exactbo,
    exactbo_partitioning,
    split_boxes,
    rbf_k_bounds,
    mu_bounds,
    sigma_bounds,
    ei_bounds,
    optimize_acqf_exactbo,
    run_botorch_exactbo_ei,
    plot_f,
    plot_log,
    plot_opt,
)
```

- `exactbo(...)`: full BO loop. Fits the GP each iteration, runs ExactBO
  partitioning to pick the next point, evaluates the objective, and returns a
  `BOResult` with `X`, `y`, `backend`, and optional `log`.
- `exactbo_partitioning(...)`: partition-only step. Expects evaluated points
  `X` plus an already fitted GP and returns the next candidate in `BOResult.X`.
- `split_boxes(...)`: cuPyNumeric DIRECT-style box splitting utility.
- `rbf_k_bounds(...)`, `mu_bounds(...)`, `sigma_bounds(...)`, `ei_bounds(...)`:
  interval-bound helpers used by the partition search.
- `optimize_acqf_exactbo(...)`: BoTorch adapter that uses ExactBO to optimize a
  standard analytic EI-style acquisition function.
- `run_botorch_exactbo_ei(...)`: full Torch/BoTorch workflow that builds a
  BoTorch GP and uses ExactBO instead of `optimize_acqf(...)`.
- `plot_f(...)`, `plot_log(...)`, `plot_opt(...)`: 2D visualization helpers.

## Package Layout

- `run.py`: ExactBO loop and partitioning implementation.
- `partition.py`: cuPyNumeric box splitting.
- `bounds.py`: cuPyNumeric GP kernel/mean/standard-deviation/EI bound propagation.
- `torch_bounds.py`, `torch_partition.py`: Torch implementation of the ExactBO
  bound propagation and partition search.
- `botorch.py`, `torch_run.py`: BoTorch-facing optimizer and full workflow.
- `plot2D.py`: 2D plotting and animation helpers.
- `__init__.py`: package exports.

## Runtime Assumptions

- The sklearn-based ExactBO path now requires `cupynumeric`.
- `backend="auto"` is still accepted for API compatibility, but it must resolve
  to `cupynumeric`.
- The old NumPy backend and the memory-tuning controls
  `predict_batch_size`, `bounds_batch_size`, and `max_target_boxes` were removed.

## Minimal Example

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from tamubo.exactbo import exactbo

bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
X0 = np.array(
    [
        [0.25, 0.25],
        [0.25, 0.75],
        [0.75, 0.25],
        [0.75, 0.75],
    ],
    dtype=float,
)

kernel = (
    ConstantKernel(1.0) * RBF(length_scale=np.full(2, 0.2))
    + WhiteKernel(noise_level=1e-3)
)
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)

def objective(X):
    X = np.asarray(X, dtype=float)
    X = X.reshape(1, -1) if X.ndim == 1 else X
    return np.sum((X - 0.5) ** 2, axis=1)

result = exactbo(
    X0=X0,
    bounds=bounds,
    epsilon_X=0.05,
    epsilon_ei=1e-4,
    gp=gp,
    f=objective,
    max_iters=5,
    max_partitions=20,
    backend="auto",
    logMask=True,
)

print(result.backend.selected)
print(result.X.shape, result.y.shape)
```

## Notes

- `f` should accept an array with shape `(N, d)` and return one value per row.
- `exactbo(...)` fits `gp` internally on every outer iteration.
- `exactbo_partitioning(...)` assumes `gp` has already been fitted and currently
  relies on the scikit-learn `GaussianProcessRegressor` attributes used in this
  repository, including kernel parameters exposed as
  `ConstantKernel * RBF + WhiteKernel`.
- `epsilon_X` may be a scalar or a per-dimension array with shape `(d,)`.
- `normalize_to_unit_cube=True` runs the internal search on `[0, 1]^d` while
  still evaluating the objective in the original finite bounds.
- `optimize_acqf_exactbo(...)` currently supports analytic single-point
  `ExpectedImprovement` and `LogExpectedImprovement` with `maximize=False`.
- `plot_f(...)` works independently for 2D problems.
- `plot_log(...)` and `plot_opt(...)` expect partition snapshots (`p0`, `p1`,
  ...) in the log structure. The current `exactbo(..., logMask=True)` runner
  emits per-iteration summaries, not full partition snapshots.
