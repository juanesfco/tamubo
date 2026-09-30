# `tamubo.exactbo`

`tamubo.exactbo` implements Exact Bayesian Optimization with a DIRECT-style
partition search and runtime backend selection.

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
- `split_boxes(...)`: backend-dispatched DIRECT-style box splitting utility.
- `rbf_k_bounds(...)`, `mu_bounds(...)`, `sigma_bounds(...)`, `ei_bounds(...)`:
  interval-bound helpers used by the partition search.
- `plot_f(...)`, `plot_log(...)`, `plot_opt(...)`: 2D visualization helpers.

## Package Layout

- `run.py`: ExactBO loop and partitioning implementation.
- `partition.py`: NumPy and CuPy box splitting behind `split_boxes(...)`.
- `bounds.py`: GP kernel/mean/standard-deviation/EI bound propagation.
- `plot2D.py`: 2D plotting and animation helpers.
- `__init__.py`: package exports.

## Backends

The current runner accepts:

- `backend="numpy"`: sequential NumPy path.
- `backend="cupy"`: vectorized CuPy path.
- `backend="auto"`: chooses CuPy when a GPU is visible, otherwise NumPy.

### Multiple GPUs (CuPy)

`exactbo(..., n_gpus=None)` splits the per-box work — EI upper bounds and the
2^d sampled points per box, which dominate the run time — row-wise across the
visible GPUs of one node, one thread per GPU in a single process. Box storage,
reductions and splitting stay on the current GPU. `n_gpus=None` uses every
visible GPU (under Slurm: the GPUs allocated to the job); `predict_batch_size`
and `bounds_batch_size` apply per GPU.

### Box sampling

`box_sampling` sets where EI is sampled inside each analyzed/active box:

- `"lhs"` (default): 2^d centered Latin-hypercube points per box.
- `"center"`: the box center only — 2^d times fewer GP posterior evaluations
  (1024x at d=10).

Peak sampling memory per GPU is set by `predict_batch_size` (points per
posterior call). Center sampling lowers it only when a GPU's share of sampled
boxes is below that cap; with multiple GPUs each GPU gets 1/n of the boxes, so
`"center"` plus several GPUs is what brings large 10-d searches within memory.

### NumPy vs CuPy

Box splitting breaks ties between equal-width dimensions with `argsort`.
CuPy's `argsort` is stable; NumPy's default batched `argsort` is not, so boxes
with tied widths can be split in a different dimension order (equally valid
children, possibly a different trajectory).

The resolved backend is available on `result.backend.selected` when using
`exactbo(...)`.

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
- `predict_batch_size`, `bounds_batch_size`, and `max_target_boxes` are exposed
  for memory/performance control on large runs.
- `plot_f(...)` works independently for 2D problems.
- `plot_log(...)` and `plot_opt(...)` expect partition snapshots (`p0`, `p1`,
  ...) in the log structure. The current `exactbo(..., logMask=True)` runner
  emits per-iteration summaries, not full partition snapshots.
