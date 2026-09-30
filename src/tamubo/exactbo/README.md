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
- `bounds.py`: GP kernel/mean/standard-deviation/EI interval bounds.
- `autobound_bounds.py`: AutoBound (JAX) Taylor-enclosure bounds on the
  posterior mean and variance; imported only when `bound_method="autobound"`.
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

### Acquisition: log EI

`acquisition="logei"` (default) scores sampled points and bounds boxes with log
EI, computed stably after LogEI (Ament et al., NeurIPS 2023): EI underflows to
0 (and cancels catastrophically before that) once z = (y_min - mu)/sigma drops
below ~-38, which in 10-d covers most of the space far from the data; log EI
stays finite and ordered there. `epsilon_ei` is then a tolerance on log EI,
i.e. relative: a box is pruned once its upper bound is below
`exp(epsilon_ei)` times the incumbent EI (`0.1` ~ 10.5%). `acquisition="ei"`
keeps the absolute tolerance in the GP's standardized target space.

### Bounds

Each box gets an upper bound on the acquisition over the box; boxes whose
bound is within `epsilon_ei` of the incumbent (the best sampled value over all
partitions so far) are pruned. Upper bounds are built as:

1. Kernel entries: exact per-entry range of `k(x, x_i)` over the box.
2. Mean: `k^T alpha` split by the sign of alpha (interval), intersected with
   the AutoBound range (below).
3. Variance: `Q = ||L^{-1} k||^2`. Each `v_j = (L^{-1} k)_j` is linear in k, so
   its range over the kernel box is exact from the sign split of `L^{-1}`
   (unlike interval forward substitution, whose intervals widen row by row);
   also `Q >= ||k_lo||^2 / lambda_max(K + sigma_n^2 I)`. Intersected with the
   AutoBound range of Q.
4. EI: EI is decreasing in mu and increasing in sigma, so its exact range over
   the (mu, sigma) rectangle is `[EI(mu_hi, sig_lo), EI(mu_lo, sig_hi)]`.

`bound_method="autobound"` (default) adds Taylor enclosures from
[AutoBound](https://github.com/google/autobound) of mu(x) and Q(x) as whole
functions of x around the box center. Per-entry intervals ignore that all
kernel entries depend on the same x, so with mixed-sign alpha they overestimate
mu by ~sum |alpha_i| (k_hi_i - k_lo_i), which only shrinks linearly with the box
width; the Taylor remainder shrinks like width^degree. On the 10-d experiment-2
GP, the mean bound's overestimate shrinks ~9x at 10 trisections, ~70x at 20
and ~250x at 50 compared with the interval bound (on the largest boxes both are
loose and the interval one can win, hence the intersection). `autobound_degree=3`
is tighter still but ~60x slower at d=10. The first call per run compiles two
XLA programs (a few seconds); set `JAX_COMPILATION_CACHE_DIR` to reuse them
across runs. `bound_method="interval"` skips AutoBound (no JAX needed).

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
- `predict_batch_size`, `bounds_batch_size`, `autobound_batch_size` and
  `max_target_boxes` are exposed for memory/performance control on large runs.
  `max_target_boxes` keeps the best boxes by sampled EI, so a run that hits the
  cap is a heuristic search, not an exact one.
- With `logMask=True`, `log["partitions"]` records per-partition box counts,
  the largest upper bound, the incumbent and the elapsed time. See
  `experiments/exactbo/experiment3/` for the bound-tightness study.
- `plot_f(...)` works independently for 2D problems.
- `plot_log(...)` and `plot_opt(...)` expect partition snapshots (`p0`, `p1`,
  ...) in the log structure. The current `exactbo(..., logMask=True)` runner
  emits per-iteration summaries, not full partition snapshots.
