# Native ExactBO implementation guide

This guide explains how the native code maps to the Python reference in
`src/tamubo/exactbo/`. Read the Python and native functions in the order shown
below; that is also the order in which one acquisition step executes.

## What runs where

Python still owns the parts that do not benefit from a custom CUDA allocator:

1. evaluate the objective;
2. fit `sklearn.gaussian_process.GaussianProcessRegressor`;
3. serialize the fitted GP state;
4. launch `exactbo_partitioning`;
5. evaluate and append the returned point.

The native executable owns one ExactBO acquisition/partitioning step. MPI gives
each rank a contiguous slice of the current box list. Each rank chooses a GPU
from its node-local MPI rank, not its global rank.

The standalone bounds and split executables are diagnostic programs. The full
BO loop only launches `exactbo_partitioning`, which contains the same formulas
and split kernel internally.

## Data model

A box is stored as two row-major arrays:

- `boxes_L[box * d + dim]`: lower coordinate;
- `boxes_U[box * d + dim]`: upper coordinate.

The GP arrays are the fitted sklearn state, in standardized-output space:

- `X_train`: training inputs;
- `alpha`: `gp.alpha_`;
- `L`: lower Cholesky factor `gp.L_`;
- `length_scale` and `sigma_f_2`: fitted RBF parameters;
- `y_min_scaled`: minimum of `gp.y_train_`.

Mean, variance, and EI must stay in this standardized space. The native
sampling kernel computes latent variance directly as
`sigma_f_2 - dot(v, v)`, so it correctly excludes WhiteKernel observation
noise. `sigma_n_2`, `y_train_mean`, and `y_train_std` remain in the file format
for compatibility and reporting but are not needed by the native acquisition.

## One partition iteration

The loop in `partitioning.cu` mirrors `exactbo_partitioning()` in `run.py`:

1. `compute_ei_hi_global()` computes a conservative EI upper bound per box.
2. Boxes within `epsilon_ei` of the largest upper bound are analyzed. Children
   of the previously best box are forcibly preserved in this set.
3. `sample_best_global()` evaluates a deterministic centered Latin hypercube
   with `2^d` points in every selected box.
4. Boxes whose EI upper bound can still beat the sampled EI become active.
5. The active boxes are sampled again. Boxes that can beat that result become
   targets; the sampled-best box is always included.
6. `split_selected_global()` splits every target into `2*d + 1` children.
7. The loop stops only when no bound can beat the sampled EI and every width of
   the best box is below `epsilon_x`, or when `max_partitions` is reached.

Strict `>` and non-strict `>=` comparisons intentionally match the Python
reference. `MPI_MAXLOC` chooses the lowest rank on an EI tie; because ranks own
contiguous slices, this reproduces NumPy's first-index `argmax` behavior.

## Bounds formulas

For a box and training point, `rbf_k_bounds` finds the nearest and farthest
possible scaled distances and applies the RBF kernel. `mu_bounds` selects the
kernel lower or upper endpoint according to the sign of each `alpha` value.
`sigma_bounds` performs interval forward substitution through `L`, then bounds
`sigma_f_2 - ||v||^2`. `ei_bounds` propagates those mean and sigma intervals
through the minimization EI expression.

These are interval bounds, not point predictions. A loose upper bound slows
pruning but remains correct; an underestimated upper bound can incorrectly
remove the optimum.

## DIRECT-style split layout

For dimension `d`, an active parent produces `stride = 2*d + 1` rows:

```text
lower child for widest normalized dimension
upper child for widest normalized dimension
lower child for next dimension
upper child for next dimension
...
center child
```

Dimensions are ordered by descending `(upper-lower)/domain_width`; dimension
index breaks ties. Every later child inherits the middle third from dimensions
already processed. This ordering matters because the next iteration preserves
the contiguous child block containing the previous best box.

## Memory model

Use `--device-batch-rows N` to cap box rows in the two largest CUDA phases. The
default is 4096. Smaller values use less device memory and launch more kernels;
they do not change numerical results.

Approximate per-batch device work, excluding the constant GP arrays, is:

```text
EI bounds:  8 * N * (4*n_train + 2*d + 1) bytes
EI samples: 8 * N * (  n_train + 3*d + 1) bytes
```

The EI-bound phase currently keeps four interval work matrices (`K_lo`,
`K_hi`, `v_lo`, `v_hi`). They are now bounded by the batch setting instead of
by the total box count. Constant GP arrays are uploaded once per phase and each
batch is freed immediately after its result is copied to the host.

Two scaling limits remain important:

- all ranks still hold the complete host-side box list after each gather and
  broadcast, so host memory is replicated rather than fully distributed;
- the split phase creates `2*d + 1` children per target and currently allocates
  its local output in one piece.

Thus `--device-batch-rows` fixes the former unbounded EI workspaces, but it does
not make the algorithm's exponentially growing box population disappear. For
very deep/high-dimensional searches, the next architectural step is keeping
boxes sharded across ranks and streaming split output rather than gathering it
after every partition.

## Build and validate

Use a fresh build directory if an existing CMake cache was created at another
absolute path:

```bash
cmake -S native -B native/build
cmake --build native/build --target exactbo_partitioning exactbo_split_boxes
```

Force multi-batch execution during a small validation run:

```bash
envs/venvTrial/bin/python native/exactbo/scripts/partitioning_workflow.py \
  --native-build-dir native/build \
  --workdir /tmp/exactbo-check \
  --max-partitions 6 \
  --device-batch-rows 2 \
  --verbose
```

A correct batch refactor produces the same `best_x`, `best_ei_scaled`, box
counts, and convergence status for batch size 2 and 4096. Use the standalone
`bounds_workflow.py` and `split_boxes_workflow.py` when isolating a discrepancy
in interval math or child-box ordering.
