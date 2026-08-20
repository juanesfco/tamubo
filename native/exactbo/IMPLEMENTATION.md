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

The repository keeps only this integrated native executable. Bounds, sampling,
and splitting are implemented directly in `partitioning.cu`; the Python BO loop
does not launch separate stage executables.

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

1. `compute_ei_hi_streamed()` reads each rank's boxes in bounded windows and
   computes a conservative EI upper bound per box.
2. Boxes within `epsilon_ei` of the global maximum are analyzed. Children of
   the previously best box are forcibly preserved in this set.
3. `sample_best_streamed()` evaluates a deterministic centered Latin
   hypercube with `2^d` points in every selected box. Batch results are reduced
   immediately instead of retaining every sampled point.
4. Boxes whose EI upper bound can still beat the sampled EI become active.
5. The active boxes are sampled again. Boxes that can beat that result become
   targets; the sampled-best box is always included.
6. `split_selected_streamed()` compacts targets into bounded parent batches
   and writes each `2*d + 1` child block directly to the next box store.
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

## Memory and box-storage model

`BoxStore` gives the partition loop one interface for two backends:

- `HostBoxStore` keeps planar lower/upper arrays in RAM;
- `FileBoxStore` reads bounded windows from a versioned temporary binary
  file using `pread`, while split batches write disjoint ranges using `pwrite`.

Use `--box-storage auto|host|file`. In `auto` mode, small populations
remain in RAM. Before every split the program knows the exact child count and
checks the peak coexistence of the current and next stores against
`--host-box-limit-bytes`. Zero derives a conservative limit from
current `MemAvailable` and cgroup-v2 or conventional cgroup-v1 state. It first
divides each node's allowance among its local ranks, then uses one global
minimum so heterogeneous nodes make the same MPI storage decision. Once a run
spills, later generations remain file-backed so a large file is never silently
loaded back into RAM. `--box-storage host` is an intentional diagnostic force;
it bypasses the automatic RAM guard and can OOM on a large run.

The default spill directory is next to the native output. Override it with
`--spill-dir PATH`. It must be a real filesystem visible at the same path
to every MPI rank; a RAM-backed `tmpfs` does not solve host OOM. Rank zero
preallocates a `.partial` file so ENOSPC is detected before CUDA work, ranks
write non-overlapping child ranges computed with `MPI_Exscan`, and rank zero
publishes it without replacing an existing file only after every rank has
synced its writes and asked the kernel to evict clean cached pages. Old
generations are removed after the collective store swap. Each launch uses an
atomically unique `run_XXXXXX` directory. `--keep-spill-files` preserves it for
debugging; a killed or MPI-aborted job may also leave it behind, and that
specific directory is safe to remove after the job exits.

`--device-batch-rows N` bounds EI-bound and sampled-EI CUDA work. Approximate
per-box device work, excluding constant GP arrays, is:

```text
EI bounds:  8 * N * (4*n_train + 2*d + 1) bytes
EI samples: 8 * N * (  n_train + 3*d + 1) bytes
```

This batch size is an explicit cap, not an automatic planner. The constant GP
arrays, including the `n_train^2` Cholesky factor, must also fit on the device.

`--split-batch-parents N` bounds splitting. Zero chooses automatically from
`cudaMemGetInfo`, reserves at least 10%/1 GiB, divides the budget between MPI
ranks sharing a GPU, and counts host staging twice on integrated or
host-page-table/coherent GPUs such as GB10. A compact
selected-parent batch uses:

```text
split device bytes = 32 * d * (d + 1) * N parents
split output rows  = (2*d + 1) * N parents
```

Each rank reads a balanced contiguous range from the global store. EI maxima,
box counts, convergence flags, and sampled winners use scalar MPI reductions.
File-backed populations are never gathered or broadcast; the small host backend
still replicates its completed population after a bounded split. Masks remain
local in both modes. Global parent order and every contiguous child block are
unchanged.

File-backed mode still retains one local EI-upper-bound value and a few local
byte masks per assigned box. Those are much smaller than the two `d`-dimensional
box arrays but remain the next host-scaling target. GPU-resident box caching is
deliberately not enabled yet: on the GB10 it consumes the same unified physical
memory and would need both old and new generations resident during a split.
Streaming is the safe fallback and CUDA batches are the reusable fast path.

## Build and validate

Use a fresh build directory if an existing CMake cache was created at another
absolute path:

```bash
cmake -S native/exactbo -B native/exactbo/executables/build
cmake --build native/exactbo/executables/build --target exactbo_partitioning
```

Force the two opposite memory paths:

```bash
envs/venvTrial/bin/python native/exactbo/exactbo_workflow.py \
  --example minimization_2d --max-iters 1 \
  --native-build-dir native/exactbo/executables --workdir /tmp/exactbo-host \
  --max-partitions 8 --device-batch-rows 4096 \
  --split-batch-parents 4096 --box-storage host

envs/venvTrial/bin/python native/exactbo/exactbo_workflow.py \
  --example minimization_2d --max-iters 1 \
  --native-build-dir native/exactbo/executables --workdir /tmp/exactbo-file \
  --max-partitions 8 --device-batch-rows 2 \
  --split-batch-parents 1 --box-storage file \
  --host-box-limit-bytes 1 --spill-dir /tmp/exactbo-spill

cmp /tmp/exactbo-host/iteration_000/partitioning_output.bin \
    /tmp/exactbo-file/iteration_000/partitioning_output.bin
```

The comparison must be byte-identical. Repeat the file-backed command with
`--mpi-ranks 2` and compare again to exercise rank-boundary target runs and
`MPI_Exscan` child offsets. On multiple nodes, choose a shared spill directory.

When isolating a discrepancy, run one BO iteration with `--verbose` and compare
the complete partition trace and binary output across storage and batch modes.

## Author's notes/tools

Bounding memory formula until now:
$$ M = 8 * BOXES * (2d + 2) + M_{input} $$

Bounding shared memory formula:
$$ SM = 8 * TPB * (2n_{train} + 1) $$

Check time and memory:

```
/usr/bin/time -v -o memory_usage.txt ./executables/exactbo_bounding
```

Check registers and spills:

```
/usr/local/cuda/bin/nvcc -std=c++17 -arch=native -Xptxas=-v -c src/bounding.cu -o /tmp/bounding.o
```

Profile Nsight Systems:

Profile Nsight Compute:
