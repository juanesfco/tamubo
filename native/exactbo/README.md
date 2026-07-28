# Native ExactBO

This directory contains the C++/MPI/CUDA ExactBO implementation. Benchmark and learning programs live under `native/benchmarks/`.

Start with [IMPLEMENTATION.md](IMPLEMENTATION.md) for the Python-to-native function map, partition-loop walkthrough, array layouts, memory formulas, and known scaling limits.

## Layout

```text
native/exactbo/
  executables/   CUDA/MPI programs that are built into native executables
  scripts/       Python orchestration scripts
  data/          generated workflow inputs/outputs, ignored except .gitkeep
  kernels/       reserved for reusable device kernels
  src/           reusable host/file box-storage implementation
```

## Bounds Workflow

`bounds_workflow.py` starts from the same simple 2D example data, trains the sklearn GPR, exports the trained parameters and box bounds, launches the native kernels, and reads back final `ei_lo` / `ei_hi` arrays for each box.

Build the native executables:

```bash
cmake -S native -B native/build
cmake --build native/build --target exactbo_rbf_k_bounds exactbo_mu_bounds exactbo_sigma_bounds exactbo_ei_bounds
```

Run the workflow:

```bash
envs/venvTrial/bin/python native/exactbo/scripts/bounds_workflow.py \
  --workdir native/exactbo/data/bounds_workflow \
  --native-build-dir native/build \
  --mpi-ranks 1
```

The workflow writes training data, GP parameters, native binary inputs/outputs, and these final NumPy arrays:

```text
K_lo.npy
K_hi.npy
mu_lo.npy
mu_hi.npy
sig_lo.npy
sig_hi.npy
ei_lo.npy
ei_hi.npy
manifest.json
```

To use a different black-box function and initial design, import `run_workflow(...)` from `bounds_workflow.py` and pass your own `f`, `X0`, domain bounds, and box bounds.

## DIRECT-Style Box Splitting

`exactbo_split_boxes` reads a list of boxes, an active-box mask, and domain widths. It applies the DIRECT-style split rule to each active box on the GPU. MPI splits the input rows across ranks, so multiple visible GPUs can process separate batches. Active split boxes are written first in `2*d + 1` blocks, followed by inactive boxes when `keep_inactive` is enabled.

Build the split executable:

```bash
cmake --build native/build --target exactbo_split_boxes
```

Run the small example workflow:

```bash
envs/venvTrial/bin/python native/exactbo/scripts/split_boxes_workflow.py \
  --workdir native/exactbo/data/split_boxes_workflow \
  --native-build-dir native/build \
  --mpi-ranks 1 \
  --split-batch-parents 0
```

Use `--mpi-ranks 2` or higher when the job has multiple MPI ranks/GPUs available.
The default `--split-batch-parents 0` derives a conservative batch from free
CUDA memory and the number of ranks sharing a device. The standalone executable
writes active children and retained inactive rows directly into their final
planar output offsets with MPI-IO; it no longer gathers a complete child
population on rank zero. Its diagnostic input file is still read completely by
each rank.

The repeated all-active stress workflow is also bounded by default:

```bash
envs/venvTrial/bin/python native/exactbo/scripts/partition_all_boxes_workflow.py \
  --workdir /tmp/exactbo-split-stress \
  --native-build-dir native/build \
  --steps 8 \
  --split-batch-parents 0
```

It keeps one rolling input/output generation, copies planar bounds in 64 MiB
chunks, and retains only the final native output. Add `--debug-save-numpy` only
for small runs where per-step arrays are useful.

## Native ExactBO Partitioning

`exactbo_partitioning` is the first native translation of `exactbo_partitioning(...)` from `src/tamubo/exactbo/run.py`. Python still trains the sklearn GP and exports its fitted state, but the partition loop itself runs in C++/MPI/CUDA: EI upper-bound pruning, deterministic LHS sample scoring, and DIRECT-style box splitting. MPI divides the current box list by contiguous row batches; each rank maps its node-local MPI rank to a visible GPU.

Build it:

```bash
cmake --build native/build --target exactbo_partitioning
```

Run the example workflow:

```bash
envs/venvTrial/bin/python native/exactbo/scripts/partitioning_workflow.py \
  --workdir native/exactbo/data/partitioning_workflow \
  --native-build-dir native/build \
  --mpi-ranks 1 \
  --max-partitions 6 \
  --device-batch-rows 4096
```

Use more MPI ranks when multiple GPUs are visible:

```bash
envs/venvTrial/bin/python native/exactbo/scripts/partitioning_workflow.py \
  --workdir native/exactbo/data/partitioning_workflow \
  --native-build-dir native/build \
  --mpi-ranks 2 \
  --max-partitions 6 \
  --device-batch-rows 4096
```

## Full Native-Acquisition BO Workflow

`exactbo_workflow.py` is the Python orchestrator for the full Bayesian optimization loop. Python receives the problem and data, evaluates the black-box function, fits the sklearn GP, and launches the native `exactbo_partitioning` executable each BO iteration to propose the next point.

Build the native acquisition executable:

```bash
cmake -S native -B native/build
cmake --build native/build --target exactbo_partitioning
```

Run the built-in 2D example:

```bash
envs/venvTrial/bin/python native/exactbo/scripts/exactbo_workflow.py \
  --workdir native/exactbo/data/exactbo_workflow \
  --native-build-dir native/build \
  --mpi-ranks 1 \
  --max-iters 3 \
  --max-partitions 6 \
  --device-batch-rows 4096
```

Run a custom problem by providing an objective and `.npy` data files:

```bash
envs/venvTrial/bin/python native/exactbo/scripts/exactbo_workflow.py \
  --example none \
  --objective path/to/problem.py:objective \
  --x0 path/to/X0.npy \
  --bounds path/to/bounds.npy \
  --workdir native/exactbo/data/custom_bo_workflow \
  --native-build-dir native/build \
  --mpi-ranks 2
```

The objective must accept an array shaped `(n, d)` and return one value per row. `bounds.npy` must have shape `(d, 2)`. If initial objective values are already available, pass `--y0 path/to/y0.npy`; otherwise the script evaluates `objective(X0)`.


## Memory control

The acquisition now streams all box phases. `--device-batch-rows` bounds EI
work, while `--split-batch-parents 0` selects a safe split batch from current
CUDA memory. Use `--box-storage auto` for normal runs: small populations remain
in RAM and large child generations are written directly to a temporary binary
store without a box gather or broadcast.

Force and test either backend with `--box-storage host|file`. The RAM threshold
is controlled by `--host-box-limit-bytes`; zero recomputes it before each split
from currently available host/cgroup memory and reaches one MPI-wide decision.
Forced `host` mode bypasses that guard and is intended for small equivalence
tests. `--spill-dir` must name a real filesystem visible to every MPI rank.
Completed writes are synced and evicted from the page cache before atomic
publication; obsolete files are removed automatically unless
`--keep-spill-files` is supplied.

The population can still grow exponentially, but file-backed peak box memory is
now proportional to a processing batch rather than the complete population.
Local EI values and masks remain proportional to each rank's assigned box count.
See `IMPLEMENTATION.md` for formulas, file lifecycle, and exact equivalence tests.
