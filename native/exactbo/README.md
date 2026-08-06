# Native ExactBO

This directory contains the C++/MPI/CUDA ExactBO implementation. Benchmark and learning programs live under `native/benchmarks/`.

Start with [IMPLEMENTATION.md](IMPLEMENTATION.md) for the Python-to-native function map, partition-loop walkthrough, array layouts, memory formulas, and known scaling limits.

## Layout

```text
native/exactbo/
  exactbo_workflow.py  Python BO orchestration
  executables/   the CUDA/MPI ExactBO partitioning executable
  data/          generated workflow inputs/outputs, ignored except .gitkeep
  include/       reusable native declarations
  src/           reusable host/file box-storage implementation
```

## Native-acquisition BO workflow

`exactbo_workflow.py` is the Python orchestrator for the full Bayesian optimization loop. Python receives the problem and data, evaluates the black-box function, fits the sklearn GP, and launches the native `exactbo_partitioning` executable each BO iteration to propose the next point.

Build the native acquisition executable:

```bash
cmake -S native/exactbo -B native/exactbo/executables/build
cmake --build native/exactbo/executables/build --target exactbo_partitioning
```

Build and run the two small step-by-step diagnostics:

```bash
cmake --build native/exactbo/executables/build \
  --target exactbo_bounding exactbo_box_partition
native/exactbo/executables/exactbo_bounding
native/exactbo/executables/exactbo_box_partition
```

`exactbo_bounding` evaluates nested boxes with a shared center. It prints the
actual EI at the center, the EI upper bound over each complete box, and their
gap, then checks that the bound tightens as the boxes shrink.
`exactbo_box_partition` prints every child generated when one 2D box is split
and checks containment, overlap, and total volume.

Run a built-in example (`minimization_2d`, `problem5d`, or `problem10d`):

```bash
envs/venvTrial/bin/python native/exactbo/exactbo_workflow.py \
  --example minimization_2d \
  --workdir native/exactbo/data/exactbo_workflow \
  --native-build-dir native/exactbo/executables \
  --mpi-ranks 1 \
  --max-iters 3 \
  --max-partitions 6 \
  --device-batch-rows 4096 \
  --log-file native/exactbo/data/exactbo_workflow/stress.log
```

Run a custom problem by providing an objective and `.npy` data files:

```bash
envs/venvTrial/bin/python native/exactbo/exactbo_workflow.py \
  --example none \
  --objective path/to/problem.py:objective \
  --x0 path/to/X0.npy \
  --bounds path/to/bounds.npy \
  --workdir native/exactbo/data/custom_bo_workflow \
  --native-build-dir native/exactbo/executables \
  --mpi-ranks 2
```

The objective must accept an array shaped `(n, d)` and return one value per row. `bounds.npy` must have shape `(d, 2)`. If initial objective values are already available, pass `--y0 path/to/y0.npy`; otherwise the script evaluates `objective(X0)`.

Every CLI run tees Python output plus native stdout/stderr to both the terminal and a log file. By default the file is `<workdir>/exactbo_run.log`; use `--log-file PATH` to choose another location. The log begins with the exact command, all parsed arguments including defaults, the start time, and working directory, and ends with the run status and finish time. The selected path is also recorded in `manifest.json`.


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
