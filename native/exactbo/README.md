# Native ExactBO

This directory contains the C++/MPI/CUDA ExactBO implementation. Benchmark and learning programs live under `native/benchmarks/`.

## Layout

```text
native/exactbo/
  executables/   CUDA/MPI programs that are built into native executables
  scripts/       Python orchestration scripts
  data/          generated workflow inputs/outputs, ignored except .gitkeep
  kernels/       reserved for reusable device kernels
  src/           reserved for reusable native ExactBO code
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
