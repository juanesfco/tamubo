# Native ExactBO

This directory is reserved for the C++/MPI/CUDA ExactBO implementation.

The existing CUDA, MPI, memory, and matrix multiplication programs live under
`native/benchmarks/` because they are reference benchmarks and development
scaffolding, not production ExactBO code.

## `rbf_k_bounds`

The first native ExactBO step mirrors `tamubo.exactbo.bounds.rbf_k_bounds`.
Python still orchestrates the workflow: it evaluates the black-box function,
fits the sklearn GPR, exports files, launches the CUDA/MPI executable, and reads
`K_lo` / `K_hi` back from disk.

Build the native executable:

```bash
cmake -S native -B native/build
cmake --build native/build --target exactbo_rbf_k_bounds
```

Run the `examples/exactbo/minimization_2d.py`-style example:

```bash
envs/venvTrial/bin/python native/exactbo/scripts/rbf_k_bounds_workflow.py \
  --workdir native/data/exactbo/rbf_k_bounds_workflow \
  --native-exe native/build/exactbo_rbf_k_bounds \
  --mpi-ranks 1
```

The script writes:

```text
X0.npy
y0.npy
domain_bounds.npy
box_bounds_L.npy
box_bounds_U.npy
gpr_parameters.npz
rbf_inputs/train_*.bin
rbf_outputs/train_*.bin
K_lo.npy
K_hi.npy
manifest.json
```

To use a different black-box function and initial design, import
`run_workflow(...)` from `rbf_k_bounds_workflow.py` and pass your own `f`, `X0`,
domain bounds, and box bounds.

