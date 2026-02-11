# `tamubo.exactbo`

ExactBO package with a shared public API and two execution paths:

- CPU/sequential path (NumPy + object-based boxes)
- GPU/vectorized path (cuPyNumeric + array-based boxes)

## Core public modules

- `__init__.py`: package exports.
- `run.py`: unified entry point `run_exactbo(...)` that selects backend (`auto`, `numpy`, `cupynumeric`) and returns a normalized result object.
- `backend.py`: backend detection/resolution utilities (`has_cupynumeric`, `resolve_backend`, `BackendInfo`).

## CPU/sequential implementation

- `loop_exactbo.py`: outer BO loop (`ExactBOLoop`) that fits GP, runs partition search, evaluates oracle, and appends data.
- `loop_partition.py`: branch-and-bound partition search (`PartitionMaxEISearch`) using `Box`/`Boxes` objects.
- `partition.py`: box data structures and split/masking utilities (`Box`, `Boxes`, `split_box`, `hypermask`).
- `bounds.py`: interval bounds for kernel, mean, sigma, and EI over a box.
- `ei.py`: pointwise expected improvement from model predictions.
- `interval_arithmetics.py`: interval arithmetic primitives used by bound propagation.
- `plots.py`: plotting utilities for iterations and partition states.

## GPU/vectorized implementation

- `vectorized_loop.py`: vectorized ExactBO loops for cuPyNumeric (`exactbo_loop_cupynumeric`, `partition_loop_cupynumeric`).
- `vectorized_partition.py`: vectorized DIRECT-style box splitting.
- `vectorized_bounds.py`: vectorized kernel/mean/sigma/EI bound propagation over batches of boxes.
- `vectorized_ei.py`: vectorized EI/CDF/PDF helpers for cuPyNumeric arrays.

## Notes

- Preferred external usage is through `run_exactbo(...)` instead of importing backend-specific internals.
