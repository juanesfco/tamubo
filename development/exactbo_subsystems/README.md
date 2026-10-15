# ExactBO subsystems, one script each

Minimal, standalone versions of each piece of `tamubo.exactbo`, written with
CuPy (or NumPy when no GPU is visible) and following the same math as the package.
Each script:

1. defines the subsystem in a few short functions,
2. checks it: against points sampled inside the boxes (bounds must contain
   them), against the `tamubo` implementation, and for tightness,
3. times it on many boxes.

| script | subsystem | `tamubo` counterpart |
|---|---|---|
| `s1_partition.py` | DIRECT trisection, widths in length-scale units, `split_ratio`, `min_width` | `partition.split_boxes` |
| `s2_distance_bounds.py` | min/max scaled distance from a box to each training point | inside `bounds.rbf_k_bounds` |
| `s3_kernel_bounds.py` | RBF kernel range from the distance range: exact (monotone) and with AutoBound | `bounds.rbf_k_bounds` |
| `s4_mu_bounds.py` | posterior mean: sign split of alpha, AutoBound Taylor enclosure of mu(x), intersection | `bounds.mu_bounds`, `autobound_bounds.taylor_mu_q_bounds` |
| `s5_sigma_bounds.py` | posterior std through Q = k^T K^-1 k: (A) sign split of L^-1, (B) eigenvalue, (C) Cauchy-Schwarz, (D) AutoBound | `bounds.sigma_bounds`, `q_cs_weights`, `q_cs_lower`, `taylor_mu_q_bounds` |
| `s6_logei_bounds.py` | stable log h / log EI (LogEI), corner bounds of log EI, the full box chain s2 → s6 | `acquisition_functions.ei.log_h`, `bounds.ei_bounds(log=True)` |
| `s7_exact_search.py` | the partition loop: sampling, incumbent, pruning, termination, certified against a dense grid | `run.exactbo_partitioning` |

`common.py` has the backend switch, a toy GP (RBF, length scales 0.15 to 1.5,
fixed hyperparameters, standardized targets), random DIRECT-like boxes (widths
3^-k), timing, and `autobound_range`, a small AutoBound wrapper: a jitted,
vmapped Taylor enclosure of any `f(x)` around each box center, bounded over the
box.

## Run

```bash
# from the repo root, environment envs/tamubo_exactbo (see envs/README.md)
cd development/exactbo_subsystems
python s4_mu_bounds.py                         # CuPy if a GPU is visible, else NumPy
python s4_mu_bounds.py --backend numpy --d 10 --N 128 --n 4096
python s7_exact_search.py --d 2                # --no-autobound for interval bounds only

# all of them on one GPU
sbatch development/exactbo_subsystems/run_all.sbatch   # from the repo root; output in logs/
```

Common options: `--backend {auto,numpy,cupy}`, `--n` boxes in the benchmark,
`--N` training points, `--d` dimension, `--seed`. On a node without a GPU, JAX
prints a `CUDA_ERROR_NO_DEVICE` traceback from its CUDA plugin when it starts; it
then falls back to the CPU and the result is unaffected.

## What the checks show (CPU, login node, d = 5, N = 64)

- **Partition**: identical to `split_boxes` for `split_ratio` 0, 1 and 1/3. The
  children tile the parent (volumes sum to 1, every point in exactly one box). In
  length-scale units the unit box splits into 5 children instead of 11: only the
  dimensions where the GP varies fast are split.
- **Distance**: both ends are attained (nearest point = the training point
  clipped to the box, farthest = the far corner), so the bounds are exact.
- **Kernel**: k(r) is monotone in r, so the exact range is
  `[k(r_hi), k(r_lo)]`. AutoBound degree 0 reproduces it; degrees 1 to 3 are
  valid but wider. For a single entry AutoBound cannot do better. It pays off on
  mu(x) and Q(x) as whole functions of x (next two rows), where per-entry
  intervals ignore that every entry depends on the same x.
- **Mean** (bound width / sampled range, median; 1 is tight):

  | box width | interval | AutoBound | both |
  |---|---:|---:|---:|
  | 1 | 49 | 1e5 | 49 |
  | 3^-2 | 30 | 10 | 10 |
  | 3^-4 | 26 | 1.3 | 1.3 |
  | 3^-8 | 27 | 1.2 | 1.2 |

  The interval bound stays ~27x too wide however small the box. AutoBound's
  error shrinks like width^2; the remaining 1.2 is mostly sampling (512 points
  miss the true extremes).
- **Std** (sigma_hi minus sampled max, median, sigma_f = 1):

  | box width | A | A+B | A+B+C | A+B+C+D |
  |---|---:|---:|---:|---:|
  | 3^-4 | 0.24 | 0.24 | 0.080 | 0.0061 |
  | 3^-8 | 0.0078 | 0.0078 | 0.0011 | 0.000023 |

  Identical to `sigma_bounds` with the Cauchy-Schwarz term (0 difference).
- **log EI**: stable `log_h` matches 50-digit arithmetic to 1.8e-15 relative
  error on z in [-1e4, 10]; the naive `log(phi + z Phi)` is -inf below z = -39.5.
  Upper bound minus sampled max log EI (nats): 0.84 → 0.016 at width 3^-8 with
  AutoBound, 20.7 → 1.7 at 3^-4.
- **Search** (2-D, eps_X = 1e-3, eps_ei = 0.1): converges in 11 partitions; the
  max log EI on a 1001^2 grid is -1.2156, within eps_ei of the incumbent
  -1.2257, as certified. It bounds 374 boxes in total. With interval bounds
  only (`--no-autobound`) it also converges, but needs 19 partitions and 1.4
  million boxes (up to 291k in one partition). 4-D (N = 32, eps_X = 0.01): 13
  partitions, at most 5742 boxes, certified against 10^6 random points.

The CuPy path has not been run yet: the numbers above are from a login node
without a GPU. `run_all.sbatch` runs every script on one GPU, plus a 10-d log EI
chain and the 4-D search.

## Not covered here

These parts of `run.py` are about scale rather than the math, and are left out:

- Storing a partition as its parents and rebuilding the children in chunks
  (8 bytes per child instead of 16d).
- Sharding the parents over several GPUs, one thread per GPU, rebalanced every
  partition (`n_gpus`). `development/cupy_benchmark` measures plain multi-GPU
  CuPy scaling.
- The analyzed/active two-stage sampling (s7 samples every box center),
  `box_sampling="lhs"`, and the heuristic `max_target_boxes` cap.
- Padding N to a multiple of 16 so the AutoBound programs survive BO iterations
  that add points (s7 pads only the number of boxes).
