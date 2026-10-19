# ExactBO subsystems, one script each

Minimal, standalone versions of each piece of `tamubo.exactbo`, written with
CuPy (or NumPy when no GPU is visible) and following the same math as the package.
Each script:

1. defines the subsystem in a few short functions,
2. checks it: against points sampled inside the boxes (bounds must contain
   them), against the `tamubo` implementation, and for tightness,
3. times it on many boxes.

## Summary (GPU run, 2026-10-19)

All seven subsystems were run on one NVIDIA GB10 (DGX Spark, CuPy 14.2, JAX 0.8.2).
The log is `logs/spark-gpu-2026-10-19.out`.

- **Every bound is valid.** On every test, every value sampled inside a box lies
  inside that box's bounds. This holds for the distances, the kernel, mu, sigma
  and log EI, at every box width.
- **The subsystems match `tamubo`.** The differences to the package functions are
  0 for partition, kernel, mu and sigma, 1.4e-14 for the AutoBound mu and Q bounds,
  and 1.8e-12 for log EI bounds.
- **The GPU reproduces the earlier CPU results exactly.** Every tightness number
  and every search trajectory is identical to the CPU run.
- **AutoBound makes the bounds converge.** As boxes shrink, the interval bound on
  mu stays about 27 times too wide. With AutoBound the overestimate drops to
  about 1.2, and its error shrinks like width².
- **The full search is certified.** In 2-D and 4-D, ExactBO converges, and a dense
  grid or 10⁶ random points finds no point that beats the incumbent by more than
  eps_ei. AutoBound cuts the number of boxes bounded in 2-D from 1.42 million to
  374, a factor of about 3800.
- **The open problem is large boxes in high dimension.** In 10-d, every box of
  width 3^-4 or smaller is pruned. The boxes that cannot be pruned are the wide
  ones (3^0 and 3^-2), where the log EI bound is still about 7.5 nats too high,
  even with AutoBound.

## The subsystems

| script | subsystem | `tamubo` counterpart |
|---|---|---|
| `s1_partition.py` | DIRECT trisection, widths in length-scale units, `split_ratio`, `min_width` | `partition.split_boxes` |
| `s2_distance_bounds.py` | min/max scaled distance from a box to each training point | inside `bounds.rbf_k_bounds` |
| `s3_kernel_bounds.py` | RBF kernel range from the distance range: exact (monotone) and with AutoBound | `bounds.rbf_k_bounds` |
| `s4_mu_bounds.py` | posterior mean: sign split of alpha, AutoBound Taylor enclosure of mu(x), intersection | `bounds.mu_bounds`, `autobound_bounds.taylor_mu_q_bounds` |
| `s5_sigma_bounds.py` | posterior std through Q = k^T K^-1 k: (A) sign split of L^-1, (B) eigenvalue, (C) Cauchy-Schwarz, (D) AutoBound | `bounds.sigma_bounds`, `q_cs_weights`, `q_cs_lower`, `taylor_mu_q_bounds` |
| `s6_logei_bounds.py` | stable log h / log EI (LogEI), corner bounds of log EI, the full box chain s2 → s6 | `acquisition_functions.ei.log_h`, `bounds.ei_bounds(log=True)` |
| `s7_exact_search.py` | the partition loop: sampling, incumbent, pruning, termination, certified against a dense grid | `run.exactbo_partitioning` |

The chain for one box `[l, u]` is:

```
box ──s2──> scaled distances [r_lo, r_hi] to each training point
    ──s3──> kernel entries [k_lo, k_hi]
    ──s4──> mu    in [mu_lo, mu_hi]     (sign split ∩ AutoBound)
    ──s5──> sigma in [sig_lo, sig_hi]   (A ∩ B ∩ C ∩ D)
    ──s6──> log EI <= log EI(mu_lo, sig_hi)   (EI decreases in mu, increases in sigma)
```

`s7` uses that upper bound to prune: a box is dropped when its bound is at most
incumbent + eps_ei. `s1` splits the boxes that remain.

`common.py` has the backend switch, a toy GP, random DIRECT-like boxes, timing,
and `autobound_range`. The toy GP uses an RBF kernel with length scales from 0.15
to 1.5, fixed hyperparameters and standardized targets. The random boxes have
widths 3^-k. `autobound_range` is a small AutoBound wrapper: a jitted, vmapped
Taylor enclosure of any `f(x)` around each box center, bounded over the box.

## Run

```bash
# from the repo root; on the Spark: source envs/venv_spark/bin/activate
cd development/exactbo_subsystems
python s4_mu_bounds.py                         # CuPy if a GPU is visible, else NumPy
python s4_mu_bounds.py --backend numpy --d 10 --N 128 --n 4096
python s7_exact_search.py --d 2                # --no-autobound for interval bounds only

# everything, as in the summary (about 6.5 minutes on the GB10)
for s in s1_partition s2_distance_bounds s3_kernel_bounds s4_mu_bounds s5_sigma_bounds \
         s6_logei_bounds s7_exact_search; do python $s.py --backend cupy; done
python s6_logei_bounds.py --backend cupy --d 10 --N 128 --n 8192
python s7_exact_search.py --backend cupy --d 4 --N 32 --eps-x 0.01
python s7_exact_search.py --backend cupy --d 2 --no-autobound

# on a Slurm cluster (Vision)
sbatch development/exactbo_subsystems/run_all.sbatch   # from the repo root; output in logs/
```

The options common to all scripts are:

- `--backend {auto,numpy,cupy}`
- `--n`: the number of boxes in the benchmark
- `--N`: the number of training points
- `--d`: the dimension
- `--seed`

When JAX starts, it may print a `CUDA_ERROR_NO_DEVICE` traceback from its CUDA
plugin. This happens on a node without a GPU. JAX then falls back to the CPU, and
the result is unaffected.

## Results

All results come from a GB10 run with d = 5 and N = 64 unless stated otherwise.
Tightness is the median over 200 random boxes of each width, compared with 512
points sampled inside each box.

### s1: partition

- **Matches `split_boxes`.** The result is identical to `split_boxes` for
  `split_ratio` 0, 1 and 1/3.
- **The children tile the parent.** After 3 generations there are 325 boxes. Their
  volumes sum to 1.000000000000, and each of 2000 random points lies in exactly
  one box.
- **Only the fast-varying dimensions are split.** Widths are measured in
  length-scale units, so the unit box splits into 5 children instead of 11. The
  dimensions where the GP is nearly flat are skipped.
- **Speed.** 65,536 parents split into 278,590 children in 4.1 ms (68 M
  children/s).

### s2: distance bounds

Both ends of the range are attained, with 0.0 error. The nearest point is the
training point clipped to the box, and the farthest is the far corner. These
bounds are therefore exact.

Speed: 16,384 boxes × 64 points in 4.7 ms.

### s3: kernel bounds

k(r) is monotone in r, so the exact range is `[k(r_hi), k(r_lo)]`. That range is
0.0 away from `rbf_k_bounds`.

| AutoBound degree | contains the exact range | median excess width |
|---|---|---:|
| 0 | yes | 0 |
| 1 | yes | 0.115 |
| 2 | yes | 0.0084 |
| 3 | yes | 0.074 |

For a single kernel entry, AutoBound cannot beat the exact range. It pays off one
level up, on mu(x) and Q(x) as whole functions of x, where the per-entry intervals
ignore that every entry depends on the same x.

### s4: posterior mean

The table shows the bound width divided by the sampled range of mu. A value of 1
means the bound is tight. All bounds are valid.

| box width | interval | AutoBound | both |
|---|---:|---:|---:|
| 1 | 49 | 1e5 | 49 |
| 3^-2 | 30 | 10 | 10 |
| 3^-4 | 26 | 1.32 | 1.32 |
| 3^-6 | 26 | 1.2 | 1.2 |
| 3^-8 | 27 | 1.2 | 1.2 |

- **The interval bound never converges.** It stays about 27 times too wide however
  small the box. With mixed-sign alpha it overestimates mu by about
  Σ|alpha_i|(k_hi − k_lo).
- **AutoBound converges.** Its error shrinks like width². The remaining 1.2 is
  mostly sampling, because 512 points miss the true extremes.
- **The intersection is needed.** On the full box AutoBound is useless (1e5), so
  the method takes the intersection of both bounds.

### s5: posterior standard deviation

The table shows sigma_hi minus the sampled maximum of sigma, with sigma_f = 1.
Each column adds one more bound on Q. All bounds are valid.

| box width | A | A+B | A+B+C | A+B+C+D |
|---|---:|---:|---:|---:|
| 1 | 0.11 | 0.11 | 0.11 | 0.11 |
| 3^-2 | 0.50 | 0.48 | 0.42 | 0.29 |
| 3^-4 | 0.24 | 0.24 | 0.080 | 0.0061 |
| 3^-6 | 0.057 | 0.057 | 0.0098 | 0.00037 |
| 3^-8 | 0.0078 | 0.0078 | 0.0011 | 0.000023 |

- **What each part adds.** The Cauchy-Schwarz bound (C) gains 3 to 7 times over
  A. AutoBound (D) gains another 13 to 50 times.
- **Match with `tamubo`.** A+B+C is identical to `sigma_bounds` with the
  Cauchy-Schwarz term (0 difference).

### s6: log EI

- **Stable log h.** The stable `log_h` matches 50-digit arithmetic to within
  1.8e-15 relative error on z in [-1e4, 10]. The naive `log(phi + z Phi)` has
  1e-2 relative error and is -inf below z = -39.5.
- **Corner bounds.** The corner bounds contain every sampled log EI on random
  (mu, sigma) rectangles, and they match `ei_bounds(log=True)` to 1.8e-12.

The full box chain is also tested. The table shows the upper bound minus the
sampled maximum of log EI, in nats; 0 would be tight. All bounds are valid.

| box width | d=5 interval | d=5 +AutoBound | d=10 interval | d=10 +AutoBound |
|---|---:|---:|---:|---:|
| 1 | 5.6 | 5.6 | 7.8 | 7.8 |
| 3^-2 | 15.4 | 14.0 | 8.0 | 7.6 |
| 3^-4 | 20.7 | 1.66 | 122 | 69 |
| 3^-6 | 6.2 | 0.145 | 1810 | 1010 |
| 3^-8 | 0.84 | 0.016 | 139 | 50 |

The large 10-d gaps for small boxes do not matter for pruning. A follow-up check
in d = 10 with N = 128 found the following:

- **The small boxes are all pruned.** Random small boxes lie where log EI is
  around −230 at width 3^-4 and −7300 at width 3^-6, compared with a maximum of
  about −2.4. Their upper bounds are just as negative, so 100% of these boxes are
  pruned, and the gap is about 14% of the value.
- **The wide boxes are not.** All 200 boxes of width 1, and 152 of the 200 boxes
  of width 3^-2, cannot be pruned, with a gap of about 7.5 nats. In d = 5, 94% of
  the 3^-2 boxes cannot be pruned. **Large boxes are where tightening is still
  needed.**

### s7: the full search

The search uses eps_ei = 0.1, so log EI is certified to within about 10.5% of EI.

| case | partitions | boxes bounded (total / peak) | time | certificate |
|---|---:|---:|---:|---|
| 2-D, N=16, eps_X=1e-3, AutoBound | 11 | 374 / 125 | 3.2 s | max on 1001² grid −1.2156 ≤ incumbent −1.2257 + 0.1 ✓ |
| 2-D, same, interval only | 19 | 1,417,314 / 290,984 | 0.54 s | −1.2156 ≤ −1.2156 + 0.1 ✓ |
| 4-D, N=32, eps_X=1e-2, AutoBound | 13 | 17,929 / 5,742 | 16.8 s | max on 10⁶ random points −0.7770 ≤ −0.7749 + 0.1 ✓ |

- **AutoBound bounds about 3800 times fewer boxes in 2-D.** Its wall time is
  still longer here because of JAX compilation. The AutoBound program compiles
  once per padded batch size, and 374 boxes at about 11 µs each is only about
  4 ms of compute.
- **What matters at scale is box count.** Box count and memory are what limit
  10-d. An uncapped 10-d run of the package fills the GB10's memory at about
  150 M boxes.

### Throughput on the GB10

The figures below are for d = 5 and N = 64, using the best of several runs after
warm-up.

| step | boxes | time | rate |
|---|---:|---:|---:|
| partition (`split_ratio` 1/3) | 65,536 parents | 4.1 ms | 68 M children/s |
| distance bounds | 16,384 × 64 | 4.7 ms | 0.23 G pairs/s |
| kernel bounds, exact | 16,384 × 64 | 0.40 ms | 2.6 G entries/s |
| mu, interval (given k bounds) | 8,192 | 0.12 ms | 70 M boxes/s |
| mu, AutoBound degree 2 | 8,192 | 15.9 ms | 0.52 M boxes/s |
| Q, (A) sign split | 8,192 | 1.1 ms | 7.5 M boxes/s |
| Q, (C) Cauchy-Schwarz weights + bound | 8,192 | 0.95 ms | 8.6 M boxes/s |
| Q and s, (D) AutoBound | 8,192 | 64 ms | 0.13 M boxes/s |
| log EI chain, interval (A+B+C) | 8,192 | 7.3 ms | 1.1 M boxes/s |
| log EI chain, + AutoBound | 8,192 | 88 ms | 0.094 M boxes/s (11 µs/box) |
| log EI chain, interval, d=10, N=128 | 8,192 | 31 ms | 0.27 M boxes/s |
| log EI chain, + AutoBound, d=10, N=128 | 8,192 | 816 ms | 0.010 M boxes/s (100 µs/box) |

AutoBound costs about 12 times more per box than the interval chain in 5-d, and
about 26 times more in 10-d. Most of that cost is the Q enclosure (D). It is still
worth it when it removes more boxes than that factor, as it does by a factor of
3800 in the 2-D search.

## Not covered here

These parts of `run.py` are about scale rather than the math, and are left out:

- **Compact storage.** `run.py` stores a partition as its parents and rebuilds the
  children in chunks, which takes 8 bytes per child instead of 16d.
- **Multiple GPUs.** `run.py` shards the parents over several GPUs, one thread per
  GPU, rebalanced every partition (`n_gpus`). `development/cupy_benchmark`
  measures plain multi-GPU CuPy scaling.
- **Sampling options.** These scripts leave out the analyzed/active two-stage
  sampling (s7 samples every box center), `box_sampling="lhs"` and the heuristic
  `max_target_boxes` cap.
- **Padding N.** `run.py` pads N to a multiple of 16 so that the AutoBound
  programs survive BO iterations that add points. s7 pads only the number of
  boxes.
