# Experiment 5: Length-Scale-Aware Box Splitting

Even with the tighter bounds of experiment 3, the uncapped (exact) 10-d search
of experiment 2 never converged: every target box was trisected along all d
dimensions (2d + 1 = 21 children), the frontier grew ~2-21x per partition, and
the run ran out of GPU memory at partition 11 (experiment 4, step 3).

The fitted GP is nearly flat in several dimensions. problem10d (`X0=32`, seed 0)
has length scales `[9.3, 100, 3.0, 4.1, 1.0, 1.1, 3.1, 0.81, 14.3, 100]` on the
unit cube; problem5d (`X0=32`), which also never converged, has
`[34.6, 17.6, 20.2, 0.25, 0.30]`. The bounds depend on box width over length
scale, so splitting a flat dimension multiplies the boxes without tightening any
bound.

`split_boxes(..., split_ratio=r)` and `exactbo(..., split_ratio=r)` therefore
measure widths in GP length-scale units and trisect a box only along the
dimensions whose scaled width is at least `r` times its widest (`r = 1`:
DIRECT's longest-side rule), skipping dimensions already narrower than
`epsilon_X` (so no dimension is refined below the resolution termination
needs). A box split along m dimensions has 2m + 1 children. `r = 0` keeps the
old rule (every dimension). Any partition of the box keeps the bound-and-prune
search exact; only the number of boxes changes. Following these results,
`exactbo` defaults to `split_ratio=1/3`; experiment-3/4 scripts and the
development benchmark pin `0` so their recorded results stay reproducible.

## Files

- `test_partition.py`: every split tiles its parent exactly (volume, containment,
  every point in exactly one child) for `r` in {0, 1/3, 1}; dimension selection
  with ties and `min_width`; NumPy and CuPy agree; chunked GPU split equals a
  single chunk; child counts match the split; ExactBO's partition loop gives
  the same partitions with any frontier chunk size. NumPy + CuPy.
- `configs/`: experiment-2 configs for the exact 20-iteration BO loop.
- `vision_split_5d.sbatch`: GPU tests, an `r = 0` regression run, and problem5d.
- `vision_split_10d.sbatch`: the uncapped problem10d search with `r` = 1, 1/3.
- `vision_frontier.sbatch`: tests and a rerun of the runs above after the switch
  to bounds-only frontier storage (`results/frontier_*`).
- `vision_shards.sbatch`: the same after sharding the frontier over the GPUs,
  on 1 and 8 GPUs (`results/shard_*`).
- `max_ei_probe.py`: rebuilds one BO iteration's GP from an experiment-2 log and
  compares ExactBO's incumbent with the true max EI (multi-start L-BFGS-B).
- `bound_tightness.py`: on a random sample of the frontier of one partition of a
  logged iteration, compares EI bounds with the true max over each box
  (`results/bound_tightness.jsonl`).
- `partition_from_log.py`: reruns the uncapped search of one logged iteration
  (`results/partition_from_log.jsonl`).
- `results/split_partition_counts.csv`, `results/split_runs.jsonl`: per-partition
  counts and per-run summaries (chosen point, incumbent, memory), written by
  `experiments/exactbo/experiment4/stress_partitioning.py --split-ratio`.

## Results (one H200, uncapped, `epsilon_X = 1e-5`, `epsilon_ei = 0.1`)

Regression: with `r = 0` the 10-d AutoBound + EI run reproduces experiment 4's
target counts exactly (1, 21, 360, ..., 7,213,648 at partition 8; incumbent
0.476884).

problem10d (`X0=32`, seed 0):

| setting | split rule | partitions | peak targets | time | peak GPU memory | incumbent |
|---|---|---:|---:|---:|---:|---:|
| AutoBound + EI, LHS | every dimension (`r = 0`) | out of memory at 11 | 31.8M (still growing) | 958 s | 141 GB | 0.477 |
| AutoBound + EI, LHS | `r = 1` | 99, converged | 32,851 | 41 s | 0.5 GB | 0.5443 |
| AutoBound + EI, LHS | `r = 1/3` | 69, converged | 14,140 | 6.7 s | 1.4 GB | 0.5367 |
| AutoBound + logEI, center (defaults) | `r = 1` | 99, converged | 69,846 | 12.4 s | 1.3 GB | EI 0.5448 |

problem5d (`X0=32`, seed 0, AutoBound + logEI):

| sampling | split rule | partitions | peak targets | time | incumbent (log EI) |
|---|---|---:|---:|---:|---:|
| LHS | every dimension | out of memory at split 10 (178M targets) | | | |
| LHS | `r = 1` | 55, converged | 95,268 | 16.2 s | -1.3986 |
| center | `r = 1` | 55, converged | 123,052 | 4.1 s | -1.3986 |
| LHS | `r = 1/3` | 41, converged | 62,180 | 4.0 s | -1.3988 |
| center | `r = 1/3` | 28, converged | 77,127 | 3.8 s | -1.4074 |

"Converged" means the partitioning loop ended on ExactBO's termination test:
every remaining box was pruned against the incumbent (targets reach 0 by
partition ~20 in 10-d), and the best box was then refined alone until it is
narrower than `epsilon_X` in every dimension. Runs that converge pick
incumbents within `epsilon_ei` of each other, as two exact searches with that
tolerance should.

Takeaways:

- Splitting in length-scale units makes the uncapped 10-d search converge, in
  seconds and with < 1.5 GB, where splitting every dimension ran out of 141 GB.
  The frontier peaks at 14-70k boxes instead of growing past 30M.
- `r = 1/3` was the fastest in every comparison (fewer partitions; more
  children per split, but each split shrinks every comparably wide dimension).

## Full BO loop, exact (experiment 2, `configs/`)

Experiment 2's problem10d config with the new defaults (`split_ratio=1/3`,
center sampling, AutoBound), no `max_target_boxes`, `max_partitions=1000`,
`predict_batch_size = bounds_batch_size = 2e7`, 20 iterations, one H200
(`experiments/exactbo/experiment2/submit_vision.sbatch configs/...`).

The first attempt stored every box's bounds. EI iteration 14 (34.3M targets,
325M boxes) and logEI iteration 4 (28.4M targets) ran out of GPU memory. The
partition loop now stores only each partition's parents (the previous
targets) and one EI bound per child, and rebuilds children in chunks of 2^24
from their parents when they are bounded, sampled or become targets
(`run.py`, `_frontier_boxes`). This gives 8 bytes per box instead of 160 at d = 10.
The reruns (jobs 658921 EI, 658922 logEI) reproduce every earlier iteration's
partitions and targets exactly:

| iteration | EI: partitions | EI: peak targets | EI: boxes created | logEI: partitions | logEI: peak targets | logEI: boxes created |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 37 | 93,653      | 7.9M   | 37 | 284,931     | 27.0M  |
| 2  | 36 | 126,485     | 13.1M  | 28 | 533,872     | 60.3M  |
| 3  | 49 | 481,971     | 73.5M  | 47 | 565,363     | 49.9M  |
| 4  | 39 | 6,778,925   | 717M   | 46 | 136,086,054 | 16.8B  |
| 5  | 34 | 318,246     | 33.2M  | 41 | 63,402,692  | 5.8B   |
| 6  | 23 | 3,856       | 0.17M  | 52 | 24,841,858  | 3.6B   |
| 7  | 34 | 24,376      | 1.3M   | time limit (8 h) after 16 | 106.2M (still growing) | 5.3B so far |
| 8  | 33 | 10,684      | 0.54M  | | | |
| 9  | 23 | 20,831      | 1.1M   | | | |
| 10 | 40 | 74,262      | 6.8M   | | | |
| 11 | 47 | 82,720      | 8.4M   | | | |
| 12 | 42 | 461,343     | 35.8M  | | | |
| 13 | 41 | 6,533,370   | 678M   | | | |
| 14 | 34 | 96,352,341  | 9.74B  | | | |
| 15 | out of memory after 14 | 375.7M (growing 1.75x/partition) | 7.1B so far | | | |

Every completed iteration converged (exact up to `epsilon_X`, `epsilon_ei`).
EI iteration 14 and logEI iteration 4 now complete: the frontier grows to
96M / 136M targets (billions of boxes in total), then pruning against the
incumbent shrinks it to a single box in ~15 partitions. They are slow on one
GPU: EI iteration 14 took ~2.5-3 h and logEI iteration 4 ~3.9 h, against seconds
to minutes for the others. Most iterations need fewer than 500k targets.
logEI keeps spiking after iteration 4: iterations 5 and 6 (63M and 25M targets)
also converge, and the job hit its 8 h limit in iteration 7 with 106M targets
and still growing.

EI iteration 15 is larger still. Its frontier grew past 375M targets (60 GB of
parent bounds) and 3.5B children, and it ran out of memory while counting those
children: `split_child_counts` built (n, d) temporaries for all parents at once,
and now works in chunks. The next limit is real, though. The following partition
needs ~60 GB of parents, 28 GB of EI bounds and the next targets' bounds at
once, and each partition takes over an hour on one H200. logEI's `epsilon_ei`
is relative (~10% of the incumbent EI), so as BO finds better points it demands
tighter bounds than EI's absolute tolerance, which is the likely reason its
spike comes ten iterations earlier.

Earlier fixes on the way: 1e8-box batches ran out of memory in `sigma_bounds`
(batch sizes are now 2e7), and the GPU split allocated slot buffers for all
2d + 1 possible children of every target (it now counts children first and
builds them in 4 GiB chunks).

Regression for the frontier storage (`vision_frontier.sbatch`,
`results/frontier_runs.jsonl`): all eight runs of `results/split_runs.jsonl`
give the same targets per partition, incumbent and chosen point. The `r = 0`
10-d run (7.2M targets) peaks at 18.1 GB of CuPy pool instead of 41.7 GB, at
the same speed. `test_partition.py` also checks that the partition loop's
results do not depend on the chunk size: 60 tests pass, NumPy + CuPy, with
`../experiment3/test_bounds.py`.

## Frontier sharded over the GPUs

One H200 runs out of both memory and time on the spike iterations. Each
partition's parents are now split into contiguous ranges with about equal numbers
of children, one per GPU (`run.py`, `_FrontierShard`, `_balanced_shards`). Each
GPU rebuilds, bounds and samples its own children and keeps their EI bounds and
masks. Only scalars (best bound, incumbent), the few analyzed boxes and the
rebalanced parents cross GPUs. Partitions too small to be worth it stay on fewer
GPUs.

Checks (`vision_shards.sbatch`, `results/shard_runs.jsonl`): on 1 and on 8 GPUs,
the `r = 0` regression, problem10d with `r = 1/3` and problem5d with `r = 1/3`
reproduce `results/split_runs.jsonl` exactly (targets per partition, incumbent,
point). `test_partition.py` also runs a search with sharding forced from the
first partition and compares it with one GPU.

Scaling, problem10d `r = 0`, logEI with center sampling and 2e7 batches (the
BO-loop settings), first 9 partitions:

| partition | boxes | 1 GPU | 8 GPUs | speed-up |
|---:|---:|---:|---:|---:|
| 7 | 38.6M | 22.0 s | 3.0 s | 7.3x |
| 8 | 151.7M | 86.1 s | 11.2 s | 7.7x |

The largest per-GPU memory pool is ~53 GB either way: at 2e7 boxes per bounds
chunk, the bound computation's own workspace dominates until the frontier is
much larger, and it is per GPU. With LHS sampling (2^10 points per box) and the
default `predict_batch_size`, active boxes are sampled 256 at a time. That run
is host-bound (GPUs ~25% busy, 2.5x on 8 GPUs) until `predict_batch_size` is
raised.

### Full BO loop on 8 GPUs (jobs 677272 EI, 677273 logEI)

Same configs, one node with 8 H200s. Iterations 1-14 (EI) and 1-6 (logEI)
reproduce the single-GPU runs above exactly.

| iteration | EI: partitions | EI: peak targets | EI: boxes created | logEI: partitions | logEI: peak targets | logEI: boxes created |
|---:|---:|---:|---:|---:|---:|---:|
| 7  | | | | 35 | 106,232,120 | 12.9B |
| 8  | | | | 32 | 19,955,889 | 1.7B |
| 9  | | | | 34 | 3,339,547 | 0.27B |
| 10 | | | | 43 | 5,574,747 | 0.39B |
| 11 | | | | 31 | 17,540,932 | 1.4B |
| 12 | | | | 42 | 1,039,073,769 | 114B |
| 13 | | | | 24 | 8,253 | 0.5M |
| 14 | | | | out of memory after 14 | 1.50B (growing 2x/partition) | 24.2B so far |
| 15 | 31 | 1,235,030,338 | 142B | | | |
| 16 | 35 | 479,606,405 | 39.8B | | | |
| 17 | 27 | 561,924,775 | 47.7B | | | |
| 18 | 47 | 970,119,192 | 86.4B | | | |
| 19 | 32 | 1,064,590,869 | 99.9B | | | |
| 20 | out of memory after 15 | 2.11B (growing 2x/partition) | 28.8B so far | | | |

EI finished 19 of 20 iterations in 16.5 h and logEI 13 in 6.5 h. Every
finished iteration converged. They ran out of memory in the bounds workspace
(EI) and while gathering one GPU's 434M targets (logEI; targets are balanced by
children at the split but end up uneven).

More memory would not finish these iterations. When they stopped, the largest
bound was still far above the incumbent and shrank only ~1.5x per partition
while the targets doubled:

| | max bound | incumbent | true max (`max_ei_probe.py`) | termination needs bound below |
|---|---:|---:|---:|---:|
| EI, iteration 20 | 1.505 | 0.0320 | 0.0426 | ~0.13-0.14 |
| logEI, iteration 14 | 0.358 | -1.297 | -1.108 | ~-1.0 |

`max_ei_probe.py` rebuilds the iteration's GP from the log (the refit kernel
matches the logged one) and maximizes EI with 257 L-BFGS-B starts. The
maximizers lie on the domain boundary (several coordinates at 0 or 1). The
incumbent explains little of the gap (0.19 of 1.66 nats for logEI; for EI,
even the true max would leave the bound ~35x too high). The bounds on the
remaining boxes are the bottleneck, not the incumbent.

## Tighter sigma bound (Cauchy-Schwarz)

`bound_tightness.py` takes a random sample of the partition-9 frontier of the
two iterations that ran out of memory: 256 random targets of partition 8 and
their children, about 4,400-4,700 boxes. For each box it compares each bound
with the max EI over the box, found by projected Adam from 8 starts. "Gap" is
log(bound / true max), in nats. "Pruned" is the share of boxes whose bound is
within `epsilon_ei` of ExactBO's incumbent at that partition.

| bound | EI it. 20: gap median | EI it. 20: pruned | logEI it. 14: gap median | logEI it. 14: pruned | time per box |
|---|---:|---:|---:|---:|---:|
| current (interval and AutoBound degree 2) | 8.74 | 79.8% | 7.82 | 75.6% | 4.4 us |
| AutoBound degree 3 | 5.44 | 95.7% | 5.35 | 94.3% | 41 us |
| exact mu range, current sigma bound | 8.67 | 86.3% | 7.81 | 78.8% | |
| current mu bound, exact sigma range | 1.87 | 99.7% | 1.17 | 99.7% | |
| exact mu and sigma ranges | 1.52 | 100% | 0.98 | 100% | |
| **current + Cauchy-Schwarz Q bound** | **2.60** | **99.6%** | **1.92** | **99.7%** | **5.0 us** |

The upper bound on sigma, from Q's lower bound, is what was loose. The mean's
bound costs little, and so does bounding mu and sigma separately rather than
jointly: exact ranges of both would prune every sampled box. A joint bound
fails outright on every box. It used a Taylor model of (mu, sigma), which is
valid because EI is jointly convex in (mu, sigma), but AutoBound's enclosure
of sigma = sqrt(sigma_f^2 - Q) is undefined once Q's enclosure reaches
sigma_f^2.

The new bound uses Cauchy-Schwarz in the (K + s_n^2 I)^-1 inner product:
(k^T w)^2 <= Q (w^T K w) for any w. With w = K^-1 k(x0) at the box center,
Q(x) >= (k(x)^T w)^2 / Q(x0). It is exact at x0, and only the linear function
k(x)^T w needs a lower bound over the box, which is bounded like the mean (sign
split of the kernel-entry bounds and a degree-2 AutoBound enclosure). It is
intersected with the existing Q bounds (`bounds.q_cs_weights`, `q_cs_lower`).
No sampled box had a sigma bound below its largest sampled sigma; the smallest
margin was 0.16%. `../experiment3/test_bounds.py` checks the bound against
dense sampling at five box depths.

With it, the two iterations converge (`partition_from_log.py`, 8 H200s,
`results/partition_from_log.jsonl`):

| iteration | before | with the Cauchy-Schwarz bound | true max (`max_ei_probe.py`) |
|---|---|---|---:|
| EI 20 | out of memory after 15 partitions (2.1B targets) | 35 partitions, 60 s, peak 2.2M targets, incumbent 0.03583 | 0.04261 |
| logEI 14 | out of memory after 14 partitions (1.5B targets) | 30 partitions, 29 s, peak 606k targets, incumbent -1.1471 | -1.1076 |

Both incumbents are within `epsilon_ei` of the true max, as the termination
test certifies. The bound changes which boxes are pruned, so the earlier
tables in this README (computed without it) no longer reproduce exactly.

### Full exact BO loop with the bound (jobs 692294 EI, 692295 logEI)

Same configs as above (`configs/`), one H200 each. Both 20-iteration loops
complete, and every partition search ends on the termination test (20 of 20
per run, no "not certified" warnings). Within the `epsilon_ei` tolerance the
search may return a different point than before, so the BO trajectories differ
from the earlier runs from iteration 1 on.

| iteration | EI: partitions | EI: peak targets | logEI: partitions | logEI: peak targets |
|---:|---:|---:|---:|---:|
| 1  | 34 | 7,148     | 37 | 13,827 |
| 2  | 33 | 44,605    | 34 | 60,303 |
| 3  | 37 | 5,518     | 47 | 24,671 |
| 4  | 38 | 16,888    | 44 | 567,808 |
| 5  | 29 | 64,790    | 42 | 5,550 |
| 6  | 31 | 5,605     | 36 | 96,859 |
| 7  | 34 | 424,740   | 43 | 16,473 |
| 8  | 25 | 50,020    | 31 | 39,837 |
| 9  | 48 | 111,944   | 33 | 86,987 |
| 10 | 31 | 93,213    | 34 | 694,224 |
| 11 | 36 | 80,613    | 44 | 52,555 |
| 12 | 27 | 3,652     | 39 | 103,924 |
| 13 | 39 | 4,985     | 33 | 810,340 |
| 14 | 29 | 3,399     | 25 | 625,468 |
| 15 | 29 | 379,485   | 39 | 833,440 |
| 16 | 26 | 834,713   | 36 | 1,757,444 |
| 17 | 23 | 281,913   | 41 | 2,311,079 |
| 18 | 49 | 1,895,266 | 40 | 45,280,778 |
| 19 | 41 | 407,547   | 34 | 7,551,590 |
| 20 | 21 | 7,497     | 25 | 72,116 |

| run | wall time | best y (y* = 3.36) |
|---|---:|---:|
| EI | 7.4 min | 8.23 |
| logEI | 2.1 h (most of it in iteration 18) | 6.99 |

Before the bound, the EI loop ran out of 8 x 141 GB in iteration 20 after 16.5 h,
and logEI in iteration 14 after 6.5 h. The largest frontier is now ~45M targets
(logEI iteration 18); every other iteration stays below 8M.

## EI or log EI?

`log_h` (`tamubo/acquisition_functions/ei.py`) follows LogEI (Ament et al.,
NeurIPS 2023). For z > -1 it evaluates log(phi(z) + z Phi(z)) directly. For
-100 <= z <= -1 it uses log phi(z) + log1mexp(log(|z| erfcx(|z|/sqrt 2)) +
log sqrt(pi/2)). Below -100 it uses the asymptotic series. Against 50-digit
mpmath its relative error is below 2.6e-15 for z in [-1e6, 10]. Naive
log(phi + z Phi) is already wrong at z = -38 (-722.9 instead of -730.2) and -inf
below -39.

log is monotone, so for the same relative tolerance both acquisitions prune
the same boxes. What differs is `epsilon_ei`. With `"ei"` it is absolute, in the
GP's standardized units, and the final EI incumbents of the EI run range from
0.06 to 0.80 (`Incumbent:` in the logs). `epsilon_ei = 0.1` therefore certifies
EI only to within 12-160% of the best value, and loosest when EI is small, late
in BO. With `"logei"`, 0.1 means within 10.5% in every iteration, which costs
more boxes. That is why logEI hits the spikes earlier. Log EI also stays finite
and ordered where EI underflows to 0 (z < -38), where plain EI can no longer
rank boxes or sampled points.
