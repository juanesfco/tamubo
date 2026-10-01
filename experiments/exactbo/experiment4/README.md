# Experiment 4: Stress Test of the Tighter Bounds on TAMU Vision

Experiment 3 tightened ExactBO's bounds (AutoBound Taylor enclosures, exact
sigma/EI ranges, logEI, global incumbent) and tested them on one DGX Spark
(GB10). This experiment runs that code on TAMU Vision (x86_64, 8x H200 141 GB
per node, CUDA 12.9) to check correctness, multi-GPU behaviour, memory limits,
the full BO loop and other problems/settings. All runs are uncapped (no
`max_target_boxes`) unless stated.

## Files

- `stress_partitioning.py`: one `exactbo_partitioning` call on the
  experiment-3 GP with every knob exposed (`--n-gpus`, `--box-sampling`,
  `--autobound-degree`, ...). Appends rows per partition to a CSV and one JSON
  line per run (target counts, chosen point, incumbent, peak GPU memory);
  `--progress` prints each box split with memory, to see where a run fails.
- `check_devices.py`: checks that CuPy and JAX map every GPU to the same device
  and that `taylor_mu_q_bounds` called from a worker thread on GPU i returns
  results on GPU i, equal to GPU 0's.
- `configs/`: experiment-2 configs for the full BO loop (results go to `results/`).
- `vision_*.sbatch`: one Slurm job per step (submit from the repository root);
  logs go to `logs/` (git-ignored).

Environment: `envs/tamubo_exactbo` (pip, `envs/exactbo.txt`) with
`jax[cuda12]==0.8.2` and `autobound==0.1.7`, `module load CUDA/12.9.1`.

## Step 1: correctness on one H200 (`vision_correctness.sbatch`)

- CuPy 14.2.0 and JAX 0.8.2 both see the H200; `test_bounds.py`: 24 passed.
- The four experiment-3 partitioning runs reproduce the GB10 target counts
  exactly, partition by partition, and the same incumbents (0.380, 0.380,
  0.445, 0.477); `results/partition_counts_vision_1gpu.csv`.
- The H200 is ~6-14x faster than the GB10:

| run | GB10 | H200 |
|---|---:|---:|
| interval + EI, to partition 5 | 514 s | 79 s |
| AutoBound + logEI, to partition 7 | 990 s | 89 s |
| AutoBound + EI, to partition 8 | 2563 s | 182 s |

## Step 2: multiple GPUs (`vision_multigpu.sbatch`)

`check_devices.py` passes: CuPy GPU i maps to JAX `cuda:i` (by PCI bus id),
and `taylor_mu_q_bounds` called from a worker thread on GPU i returns its
outputs on GPU i, bit-identical to GPU 0's.

1 vs 2 GPUs (`results/multigpu_runs.jsonl`): in every run the target counts,
chosen point and incumbent are bit-identical to 1 GPU.

| run | node | 1 GPU | 2 GPUs | speedup |
|---|---|---:|---:|---:|
| AutoBound + EI, to partition 8 | dgx007 (job 637266) | 179.1 s | 100.0 s | 1.79x |
| AutoBound + logEI, to partition 7 | dgx007 (job 637266) | 86.3 s | 51.7 s | 1.67x |
| AutoBound + EI, to partition 8 | dgx026 (`profile_sharding.py`, job 637207) | 193.8 s | 99.1 s | 1.96x |
| AutoBound + EI, to partition 8 | dgx004 (job 637049) | 193.0 s | 148.2 s | 1.30x |
| AutoBound + logEI, to partition 7 | dgx004 (job 637049) | 100.3 s | 90.8 s | 1.10x |

The dgx004 runs are outliers (same allocation, other nodes reproduce
1.7-2.0x); Vision nodes are shared with other jobs. Time breakdown on dgx026
(`profile_sharding.py`), 1 GPU -> 2 GPUs:

- box sampling (2^d = 1024 LHS points per box): 122.9 s -> 61.9 s; both GPUs
  busy 61.8 s, so the split is even;
- EI bounds incl. AutoBound: 61.6 s -> 34.8 s (GPU0 32.6 s, GPU1 25.1 s; the
  small early partitions stay on one GPU by design);
- box splitting 0.2-0.3 s, everything else on the main GPU 2-9 s.

On the second GPU's first use JAX compiles its programs again (~3 s), visible
in partition 4.

1/2/4/8 GPUs on one node (dgx080, job 635625): the device check passes on all
8 GPUs and every run is bit-identical to 1 GPU, but scaling flattens after 2:

| GPUs | AutoBound + EI, to partition 8 | AutoBound + logEI, to partition 7 |
|---:|---:|---:|
| 1 | 185.2 s | 91.9 s |
| 2 | 102.5 s (1.81x) | 54.8 s (1.68x) |
| 4 | 93.7 s (1.98x) | 60.2 s (1.53x) |
| 8 | 86.0 s (2.15x) | 56.8 s (1.62x) |

The heaviest partition scales better (EI partition 8: 114.1 s -> 44.3 s on 8
GPUs, 2.6x). Fixed costs cap the total: ~10 s of JAX compilation (partitions 0
and 3, plus ~3 s when each extra GPU is first used) and early partitions too
small to shard.

JAX persistent compilation cache (`vision_jax_cache.sbatch`, job 642811, 2
GPUs, same run twice against a fresh `JAX_COMPILATION_CACHE_DIR`): 113.4 s cold
-> 88.5 s warm, identical counts. Partition 0 drops from 12.3 s to 1.0 s and
partition 3 from 5.1 s to 0.5 s; later partitions are unchanged. The cache
holds 2 entries (the 512- and 8192-box AutoBound programs) that every GPU
reuses, so even the cold run no longer pays the second GPU's ~3 s compile.
`experiments/exactbo/experiment2/submit_vision.sbatch`, `vision_multigpu.sbatch`
and the development benchmark now set
`JAX_COMPILATION_CACHE_DIR=${HOME}/.cache/jax_compilation_cache` (unless
already set), so every job after the first starts warm.

## Step 3: memory, pushing past partition 8 (`vision_memory.sbatch`)

AutoBound + EI, uncapped, one H200 (job 637050). It completes partition 10,
two partitions past the GB10's limit, and runs out of memory splitting into
partition 11:

| partition | targets | children (next partition's boxes) | CuPy pool before split | time at split |
|---:|---:|---:|---:|---:|
| 8  | 7,213,648  | 151,486,608 | 41.7 GB  | 194 s |
| 9  | 16,167,412 | 339,515,652 | 114.4 GB | 447 s |
| 10 | 31,828,811 | 668,405,031 | 138.6 GB | 958 s (out of memory: +53.5 GB with 121 GB allocated) |

Growth per partition keeps falling (2.6x, 2.2x, 2.0x over partitions 8-10),
but the box arrays live on the main GPU only, so more GPUs do not raise this
limit: the next step needs box storage spread over GPUs, or children that are
not all materialized at once. JAX's own peak stays at ~1.1 GB.

On 8 GPUs (job 636708) the run stops at the same split with the same counts,
2.6x sooner (375 s vs 958 s). `nvidia-smi` peaks: GPU 0 ~111 GB, GPUs 1-7
~14 GB each.

## Step 4: full BO loop (experiment 2, `configs/`)

problem2d, `X0=32`, seed 0, two iterations, experiment-2 settings
(`epsilon_X=1e-5`, `epsilon_ei=0.1`, `max_target_boxes=1e7`), one GPU:

| iteration | old Vision run (before experiment 3) | logEI + AutoBound | EI + AutoBound |
|---:|---|---|---|
| 1 | y = -1.9038, EI 0.216, 0.44 s | y = -1.9043, EI 0.196, 7.9 s | y = -1.9043, EI 0.196, 8.0 s |
| 2 | y = -0.0161, EI 0.306, 0.30 s | y = 0.1381, EI 0.059, 7.3 s | y = 0.1381, EI 0.059, 4.1 s |

logEI and EI choose the same points; the cap never triggered, so both runs are
exact. The iteration time is mostly JAX compilation (AutoBound costs more than
it saves in 2-d).

problem10d, the experiment-2 config as committed (`predict_batch_size` =
`bounds_batch_size` = 1e8, LHS sampling, `max_target_boxes=1e7`), 8 GPUs
(jobs 636709, 636710): both run out of memory in the first iteration, in box
sampling (a 25.6 GB GP-posterior allocation = 1e8 points x N=32 x 8 bytes) with
~23M active boxes: logEI at partition 9 (cap never triggered), EI at partition
10 (the cap had cut partition 9 from 16.2M to 10M targets, so that run was no
longer exact). The 1e8 batch size is the problem; experiment 2 needs a smaller
`predict_batch_size` (e.g. 2e7) or `box_sampling="center"`.

Rerun with `box_sampling="center"` (now the library default) and
`predict_batch_size = bounds_batch_size = 2e7`, 3 iterations, 2 GPUs (jobs
642812, 642813; `configs/bo_problem10d_*.json`): both complete, and every
partitioning loop terminates normally, but the `max_target_boxes=1e7` cap
fired in 50 of 119 partitions (logEI) and 40 of 119 (EI), so the runs are not
exact.

| iteration | logEI: y, EI | time | EI: y, EI | time |
|---:|---|---:|---|---:|
| 1 | 26.564, 26.42 | 1222 s | 26.008, 26.40 | 952 s |
| 2 | 10.476, 18.29 | 1387 s | 19.868, 18.38 | 1334 s |
| 3 | 29.816, 31.27 | 1802 s | 24.931, 21.93 | 1327 s |

`y* = 3.36`. logEI and EI pick different points from iteration 1 on (different
pruning tolerance, and the cap keeps different boxes). For comparison, the
pre-experiment-3 code (interval bounds, center sampling, same cap, 8 GPUs)
chose y = 26.673 in iteration 1 in 144 s, against 952-1222 s here on 2 GPUs
(not a like-for-like comparison: 8 vs 2 GPUs). On these capped searches each
partition bounds up to 2.1e8 boxes, and AutoBound's per-box cost dominates.

## Step 5: other problems and settings (`vision_other_settings.sbatch`)

One H200, logEI + AutoBound unless stated, `epsilon_X=1e-5`, at most 100
partitions (`results/other_runs.jsonl`, `results/other_partition_counts.csv`).

To convergence, AutoBound vs interval bounds (seeds 0, 1, 2):

| problem | X0 | AutoBound | interval |
|---|---:|---|---|
| problem2d | 4  | converges, 16-18 partitions, 8-33 s | converges, 16-19 partitions, 1.5 s |
| problem2d | 32 | converges, 17-19 partitions, 5-9 s | converges, 18 partitions, 1.6 s |
| problem5d | 8  | converges, 27-31 partitions, 14-15 s, <= 1.5 GB | seed 0: converges, 30 partitions, 40 s, 145 GB; seeds 1-2: out of memory |
| problem5d | 32 | out of memory (all seeds) | out of memory (all seeds) |

- 2-d: both converge to the same incumbent; AutoBound is slower because of JAX
  compilation (the 33 s run includes the first compiles).
- 5-d, `X0=8`: only AutoBound converges reliably. Its targets peak at ~107k
  around partition 10 and then shrink; the interval run reaches 8.9M targets at
  partition 11.
- 5-d, `X0=32`: nothing converges. Follow-up (`vision_5d_followup.sbatch`,
  seed 0): ~96% of children stay targets through split 6; logEI fails at split
  10 (178M targets -> 2.0B boxes), EI also fails at split 10 (123M targets).
  The absolute EI tolerance prunes only 15-25% more than logEI, so the limit is
  bound tightness, not the tolerance definition.

10-d, AutoBound + EI to partition 8:

| run | targets at partition 8 | incumbent | time |
|---|---:|---:|---:|
| seed 0, LHS (step 1) | 7,213,648 | 0.477 | 182 s |
| seed 1, LHS | 5,350,713 | 0.858 | 119 s |
| seed 2, LHS | 7,463,506 | 1.121 | 162 s |
| seed 0, `box_sampling="center"` | 14,010,996 | 0.351 | 91 s |

None converges by partition 8. Center sampling finds a lower incumbent, so it
prunes less (2x the targets) but is faster per box (1 posterior evaluation
instead of 1024).

AutoBound degree 3 vs 2: problem5d (`X0=8`) keeps ~15% fewer targets in the
middle partitions but converges in the same 27 partitions to the same
incumbent, 17 s vs 13.5 s; problem2d gains similarly little.
