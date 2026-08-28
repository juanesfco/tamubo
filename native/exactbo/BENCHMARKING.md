# Benchmarking native ExactBO on DGX Spark and VISION

This guide walks through benchmarking `exactbo_partitioning` (the C++/CUDA/MPI
executable in this directory) on two platforms: the local NVIDIA DGX Spark and
the Texas A&M System VISION supercomputer (an NVIDIA DGX SuperPOD,
https://vision.tamus.edu/, docs at https://docs.vision.tamus.edu/). It also
covers how to measure *actual* GPU FLOPs and memory allocation, not just
wall-clock time, since Figure 3 of the ExactBO paper only reports wall-clock
`t_n` and this native backend is a different (C++/CUDA/MPI, not
cuPyNumeric/Legate) implementation of the same algorithm.

Every command below was exercised on the DGX Spark during preparation of this
guide (build, run, `nsys profile`, reading a `.ncu-rep`); the VISION section is
based on https://docs.vision.tamus.edu/specifications,
https://docs.vision.tamus.edu/slurm/, https://docs.vision.tamus.edu/onboarding/,
and https://docs.vision.tamus.edu/using-containers/, since this session has no
VISION account to test against directly. Verify the VISION-specific flags
against current docs before a real run — that portal was in "Controlled Beta"
when this guide was written (2026-08-28).

## 1. Platform summary

| | DGX Spark (this machine) | VISION (per platform docs) |
|---|---|---|
| GPU | 1x NVIDIA GB10, compute capability 12.1 (`sm_121`) | 8x NVIDIA H200 per node, 144 GB HBM3e each, compute capability 9.0 (`sm_90`), NVLink (7.2 TB/s bidirectional GPU-GPU) |
| CPU | Grace (aarch64), 20 cores (Cortex-X925/A725) | 2x Intel Xeon Platinum 8480C, 112 cores/node |
| Memory model | **Unified**: 121 GB shared CPU+GPU (`nvidia-smi` reports `memory.total=N/A`; use `cudaMemGetInfo`, which this binary already calls) | **Discrete**: 144 GB/GPU HBM3e + 2 TB DDR5 host RAM/node (`nvidia-smi` memory queries work normally) |
| CUDA / driver | CUDA 13.0 (`nvcc V13.0.88`), driver 580.126.09 | Unknown ahead of time — check `nvcc --version` / `nvidia-smi` on an allocated node |
| Interconnect | single node, no MPI network needed | 400G InfiniBand between nodes, 100G/40G Ethernet |
| Scheduler | none (interactive) | SLURM, QoS `high`/`standard`(default)/`low`/`scavenger`, max 48h walltime, max 240 GPUs/job |
| Software delivery | native OS packages (`nvcc`, `mpicxx`, `nsys`, `ncu` already on `PATH`) | **Pyxis/Enroot containers only** — VISION explicitly does not support Apptainer, Singularity, or native Docker |
| Local scratch | `/` (3.7 TB NVMe, this session) | `/raid` (30 TB/node local NVMe), `$SCRATCH` on 8 PB DDN EXAScaler |

The unified-vs-discrete memory difference matters for this codebase
specifically: `IMPLEMENTATION.md` notes the box-storage host-memory-limit
logic "counts host staging twice on integrated or host-page-table/coherent
GPUs such as GB10." That doubling should trigger on DGX Spark and should
**not** trigger on VISION's discrete H200s — comparing the two
`host_box_limit=...` values the binary prints at startup across platforms is
a good sanity check that this code path is actually platform-aware, not just
documented as such.

## 2. Build

Both platforms use the same CMake project; only the target compute
architecture and (on VISION) the delivery mechanism differ.

```bash
cmake -S native/exactbo -B native/exactbo/executables/build \
  -DCMAKE_CUDA_ARCHITECTURES=<arch>
cmake --build native/exactbo/executables/build -j"$(nproc)"
```

- DGX Spark: omit `-DCMAKE_CUDA_ARCHITECTURES` (defaults to `native`, which
  correctly resolves to `121` on this machine — confirmed by a clean build in
  this session) **or** pass `-DCMAKE_CUDA_ARCHITECTURES=121` explicitly.
- VISION: pass `-DCMAKE_CUDA_ARCHITECTURES=90` explicitly. `native` requires
  `nvcc` to see a live GPU at configure time, which login/build nodes
  typically don't have — build inside a GPU-allocated job instead (see §4),
  or just hardcode `90` for the H200's Hopper architecture.

This produces `exactbo_partitioning`, `exactbo_bounding`, and
`exactbo_box_partition` under `native/exactbo/executables/`.

## 3. Python environment

`exactbo_workflow.py` needs `numpy` and `scikit-learn` (`scipy` is pulled in
transitively). There is no committed lockfile for this specific script beyond
the broader `envs/exactbo.txt` env; a minimal venv is enough for benchmarking
purposes:

```bash
python3 -m venv /path/to/venv
/path/to/venv/bin/pip install numpy scipy scikit-learn
```

## 4. Matching the paper's Section 5.3 protocol

The paper's methodology is: `N0 = 2^ceil(d/2)` Latin-hypercube initial
points, 10 random seeds, 10 BO iterations, per-`d` `epsilon_X`/`epsilon_EI`
settings. Two things to know before reusing that protocol against this native
executable:

1. **`exactbo_workflow.py` has no seed flag and no LHS-sized initial design
   built in.** Its `--example` presets use small *fixed* `X0` arrays (4 fixed
   points for `minimization_2d`; a single zero point for `problem5d` /
   `problem10d`) — good enough as a smoke test, but not the paper's `N0`
   protocol. For a real seed sweep, generate `X0`/`y0` `.npy` files yourself
   and pass them via `--x0`/`--y0` with `--example none`.
2. **The built-in `--example problem5d` is not the paper's Biggs benchmark.**
   Its objective (`make_problem5d_data`/`problem5d_objective` in
   `exactbo_workflow.py`, mirrored in
   `experiments/exactbo/experiment2/problems.py`) is a different nonlinear
   least-squares function on `[-1, 1]^5`, not the Biggs EXP6 function
   (Eq. 31/35–37) on `[-10, 10]^5` described in the paper's Appendix A.2. See
   the correctness notes at the end of this document. If you want the
   benchmark the paper actually describes, supply the Biggs objective
   yourself (script below); if you want to reproduce the paper's *reported*
   5D numbers as generated, `experiments/exactbo/experiment2/problems.py`'s
   `problem5d` is the one that actually produced them (still on `[-1,1]^5`,
   still not Biggs).

Seed/design generator (`gen_x0.py`, save anywhere outside the repo, e.g. your
scratch dir):

```python
import argparse, numpy as np
from scipy.stats import qmc

p = argparse.ArgumentParser()
p.add_argument("--d", type=int, required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--lower", type=float, nargs="+", required=True)
p.add_argument("--upper", type=float, nargs="+", required=True)
p.add_argument("--out-prefix", required=True)
args = p.parse_args()

n0 = 2 ** int(np.ceil(args.d / 2))
sampler = qmc.LatinHypercube(d=args.d, seed=args.seed)
unit = sampler.random(n=n0)
lower, upper = np.array(args.lower), np.array(args.upper)
X0 = qmc.scale(unit, lower, upper)
np.save(f"{args.out_prefix}_X0.npy", X0.astype(np.float64))
np.save(f"{args.out_prefix}_bounds.npy", np.stack([lower, upper], axis=1).astype(np.float64))
print(f"d={args.d} seed={args.seed} n0={n0} -> {args.out_prefix}_X0.npy")
```

True Biggs objective (`biggs_objective.py`, matching Eq. 35–37 exactly,
domain `[-10, 10]^5`, `y* = 0`):

```python
import numpy as np

def objective(X):
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    t = 0.1 * np.arange(1, 14)  # t_i = 0.1*i, i=1..13
    y = np.exp(-t) - 5.0 * np.exp(-10.0 * t) + 3.0 * np.exp(-4.0 * t)
    r = (x3[:, None] * np.exp(-np.outer(x1, t))
         - x4[:, None] * np.exp(-np.outer(x2, t))
         + 3.0 * np.exp(-np.outer(x5, t)) - y[None, :])
    return np.sum(r * r, axis=1)
```

## 5. Running the sweep

For each `d` in `{2, 5, 10}`, each seed in `{0..9}`, and each `epsilon_EI` the
paper reports for that `d` (2D: `0.1`, `0.01`; 5D: `1`, `0.1`; 10D: `1`):

```bash
# example: d=5 (true Biggs), seed=3, epsilon_EI=0.1
python3 gen_x0.py --d 5 --seed 3 \
  --lower -10 -10 -10 -10 -10 --upper 10 10 10 10 10 \
  --out-prefix /scratch/$USER/exactbo_bench/d5_seed3

VENV/bin/python native/exactbo/exactbo_workflow.py \
  --example none \
  --objective biggs_objective.py:objective \
  --x0 /scratch/$USER/exactbo_bench/d5_seed3_X0.npy \
  --bounds /scratch/$USER/exactbo_bench/d5_seed3_bounds.npy \
  --workdir /scratch/$USER/exactbo_bench/d5_seed3_run \
  --native-build-dir native/exactbo/executables \
  --mpi-ranks <ranks> \
  --max-iters 10 \
  --epsilon-x 0.1 --epsilon-ei 0.1 \
  --max-partitions 100 \
  --device-batch-rows 4096 \
  --verbose \
  --log-file /scratch/$USER/exactbo_bench/d5_seed3_run/run.log
```

For `d=2`, `--example minimization_2d` already matches the paper exactly (its
`X0`/objective/bounds/`D` constant were checked against Eq. 30/34 and match)
— you only need the custom `--x0`/seed generator if you want the paper's
`N0=2` LHS design instead of the built-in 4-point grid. For `d=10`,
`--example problem10d` matches the paper's Appendix A.3 data and `y*`
exactly (checked numerically: the code's computed `y*` reproduces
`3.363385360111855` to full double precision) — same caveat about `X0`.

`--mpi-ranks` on DGX Spark should stay at `1` (one GPU). On VISION, set it to
the number of GPUs you request per node (up to 8) — the code picks a GPU
"from its node-local MPI rank," so `--mpi-ranks 8` with `--gres=gpu:8`
exercises all GPUs on one node; going beyond one node exercises the
InfiniBand fabric through MPI's box-list partitioning across ranks.

`--log-file` captures Python + native stdout/stderr together; parse the
`partition N boxes=... max_ei_hi=... best_ei=... target=...` lines it already
prints (one per partition step) plus the final `iteration=... x_next=...
y_next=... best_y=...` line for regret/best-objective curves matching the
paper's `R_n` and `y*_n` metrics (Eq. 33 and the min-so-far series). Wall time
per BO iteration (`t_n`) — time the `exactbo_partitioning` subprocess launch
directly (see the `launch: native/exactbo/executables/exactbo_partitioning
...` line the script prints) rather than the whole Python process, to exclude
GP-fit and objective-evaluation time from the acquisition-optimization
number, matching how the paper defines `t_n`.

## 6. Measuring actual GPU FLOPs

The three CUDA kernels doing the work are `ei_hi_bounds_kernel` (interval EI
upper bound, Eq. 9–15), `sample_best_kernel` (center-point EI sampling after
this session's modification), and `split_dense_boxes_kernel` (DIRECT-style
split). All three run at 128 threads/block currently.

### Nsight Systems (kernel-level timing — always works, no special permissions)

```bash
nsys profile --trace=cuda,mpi,osrt --output=exactbo_profile --force-overwrite=true \
  mpirun -np <ranks> native/exactbo/executables/exactbo_partitioning \
    --input <partitioning_input.bin> --output <out.bin> \
    --device-batch-rows 4096 --box-storage host --verbose

nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_api_sum exactbo_profile.nsys-rep
```

This was run in this session against a real `partitioning_input.bin` and
produced a per-kernel time breakdown with counts that exactly track
Algorithm 1's structure: one `ei_hi_bounds_kernel` launch and two
`sample_best_kernel` launches per partition step (the `S`-set/`tau1` sample
and the `A`-set/`tau2` sample), one `split_dense_boxes_kernel` launch per
partition step that doesn't terminate. That launch-count pattern is itself a
useful correctness check to run once per platform.

### Nsight Compute (achieved FLOP/s and memory throughput)

```bash
ncu --set full --target-processes all \
  --kernel-name regex:"ei_hi_bounds_kernel|sample_best_kernel|split_dense_boxes_kernel" \
  --export exactbo_ncu --force-overwrite \
  mpirun -np <ranks> native/exactbo/executables/exactbo_partitioning \
    --input <partitioning_input.bin> --output <out.bin> --device-batch-rows 4096 --box-storage host
```

Then either read `--page details` on the exported report, or pull specific
metrics as CSV (works even without live GPU counter access, since it's
replaying an already-captured report):

```bash
ncu --import exactbo_ncu.ncu-rep --csv --metrics \
  gpu__time_duration.sum,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__sass_thread_inst_executed_op_dadd_pred_on.sum,\
  sm__sass_thread_inst_executed_op_dmul_pred_on.sum,\
  sm__sass_thread_inst_executed_op_dfma_pred_on.sum \
  > exactbo_ncu.csv
```

All three kernels are double-precision (interval arithmetic needs it), so
compute achieved FLOP/s as:

```
total_flops = dadd_sum + dmul_sum + 2 * dfma_sum   # an FMA is 2 FLOPs
achieved_gflops_per_s = total_flops / (gpu__time_duration.sum in seconds) / 1e9
```

and compare `sm__throughput...pct_of_peak_sustained_elapsed` /
`dram__throughput...pct_of_peak_sustained_elapsed` to see whether each kernel
is compute-bound or memory-bound — `ei_hi_bounds_kernel` (dense per-training-point
loop, Eq. 9–14) should be far more compute-heavy per byte moved than
`split_dense_boxes_kernel` (mostly writes new box bounds, little arithmetic).

**Permission note (hit and confirmed in this session):** `ncu` failed with
`ERR_NVGPUCTRPERM — the user does not have permission to access NVIDIA GPU
Performance Counters` when run as an unprivileged user in this environment.
Fix one of:
- run via `envs/dgx/dev.sh profile` (already in this repo — it launches the
  dev container with `--cap-add=SYS_ADMIN --user root`, which is exactly what
  `ncu` needs), or
- `sudo ncu ...` / run the whole benchmark as root, or
- persist `NVreg_RestrictProfilingToAdminUsers=0` for the `nvidia` kernel
  module (`/etc/modprobe.d/nvidia-profiling.conf`, then reload the module or
  reboot) if you control the machine.

On VISION, request `--cap-add=SYS_ADMIN` isn't something Pyxis exposes the
same way as Docker; ask `help@vision.tamus.edu` whether GPU performance
counters are enabled for job containers, or fall back to `nsys` (which does
not need elevated counter access) for the VISION side of the comparison.

**Don't reuse the checked-in `.ncu-rep`/`.nsys-rep` files in
`native/exactbo/profiles/` as a current baseline.** They were captured
against `exactbo_bounding`'s `evaluate_boxes` kernel with a grid/block shape
of `(390625,1,1)`×`(256,1,1)` (`390625*256 = 100,000,000`, matching the
`p1e8` filename prefix and the `BOXES` constant), but the current source has
`THREADS_PER_BLOCK = 128`, which would produce `(781250,1,1)`×`(128,1,1)`.
The checked-in profiles predate that change — regenerate fresh ones rather
than trusting the numbers already in that directory.

## 7. Measuring actual GPU memory allocation

Three complementary sources, because no single one is complete on both
platforms:

1. **Built into the binary**: `print_device_memory()` calls `cudaMemGetInfo`
   and prints `rank N memory [startup] used=... free=... total=...` once,
   at MPI startup. This works correctly on both unified (DGX Spark) and
   discrete (VISION) memory — it's the one number that is directly
   comparable across platforms without translation. It is only printed once
   per run, though, so it won't show peak usage mid-sweep by itself.

2. **External time-series sampling**, run concurrently with the benchmark:
   ```bash
   nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
     --format=csv -l 1 > gpu_mem_timeline.csv &
   SMI_PID=$!
   # ... run the benchmark ...
   kill $SMI_PID
   ```
   On DGX Spark, `memory.used`/`memory.total` report `[N/A]` (confirmed in
   this session — GB10 is unified memory, so use `free -h` /
   `/proc/meminfo` alongside `cudaMemGetInfo` instead). On VISION's discrete
   H200s this query works normally and is the simplest option. For a more
   detailed per-process breakdown, `nvidia-smi --query-compute-apps=pid,used_memory
   --format=csv -l 1` or DCGM (`dcgmi dmon -e 203,204` for FB used/free) if
   available on the target node.

3. **`nsys`'s own memory-operations summary** (`cuda_gpu_mem_size_sum` /
   `cuda_gpu_mem_time_sum` reports, shown above) — gives H2D/D2H copy volume
   and count, useful for confirming the box-batching design (`--device-batch-rows`,
   `--split-batch-parents`) is actually keeping transfers small and bounded
   rather than growing with total box count.

4. **Cross-check against the documented formulas** in `IMPLEMENTATION.md`:
   `EI bounds: 8 * N * (4*n_train + 2*d + 1) bytes`,
   `EI samples: 8 * N * (n_train + 3*d + 1) bytes`,
   `split device bytes = 32 * d * (d + 1) * N parents`, where `N` is
   `--device-batch-rows` (or the chosen split-parent batch). Compute the
   *requested* bytes from these formulas for your `--device-batch-rows` value
   and compare against the *observed* peak from (1)/(2)/(3) — they should be
   close; a large gap would mean either the formulas are stale or something
   else is holding device memory.

## 8. VISION-specific setup

VISION is in controlled beta; access requires an invitation, SSO, Duo, and
Cloudflare One (see https://docs.vision.tamus.edu/onboarding/). Once you have
access:

```bash
# ~/.ssh/config, then:
ssh vision
```

**Containers, not modules.** VISION's documented runtime is Pyxis/Enroot;
Apptainer, Singularity, and native Docker are explicitly unsupported. Build a
container image with CUDA 13 + OpenMPI (this repo already has one at
`envs/dgx/Dockerfile`, currently built for `linux/arm64` for DGX Spark's
Grace CPU) — **rebuild it for `linux/amd64`** for VISION's Xeon nodes:

```bash
PLATFORM=linux/amd64 IMAGE=<your-registry>/tamubo-vision:cuda13-amd64 \
  envs/dgx/dev.sh build
docker push <your-registry>/tamubo-vision:cuda13-amd64
```

Push it somewhere VISION's Pyxis can pull from (your own Docker Hub/GHCR
namespace, or an NGC private registry) — VISION nodes don't have Docker to
build images themselves.

Example `sbatch` script (adapt from `native/benchmarks/matrix/hprc_matrix_benchmark.slurm`,
but note that script targets a different HPRC cluster with `module load
CUDA/...`; VISION's own docs show no module system, only containers):

```bash
#!/bin/bash
#SBATCH --job-name=exactbo-native-bench
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=14
#SBATCH --mem=1500G
#SBATCH --time=04:00:00
#SBATCH --qos=standard
#SBATCH --output=exactbo-bench-%j.out
#SBATCH --container-image=<your-registry>/tamubo-vision:cuda13-amd64
#SBATCH --container-mounts=/scratch/user/$USER/tamubo:/workspace

cd /workspace
cmake -S native/exactbo -B native/exactbo/executables/build -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build native/exactbo/executables/build -j"$(nproc)"
nvidia-smi -L

# single-node, all 8 H200s over NVLink:
mpirun -np "$SLURM_NTASKS" native/exactbo/executables/exactbo_partitioning \
  --input <input.bin> --output <output.bin> --device-batch-rows 4096 --verbose
```

For multi-node MPI scaling (exercising the 400G InfiniBand fabric and this
codebase's file-backed `BoxStore` across nodes — the paper's own Future Work
section calls out scaling "from a single DGX Spark to multi-GPU systems such
as a DGX SuperPOD," and VISION literally is a DGX SuperPOD), increase
`--nodes` and point `--spill-dir` at a path under `$SCRATCH` (the 8 PB DDN
EXAScaler filesystem, visible to every node) rather than node-local `/raid`.

Ask `help@vision.tamus.edu` to confirm before relying on any of this section
against a live allocation — this guide could not be tested against VISION
directly in this session (no account), and Pyxis/GPU-counter-permission
specifics may have changed since 2026-08-28.

## 9. Suggested results schema

Extend the existing convention from `experiments/exactbo/experiment2/experiment_config.json`
(`results_filename: experiment_results_vision.csv`) with one CSV per
platform, one row per (`d`, seed, `epsilon_EI`, BO iteration, partition
step):

```
platform, gpu_model, d, seed, epsilon_x, epsilon_ei, bo_iteration, partition_step,
n_boxes, max_ei_hi, best_ei, wall_time_s, gpu_mem_used_gib, gpu_mem_startup_gib,
sm_throughput_pct, dram_throughput_pct, achieved_gflops_per_s
```

That makes it possible to reproduce the paper's `R_n` / `y*_n` / `t_n` curves
(Figure 3 columns 1–3) plus the two new hardware columns this guide adds.

## 10. Correctness caveats relevant to benchmarking

Findings from reviewing this code against the paper while preparing this
guide (see the separate correctness summary for the full list):

- **`--example problem5d` in this script (and `problem5d` in
  `experiments/exactbo/experiment2/problems.py`) is not the paper's Biggs
  benchmark** — different formula, different domain (`[-1,1]^5` vs. the
  paper's `[-10,10]^5`). Use the Biggs objective script in §4 if you want the
  benchmark the paper describes.
- `exactbo_bounding`'s hardcoded fixture path
  (`data/logs/checkBounding/input10d.bin`) does not exist anywhere in the
  repo and nothing generates it, so that diagnostic cannot currently be run
  as documented in `README.md`. Use `exactbo_partitioning` (driven through
  `exactbo_workflow.py`, or directly as in §6) for profiling instead — it is
  the actual production path and it does run correctly (verified in this
  session, including after the center-sampling change).
- The checked-in `.ncu-rep`/`.nsys-rep` files under `native/exactbo/profiles/`
  predate the current `THREADS_PER_BLOCK = 128` kernel configuration (see
  §6) — do not treat them as a current baseline.
