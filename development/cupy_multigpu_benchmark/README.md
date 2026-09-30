# ExactBO on multiple GPUs (CuPy, TAMU Vision)

Scaling benchmark for `tamubo.exactbo` with `backend="cupy"`: the same search
is run with `n_gpus` = 1, 2, 4, 8 on one Vision node (8x H200), reporting
partitioning time per iteration, speedup over 1 GPU, and whether each run
selects the same points as the 1-GPU run. Objective: Levy on [-10, 10]^d.

```bash
# from the repo root (environment: envs/exactbo.txt, see the repo README)
sbatch development/cupy_multigpu_benchmark/bench_multigpu.sbatch
```

Output: `logs/bench-<jobid>.out` (one `RESULT` line per run) and
`logs/results_<jobid>.csv`. Presets (`bench_multigpu.py`):

- `moderate` — tolerances comparable to experiment 2's published runs.
- `stress` — `epsilon_X=1e-5`, up to 1e7 target boxes, batch size 2e7
  (minutes per iteration on one GPU).

Options: `--box-sampling center` samples EI only at each box center instead of
2^d points; `--batch` overrides the preset's per-GPU batch sizes.

## Results (job 614864, 8x H200, one node)

Partitioning time per iteration (s) and speedup over 1 GPU. Every multi-GPU
run selected exactly the same points as the 1-GPU run.

| Search                       | 1 GPU            | 2 GPUs           | 4 GPUs           | 8 GPUs          | Speedup at 8 GPUs | Peak GPU memory |
|------------------------------|------------------|------------------|------------------|-----------------|-------------------|-----------------|
| 2d moderate, lhs (3 it)      | 0.33, 0.20, 0.16 | 0.15, 0.17, 0.16 | 0.14, 0.14, 0.13 | 0.12, 0.14, 0.13 | ~1.0x            | < 0.1 GB        |
| 5d moderate, lhs (3 it)      | 0.52, 0.48, 0.63 | 0.48, 0.48, 0.66 | 0.47, 0.48, 0.63 | 0.47, 0.48, 0.63 | ~1.0x            | 14 GB           |
| 10d moderate, lhs (3 it)     | 26.0, 48.3, 63.5 | 15.0, 25.4, 33.4 | 10.0, 14.2, 18.6 | 7.8, 8.8, 11.5  | 3.3x, 5.5x, 5.5x  | 20 GB           |
| 5d stress, lhs (1 it)        | 210.4            | 107.9            | 56.9             | 31.7            | 6.6x              | 42-46 GB        |
| 10d stress, center (1 it)    | 280.6            | 149.0            | 82.4             | 50.1            | 5.6x              | 125-135 GB      |

Small searches have too little work per call to spread across GPUs, so
`tamubo.exactbo` keeps each call on as many GPUs as its work justifies
(`_MIN_WORK_PER_GPU` in `run.py`); they run as fast as on one GPU (the >1x
readings at 2d come from a slower 1-GPU baseline, not from sharding). Without
that cutoff (job 614463) extra GPUs made small searches 2-12x slower and gave
lower gains on large ones.

Peak GPU memory is the largest CuPy memory-pool size on any GPU. Only the
per-box work is split; every box, the per-box EI bounds and the masks stay on
the main GPU, so the peak falls only a little with more GPUs (10d stress:
135 GB on 1 GPU, 124 GB on 8, against the H200's 141 GB). 10d stress with
`"lhs"` sampling was not run here; with experiment 2's larger batch sizes (1e8)
it runs out of memory on 1 and 8 GPUs.
