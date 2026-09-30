# Experiment 3: Tighter Bounds for ExactBO Pruning

ExactBO prunes a box once an upper bound on its acquisition value is within
`epsilon_ei` of the incumbent. On the experiment-2 10-d problem the original
bounds pruned nothing through partition 4 (0.6% at partition 5), so the search
refined every box and ran out of memory at partition 6 (78.7M target boxes).
This experiment measures where the bounds were loose and what the tightened
bounds in `tamubo.exactbo` recover:

- **Sigma bound** (`bounds.sigma_bounds`): exact per-entry range of
  `v = L^{-1} k` from the sign split of `L^{-1}` instead of interval forward
  substitution, plus `Q >= ||k_lo||^2 / lambda_max(K + sigma_n^2 I)`.
- **EI bound** (`bounds.ei_bounds`): exact range over the (mu, sigma)
  rectangle, since EI is decreasing in mu and increasing in sigma.
- **AutoBound** (`autobound_bounds.py`, `bound_method="autobound"`):
  degree-2 Taylor enclosures of mu(x) and Q(x) over each box, intersected with
  the interval bounds.
- **logEI** (`acquisition="logei"`): numerically stable log EI for sampling
  and bounding; `epsilon_ei` becomes a relative tolerance.
- **Incumbent**: pruning compares against the best point sampled over all
  partitions, not only the current one.

## Files

- `common.py`: experiment-2 GP (problem, initial design, kernel), DIRECT-like
  test boxes, and a reference posterior
- `bound_tightness.py`: bound overestimate vs box depth; writes
  `results/bound_tightness.csv`
- `run_partitioning.py`: one uncapped `exactbo_partitioning` call; appends
  target boxes, incumbent and time per partition to
  `results/partition_counts.csv`
- `test_bounds.py`: pytest soundness tests; every bound must contain the
  posterior sampled densely inside random boxes (NumPy and CuPy)

## Run

From this directory (the default GP is problem10d, `X0=32`, seed 0, the
first iteration of experiment 2):

```bash
python bound_tightness.py
python run_partitioning.py --acquisition ei --bound-method interval --max-partitions 6
python run_partitioning.py --acquisition logei --bound-method interval --max-partitions 6
python run_partitioning.py --acquisition logei --bound-method autobound --max-partitions 8
python run_partitioning.py --acquisition ei --bound-method autobound --max-partitions 9
python -m pytest test_bounds.py
```

`bound_method="autobound"` needs `jax` and `autobound` (see `envs/README.md`).
`run_partitioning.py` appends to its CSV; delete it to start over.

## Results

DGX Spark (one GB10, 121 GB unified memory), `epsilon_ei = 0.1`,
`epsilon_X = 1e-5`, LHS sampling (1024 points per box), no
`max_target_boxes` cap.

### Bound overestimate vs depth (`results/bound_tightness.csv`)

Median over 256 random boxes per depth; the gap is bound minus the extreme of
2000 posterior samples per box. Partition p produces boxes of depth p to 10p.
Every bound contained every sample.

| depth | mu_lo gap, interval | mu_lo gap, AutoBound | EI_hi, interval | EI_hi, AutoBound | prunable, interval | prunable, AutoBound |
|---:|---:|---:|---:|---:|---:|---:|
| 0  | 135    | 135    | 135    | 135     | 0.00 | 0.00 |
| 5  | 100    | 55.6   | 99.1   | 55.5    | 0.00 | 0.00 |
| 10 | 58.8   | 6.61   | 57.8   | 5.87    | 0.00 | 0.00 |
| 15 | 40.8   | 2.38   | 40.5   | 1.78    | 0.00 | 0.12 |
| 20 | 20.1   | 0.306  | 18.7   | 0.0907  | 0.00 | 0.78 |
| 30 | 6.80   | 0.0348 | 5.34   | 0.0152  | 0.00 | 0.99 |
| 50 | 0.747  | 0.00303 | 0.0124 | 2.4e-20 | 0.82 | 1.00 |

"Prunable" is the share of boxes whose EI upper bound is within 0.1 of the
best sampled EI. The true mean range per box is ~5 at depth 0 and ~0.02 at
depth 50, so the interval mean bound overestimates by up to ~30x. Per-entry
kernel intervals ignore that all entries depend on the same x, and with
mixed-sign alpha (|alpha| up to 12.8 here) that error only shrinks linearly with
the box width. The mean bound dominates EI_hi; the sigma and EI improvements
alone change EI_hi by <1% at depth <= 30.

### Target boxes per partition (`results/partition_counts.csv`)

| partition | original (7f81d2d) | interval, EI | interval, logEI | AutoBound, logEI | AutoBound, EI |
|---:|---:|---:|---:|---:|---:|
| 0 | 1          | 1         | 1         | 1         | 1         |
| 1 | 21         | 21        | 21        | 21        | 21        |
| 2 | 441        | 441       | 441       | 366       | 360       |
| 3 | 9,261      | 9,261     | 9,261     | 4,124     | 3,993     |
| 4 | 194,481    | 194,481   | 194,481   | 34,336    | 31,555    |
| 5 | 4,061,025  | 4,028,405 | 4,037,990 | 213,280   | 186,277   |
| 6 | 78,655,301 |           |           | 991,664   | 831,027   |
| 7 | out of memory |        |           | 3,377,730 | 2,770,874 |
| 8 |            |           |           |           | 7,213,648 |
| time to reach partition 5 | | 514 s | 528 s | 48 s | 43 s |
| incumbent EI at last partition | | 0.380 | 0.380 | 0.445 | 0.477 |

The original column is from commit 7f81d2d: partitions 0-4 were rerun here,
and partitions 5-6 come from the earlier TAMU Vision run (it ran out of GPU
memory splitting partition 6). Without a cap, a partition's target count
multiplies by 2d + 1 = 21 when nothing is pruned. With AutoBound the growth
factor falls to ~9, 6, 4.5, 3.3 and 2.6 over partitions 4-8.

Takeaways:

- The mean bound is the bottleneck. Only AutoBound makes the 10-d search prune:
  −57% of targets at partition 3, −95% at 5 and −98.9% at 6. The interval-only
  variants barely prune.
- logEI with `epsilon_ei = 0.1` is a relative tolerance (~10.5%). With EI near
  0.45 that is stricter than an absolute 0.1, so it keeps ~20% more targets.
- AutoBound costs ~13 µs/box on the GB10 against ~0.4 µs for the interval
  bounds. From partition 4 on it cuts the box count 6-20x, so the run is 11x
  faster to partition 5.
- Not converged: at partition 8 the run still grows 2.6x per partition, and
  splitting into partition 9 (~151M boxes) fills the 121 GB of memory. The
  remaining looseness is in large DIRECT side children, which are still full
  width in 3-4 dimensions: max EI_hi stays at ~8-10 while the incumbent EI is
  ~0.48.
